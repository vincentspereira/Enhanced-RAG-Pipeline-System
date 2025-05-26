from typing import List, Dict, Any, Optional, Callable, Union
import asyncio
import logging
from dataclasses import dataclass
from enum import Enum
import aioredis
import json
import pickle
import uuid
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor
import threading
import queue
import signal

logger = logging.getLogger(__name__)

class TaskStatus(Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

class TaskType(Enum):
    DOCUMENT_PROCESSING = "document_processing"
    QUERY_PROCESSING = "query_processing"
    EMBEDDING_GENERATION = "embedding_generation"
    BATCH_OPERATION = "batch_operation"

@dataclass
class Task:
    id: str
    type: TaskType
    payload: Dict[str, Any]
    status: TaskStatus = TaskStatus.PENDING
    created_at: datetime = None
    updated_at: datetime = None
    result: Optional[Any] = None
    error: Optional[str] = None
    worker_id: Optional[str] = None
    priority: int = 0
    retries: int = 0
    max_retries: int = 3

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now()
        if not self.updated_at:
            self.updated_at = self.created_at

@dataclass
class WorkerConfig:
    worker_id: str = None
    redis_url: str = "redis://localhost:6379"
    task_batch_size: int = 10
    max_concurrent_tasks: int = 4
    poll_interval: float = 1.0
    heartbeat_interval: float = 5.0
    task_timeout: int = 300
    enable_retries: bool = True

class DistributedWorker:
    def __init__(
        self,
        config: Optional[WorkerConfig] = None,
        task_handlers: Optional[Dict[TaskType, Callable]] = None
    ):
        self.config = config or WorkerConfig()
        if not self.config.worker_id:
            self.config.worker_id = str(uuid.uuid4())
        
        self.task_handlers = task_handlers or {}
        self.redis = None
        self.running = False
        self.current_tasks: Dict[str, Task] = {}
        self.executor = ThreadPoolExecutor(
            max_workers=self.config.max_concurrent_tasks
        )
        self.task_queue = queue.PriorityQueue()
        self._setup_signal_handlers()

    def _setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown"""
        for sig in (signal.SIGINT, signal.SIGTERM):
            signal.signal(sig, self._signal_handler)

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logger.info(f"Received signal {signum}. Starting graceful shutdown...")
        self.stop()

    async def start(self):
        """Start the worker"""
        logger.info(f"Starting worker {self.config.worker_id}")
        self.redis = await aioredis.from_url(self.config.redis_url)
        self.running = True
        
        # Start background tasks
        asyncio.create_task(self._heartbeat_loop())
        asyncio.create_task(self._cleanup_loop())
        
        await self._process_loop()

    def stop(self):
        """Stop the worker"""
        logger.info(f"Stopping worker {self.config.worker_id}")
        self.running = False

    async def _heartbeat_loop(self):
        """Send periodic heartbeats to Redis"""
        while self.running:
            try:
                await self.redis.hset(
                    "worker_heartbeats",
                    self.config.worker_id,
                    datetime.now().isoformat()
                )
                await asyncio.sleep(self.config.heartbeat_interval)
            except Exception as e:
                logger.error(f"Error in heartbeat loop: {e}")
                await asyncio.sleep(1)

    async def _cleanup_loop(self):
        """Clean up stale tasks and worker records"""
        while self.running:
            try:
                # Clean up stale worker heartbeats
                heartbeats = await self.redis.hgetall("worker_heartbeats")
                current_time = datetime.now()
                
                for worker_id, last_heartbeat in heartbeats.items():
                    worker_id = worker_id.decode()
                    last_heartbeat = datetime.fromisoformat(
                        last_heartbeat.decode()
                    )
                    
                    if (current_time - last_heartbeat).seconds > \
                        self.config.heartbeat_interval * 3:
                        # Worker is considered dead, reassign its tasks
                        await self._reassign_worker_tasks(worker_id)
                        await self.redis.hdel("worker_heartbeats", worker_id)
                
                await asyncio.sleep(self.config.heartbeat_interval * 2)
            except Exception as e:
                logger.error(f"Error in cleanup loop: {e}")
                await asyncio.sleep(1)

    async def _reassign_worker_tasks(self, worker_id: str):
        """Reassign tasks from a dead worker"""
        try:
            tasks = await self.redis.hgetall(f"worker_tasks:{worker_id}")
            for task_id, task_data in tasks.items():
                task = pickle.loads(task_data)
                if task.status == TaskStatus.PROCESSING:
                    task.status = TaskStatus.PENDING
                    task.worker_id = None
                    if task.retries < task.max_retries:
                        task.retries += 1
                        await self._enqueue_task(task)
                    else:
                        task.status = TaskStatus.FAILED
                        task.error = "Worker died, max retries exceeded"
                        await self._update_task(task)
        except Exception as e:
            logger.error(f"Error reassigning tasks from worker {worker_id}: {e}")

    async def _process_loop(self):
        """Main task processing loop"""
        while self.running:
            try:
                # Get batch of tasks
                tasks = await self._get_pending_tasks()
                
                for task in tasks:
                    if not self.running:
                        break
                    
                    if len(self.current_tasks) >= self.config.max_concurrent_tasks:
                        break
                    
                    # Process task in thread pool
                    self.current_tasks[task.id] = task
                    asyncio.create_task(self._process_task(task))
                
                await asyncio.sleep(self.config.poll_interval)
            
            except Exception as e:
                logger.error(f"Error in process loop: {e}")
                await asyncio.sleep(1)

    async def _get_pending_tasks(self) -> List[Task]:
        """Get pending tasks from Redis"""
        try:
            # Get tasks sorted by priority and creation time
            task_ids = await self.redis.zrange(
                "pending_tasks",
                0,
                self.config.task_batch_size - 1,
                withscores=True
            )
            
            tasks = []
            for task_id, _ in task_ids:
                task_data = await self.redis.get(f"task:{task_id.decode()}")
                if task_data:
                    task = pickle.loads(task_data)
                    tasks.append(task)
            
            return tasks
        
        except Exception as e:
            logger.error(f"Error getting pending tasks: {e}")
            return []

    async def _process_task(self, task: Task):
        """Process a single task"""
        try:
            logger.info(f"Processing task {task.id} of type {task.type}")
            task.status = TaskStatus.PROCESSING
            task.worker_id = self.config.worker_id
            await self._update_task(task)
            
            # Get task handler
            handler = self.task_handlers.get(task.type)
            if not handler:
                raise ValueError(f"No handler for task type {task.type}")
            
            # Execute task in thread pool
            future = self.executor.submit(handler, task.payload)
            try:
                result = await asyncio.wrap_future(future)
                task.result = result
                task.status = TaskStatus.COMPLETED
            except Exception as e:
                task.error = str(e)
                task.status = TaskStatus.FAILED
                if self.config.enable_retries and task.retries < task.max_retries:
                    task.retries += 1
                    task.status = TaskStatus.PENDING
                    task.worker_id = None
            
            await self._update_task(task)
        
        except Exception as e:
            logger.error(f"Error processing task {task.id}: {e}")
            task.status = TaskStatus.FAILED
            task.error = str(e)
            await self._update_task(task)
        
        finally:
            self.current_tasks.pop(task.id, None)

    async def _update_task(self, task: Task):
        """Update task status in Redis"""
        try:
            task.updated_at = datetime.now()
            
            # Store task data
            await self.redis.set(
                f"task:{task.id}",
                pickle.dumps(task)
            )
            
            # Update task status
            if task.status == TaskStatus.PENDING:
                score = (task.priority * 1e10) + task.created_at.timestamp()
                await self.redis.zadd("pending_tasks", {task.id: score})
            else:
                await self.redis.zrem("pending_tasks", task.id)
            
            # Update worker task mapping
            if task.worker_id:
                await self.redis.hset(
                    f"worker_tasks:{task.worker_id}",
                    task.id,
                    pickle.dumps(task)
                )
            
            # Publish task update
            await self.redis.publish(
                "task_updates",
                json.dumps({
                    "task_id": task.id,
                    "status": task.status.value,
                    "updated_at": task.updated_at.isoformat()
                })
            )
        
        except Exception as e:
            logger.error(f"Error updating task {task.id}: {e}")

    async def _enqueue_task(self, task: Task):
        """Enqueue a task for processing"""
        try:
            score = (task.priority * 1e10) + task.created_at.timestamp()
            await self.redis.zadd("pending_tasks", {task.id: score})
            await self.redis.set(f"task:{task.id}", pickle.dumps(task))
        except Exception as e:
            logger.error(f"Error enqueuing task {task.id}: {e}")

class DistributedProcessingManager:
    """Manager class for distributed processing"""
    
    def __init__(
        self,
        redis_url: str = "redis://localhost:6379",
        task_timeout: int = 300
    ):
        self.redis_url = redis_url
        self.task_timeout = task_timeout
        self.redis = None
        self._subscriber = None
        self.task_callbacks: Dict[str, Callable] = {}

    async def start(self):
        """Start the manager"""
        self.redis = await aioredis.from_url(self.redis_url)
        self._subscriber = self.redis.pubsub()
        await self._subscriber.subscribe("task_updates")
        asyncio.create_task(self._monitor_updates())

    async def stop(self):
        """Stop the manager"""
        if self._subscriber:
            await self._subscriber.unsubscribe()
        if self.redis:
            await self.redis.close()

    async def submit_task(
        self,
        task_type: TaskType,
        payload: Dict[str, Any],
        priority: int = 0
    ) -> str:
        """Submit a task for processing"""
        task = Task(
            id=str(uuid.uuid4()),
            type=task_type,
            payload=payload,
            priority=priority
        )
        
        score = (task.priority * 1e10) + task.created_at.timestamp()
        await self.redis.zadd("pending_tasks", {task.id: score})
        await self.redis.set(f"task:{task.id}", pickle.dumps(task))
        
        return task.id

    async def get_task_status(self, task_id: str) -> Optional[Task]:
        """Get the status of a task"""
        task_data = await self.redis.get(f"task:{task_id}")
        return pickle.loads(task_data) if task_data else None

    async def cancel_task(self, task_id: str) -> bool:
        """Cancel a pending task"""
        task = await self.get_task_status(task_id)
        if not task or task.status != TaskStatus.PENDING:
            return False
        
        await self.redis.zrem("pending_tasks", task_id)
        task.status = TaskStatus.FAILED
        task.error = "Task cancelled"
        await self.redis.set(f"task:{task_id}", pickle.dumps(task))
        return True

    def register_callback(
        self,
        task_id: str,
        callback: Callable[[Task], None]
    ):
        """Register a callback for task updates"""
        self.task_callbacks[task_id] = callback

    async def _monitor_updates(self):
        """Monitor task updates"""
        while True:
            try:
                message = await self._subscriber.get_message(
                    ignore_subscribe_messages=True
                )
                if message:
                    data = json.loads(message["data"])
                    task_id = data["task_id"]
                    
                    if task_id in self.task_callbacks:
                        task = await self.get_task_status(task_id)
                        if task:
                            self.task_callbacks[task_id](task)
                
                await asyncio.sleep(0.1)
            
            except Exception as e:
                logger.error(f"Error monitoring updates: {e}")
                await asyncio.sleep(1)

    async def get_worker_status(self) -> Dict[str, Any]:
        """Get status of all workers"""
        workers = {}
        heartbeats = await self.redis.hgetall("worker_heartbeats")
        
        for worker_id, last_heartbeat in heartbeats.items():
            worker_id = worker_id.decode()
            last_heartbeat = datetime.fromisoformat(last_heartbeat.decode())
            
            # Get worker's tasks
            tasks = await self.redis.hgetall(f"worker_tasks:{worker_id}")
            task_count = len(tasks)
            active_tasks = sum(
                1 for t in tasks.values()
                if pickle.loads(t).status == TaskStatus.PROCESSING
            )
            
            workers[worker_id] = {
                "last_heartbeat": last_heartbeat.isoformat(),
                "total_tasks": task_count,
                "active_tasks": active_tasks,
                "status": "alive" if (
                    datetime.now() - last_heartbeat
                ).seconds < 15 else "dead"
            }
        
        return workers

    async def get_queue_status(self) -> Dict[str, int]:
        """Get status of task queues"""
        return {
            "pending": await self.redis.zcard("pending_tasks"),
            "processing": len(await self.redis.keys("task:*"))
        }
