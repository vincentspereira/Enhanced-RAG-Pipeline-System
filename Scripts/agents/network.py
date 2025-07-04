"""
Agent Network and Message Router for the Collaborative Agent System.
"""
import logging
import asyncio
from typing import Dict, Optional, List, Callable, Any # Added Any
from datetime import datetime # Added for network_stats timestamp

from .base_agent import Agent, Message

logger = logging.getLogger(__name__)

class AgentNetwork:
    """
    Manages agent registration, discovery, and message routing.
    """
    def __init__(self):
        self.agents: Dict[str, Agent] = {} # agent_id -> Agent instance
        self.agent_capabilities: Dict[str, List[str]] = {} # capability -> List[agent_id]
        self.message_bus: asyncio.Queue = asyncio.Queue() # Central bus for all messages (optional, can also route directly)
        self._is_routing = False
        self._router_task: Optional[asyncio.Task] = None

    async def register_agent(self, agent: Agent):
        """Registers an agent with the network."""
        if agent.agent_id in self.agents:
            logger.warning(f"Agent {agent.agent_name} ({agent.agent_id}) already registered.")
            return

        self.agents[agent.agent_id] = agent
        logger.info(f"Agent {agent.agent_name} ({agent.agent_id}) registered with the network.")

        for capability in agent.capabilities:
            if capability not in self.agent_capabilities:
                self.agent_capabilities[capability] = []
            if agent.agent_id not in self.agent_capabilities[capability]:
                self.agent_capabilities[capability].append(agent.agent_id)
        logger.info(f"Agent {agent.agent_name} capabilities registered: {agent.capabilities}")

        # Start the agent's own processing loop if it's not already running
        # This depends on how agent lifecycle is managed. For now, assume agent.start() is called externally.

    async def unregister_agent(self, agent_id: str):
        """Unregisters an agent from the network."""
        if agent_id in self.agents:
            agent = self.agents.pop(agent_id)
            logger.info(f"Agent {agent.agent_name} ({agent_id}) unregistered.")

            # Remove from capability lookup
            for capability in list(self.agent_capabilities.keys()):
                if agent_id in self.agent_capabilities[capability]:
                    self.agent_capabilities[capability].remove(agent_id)
                if not self.agent_capabilities[capability]: # If no agent has this capability anymore
                    del self.agent_capabilities[capability]
            # TODO: Handle any pending messages for this agent?
        else:
            logger.warning(f"Attempted to unregister non-existent agent: {agent_id}")

    def get_agent(self, agent_id: str) -> Optional[Agent]:
        """Retrieves a registered agent by its ID."""
        return self.agents.get(agent_id)

    def find_agents_with_capability(self, capability: str) -> List[Agent]:
        """Finds agents that have a specific capability."""
        agent_ids = self.agent_capabilities.get(capability, [])
        return [self.agents[aid] for aid in agent_ids if aid in self.agents]

    async def route_message(self, message: Message):
        """Routes a message to the recipient agent's mailbox."""
        recipient_agent = self.get_agent(message.recipient_id)
        if recipient_agent:
            await recipient_agent.mailbox.put(message)
            logger.debug(f"Message {message.message_id} (type: {message.message_type}) routed to agent {message.recipient_id}")
        else:
            logger.error(f"Recipient agent {message.recipient_id} not found for message {message.message_id}. Message dropped.")
            # TODO: Implement dead-letter queue or return-to-sender logic

    async def _message_router_loop(self):
        """Continuously polls the central message bus and routes messages."""
        self._is_routing = True
        logger.info("Message router loop started.")
        while self._is_routing:
            try:
                message = await self.message_bus.get()
                await self.route_message(message)
                self.message_bus.task_done()
            except asyncio.CancelledError:
                logger.info("Message router loop cancelled.")
                self._is_routing = False
                break
            except Exception as e:
                logger.error(f"Error in message router loop: {e}", exc_info=True)
                # Avoid continuous fast loops on persistent error
                await asyncio.sleep(1)
        logger.info("Message router loop stopped.")

    async def start_router(self):
        """Starts the central message router task."""
        if not self._is_routing and (self._router_task is None or self._router_task.done()):
            self._router_task = asyncio.create_task(self._message_router_loop())
            logger.info("Agent network message router task created.")
        else:
            logger.warning("Router task already running or pending.")

    async def stop_router(self):
        """Stops the central message router task."""
        if self._is_routing and self._router_task:
            logger.info("Stopping agent network message router...")
            self._is_routing = False # Signal loop to stop
            self._router_task.cancel() # Cancel the task
            try:
                await self._router_task
            except asyncio.CancelledError:
                logger.info("Router task successfully cancelled.")
            self._router_task = None
        else:
            logger.info("Router not running or already stopped.")

    # --- Higher-level interaction patterns ---

    async def send_message_to_agent(self, sender_agent_id: str, recipient_agent_id: str,
                                   message_type: str, content: Dict[str, Any],
                                   use_bus: bool = True) -> Message:
        """
        Creates and sends/posts a message.
        If use_bus is True, posts to the central bus for routing.
        Otherwise, attempts direct routing (if recipient is known).
        """
        msg = Message(
            sender_id=sender_agent_id,
            recipient_id=recipient_agent_id,
            message_type=message_type,
            content=content
        )
        if use_bus:
            await self.message_bus.put(msg)
            logger.debug(f"Message {msg.message_id} from {sender_agent_id} to {recipient_agent_id} posted to bus.")
        else:
            await self.route_message(msg) # Direct route
        return msg

    async def delegate_task_to_capable_agent(self, sender_agent_id: str, capability: str,
                                            task_content: Dict[str, Any],
                                            task_message_type: Optional[str] = None) -> Optional[Message]:
        """
        Finds an agent with the required capability and delegates a task to them.
        Uses simple round-robin if multiple agents available.
        """
        capable_agents = self.find_agents_with_capability(capability)
        if not capable_agents:
            logger.error(f"No agent found with capability: {capability} for task from {sender_agent_id}")
            return None

        # Simple load balancing: round robin or random choice (for now, first available)
        # TODO: Implement more sophisticated load balancing based on agent status/load
        target_agent = capable_agents[0]

        msg_type = task_message_type or f"TASK_FOR_{capability.upper()}"

        logger.info(f"Delegating task (type: {msg_type}) for capability '{capability}' to agent {target_agent.agent_name} ({target_agent.agent_id})")
        return await self.send_message_to_agent(
            sender_agent_id=sender_agent_id,
            recipient_agent_id=target_agent.agent_id,
            message_type=msg_type,
            content=task_content
        )

    # --- Performance Monitoring & Scaling ---
    def get_network_stats(self) -> Dict[str, Any]:
        """Returns detailed statistics about the agent network, including performance."""
        agent_details = {}
        total_tasks_processed = 0
        total_successful_tasks = 0
        total_failed_tasks = 0

        for agent_id, agent_instance in self.agents.items():
            # Ensure agent_instance is not None and has performance_metrics attribute
            if agent_instance and hasattr(agent_instance, 'performance_metrics') and isinstance(agent_instance.performance_metrics, dict):
                perf_metrics = agent_instance.performance_metrics
                total_tasks_processed += perf_metrics.get("tasks_processed", 0)
                total_successful_tasks += perf_metrics.get("successful_tasks", 0)
                total_failed_tasks += perf_metrics.get("failed_tasks", 0)

                agent_details[agent_id] = {
                    "name": agent_instance.agent_name,
                    "status": agent_instance.status,
                    "capabilities": agent_instance.capabilities,
                    "mailbox_size": agent_instance.mailbox.qsize(),
                    "performance": perf_metrics, # Includes tasks_processed, successful, failed, avg_time
                    "task_history_count": len(getattr(agent_instance, '_task_history', [])), # Accessing protected member for stats
                }
            elif agent_instance: # Basic info if performance_metrics missing
                 agent_details[agent_id] = {
                    "name": agent_instance.agent_name,
                    "status": agent_instance.status,
                    "capabilities": agent_instance.capabilities,
                    "mailbox_size": agent_instance.mailbox.qsize(),
                    "performance": "N/A",
                    "task_history_count": len(getattr(agent_instance, '_task_history', [])),
                }


        return {
            "timestamp": datetime.utcnow().isoformat(), # Added timestamp
            "total_agents": len(self.agents),
            "message_bus_size": self.message_bus.qsize(),
            "capability_distribution": {cap: len(ids) for cap, ids in self.agent_capabilities.items()},
            "agents": agent_details, # Detailed per-agent stats
            "overall_performance": {
                "total_tasks_processed": total_tasks_processed,
                "total_successful_tasks": total_successful_tasks,
                "total_failed_tasks": total_failed_tasks,
            }
        }

    async def scale_agents_for_capability(self, capability: str, desired_count: int, agent_type_to_create: Optional[type] = None, agent_init_kwargs: Optional[Dict] = None):
        """
        Simulates dynamic scaling of agents for a given capability.
        If current agent count for the capability is less than desired_count,
        it creates and registers new agents of agent_type_to_create.
        """
        agent_init_kwargs = agent_init_kwargs or {}
        current_agents_with_capability = self.find_agents_with_capability(capability)
        current_count = len(current_agents_with_capability)
        agents_to_add = desired_count - current_count

        logger.info(f"Scaling check for capability '{capability}': Current={current_count}, Desired={desired_count}")

        if agents_to_add <= 0:
            logger.info(f"No scaling needed for capability '{capability}'. Current count ({current_count}) meets or exceeds desired ({desired_count}).")
            return

        if not agent_type_to_create:
            logger.error(f"Cannot scale capability '{capability}': agent_type_to_create not provided.")
            return

        if not issubclass(agent_type_to_create, Agent):
            logger.error(f"Cannot scale capability '{capability}': agent_type_to_create is not a subclass of Agent.")
            return

        logger.info(f"Scaling up capability '{capability}' by adding {agents_to_add} new agents of type {agent_type_to_create.__name__}.")

        newly_created_agents: List[Agent] = []
        for i in range(agents_to_add):
            try:
                # Ensure common arguments for Agent are passed if not in agent_init_kwargs
                # For SpecialistAgent, 'expertise' is key. If agent_type_to_create is SpecialistAgent or subclass,
                # and expertise is not in kwargs, it might use capability.
                # This part is tricky without knowing the exact constructor signature of agent_type_to_create.
                # A robust solution would involve a factory pattern or standardized agent constructors.

                # A common pattern for SpecialistAgent is that its expertise is related to the capability.
                # E.g. capability "execute_task:research" -> expertise "research"
                # We assume agent_init_kwargs will contain necessary args like 'expertise' if it's a SpecialistAgent.
                # Or, if it's a generic agent_type_to_create, it might not need specific args beyond agent_id/name.

                # Example: if agent_type_to_create is a SpecialistAgent and needs 'expertise'
                if "expertise" not in agent_init_kwargs and capability.startswith("execute_task:"):
                     agent_init_kwargs_specialized = agent_init_kwargs.copy()
                     agent_init_kwargs_specialized["expertise"] = capability.split(":",1)[1]
                     new_agent = agent_type_to_create(network=self, **agent_init_kwargs_specialized)
                else:
                     new_agent = agent_type_to_create(network=self, **agent_init_kwargs) # Pass self as network

                await self.register_agent(new_agent)
                # Important: Start the new agent's processing loop
                # This assumes the agent's start() method is designed to be run like this.
                asyncio.create_task(new_agent.start())
                logger.info(f"Created and registered new agent {new_agent.agent_name} ({new_agent.agent_id}) for capability '{capability}'. Task loop started.")
                newly_created_agents.append(new_agent)
            except Exception as e:
                logger.error(f"Error creating or registering new agent for capability '{capability}': {e}", exc_info=True)

        if newly_created_agents:
            logger.info(f"Successfully added {len(newly_created_agents)} agents for capability '{capability}'.")
        else:
            logger.warning(f"No new agents were added for capability '{capability}' despite request for {agents_to_add}.")

        # Note: This doesn't handle scaling down or resource limits.
        # True dynamic scaling would involve a resource manager, agent lifecycle events, etc.
