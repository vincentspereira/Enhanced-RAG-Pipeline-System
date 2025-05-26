from typing import Dict, Any, List, Callable, Optional
from dataclasses import dataclass
from enum import Enum
import asyncio
from datetime import datetime
import logging

class WorkflowStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

@dataclass
class WorkflowStep:
    """Represents a single step in a workflow."""
    name: str
    function: Callable
    dependencies: List[str]  # Names of steps that must complete before this one
    retry_count: int = 3
    timeout_seconds: Optional[int] = None

@dataclass
class WorkflowContext:
    """Stores workflow execution context and intermediate results."""
    workflow_id: str
    start_time: datetime
    status: WorkflowStatus
    results: Dict[str, Any]
    errors: Dict[str, str]
    current_step: Optional[str] = None

class WorkflowEngine:
    """Workflow automation engine for managing complex document processing pipelines."""
    
    def __init__(self):
        self.workflows: Dict[str, Dict[str, WorkflowStep]] = {}
        self.contexts: Dict[str, WorkflowContext] = {}
        self.logger = logging.getLogger(__name__)
    
    def register_workflow(self, workflow_name: str, steps: List[WorkflowStep]):
        """Register a new workflow with its steps."""
        # Validate workflow
        self._validate_workflow(steps)
        
        # Store workflow steps indexed by name
        self.workflows[workflow_name] = {step.name: step for step in steps}
        self.logger.info(f"Registered workflow '{workflow_name}' with {len(steps)} steps")
    
    def _validate_workflow(self, steps: List[WorkflowStep]):
        """Validate workflow structure and dependencies."""
        step_names = {step.name for step in steps}
        
        # Check for duplicate names
        if len(step_names) != len(steps):
            raise ValueError("Duplicate step names found in workflow")
        
        # Check dependencies exist
        for step in steps:
            for dep in step.dependencies:
                if dep not in step_names:
                    raise ValueError(f"Step '{step.name}' depends on non-existent step '{dep}'")
        
        # Check for cycles
        self._check_for_cycles(steps)
    
    def _check_for_cycles(self, steps: List[WorkflowStep]):
        """Check for dependency cycles in the workflow."""
        def visit(step: WorkflowStep, visited: set, path: set):
            if step.name in path:
                raise ValueError(f"Circular dependency detected: {' -> '.join(path)}")
            if step.name in visited:
                return
            
            visited.add(step.name)
            path.add(step.name)
            
            step_dict = {s.name: s for s in steps}
            for dep in step.dependencies:
                visit(step_dict[dep], visited, path)
            
            path.remove(step.name)
        
        visited = set()
        for step in steps:
            if step.name not in visited:
                visit(step, visited, set())
    
    async def execute_workflow(self, workflow_name: str, 
                             initial_data: Dict[str, Any]) -> WorkflowContext:
        """Execute a workflow with given initial data."""
        if workflow_name not in self.workflows:
            raise ValueError(f"Workflow '{workflow_name}' not found")
        
        # Create new context
        context = WorkflowContext(
            workflow_id=f"{workflow_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            start_time=datetime.now(),
            status=WorkflowStatus.RUNNING,
            results=initial_data,
            errors={}
        )
        self.contexts[context.workflow_id] = context
        
        try:
            # Get executable steps (those with completed dependencies)
            steps = self.workflows[workflow_name]
            remaining_steps = set(steps.keys())
            
            while remaining_steps:
                # Find steps whose dependencies are met
                executable = [
                    step_name for step_name in remaining_steps
                    if all(dep not in remaining_steps 
                          for dep in steps[step_name].dependencies)
                ]
                
                if not executable:
                    raise ValueError("No executable steps found, possible circular dependency")
                
                # Execute steps in parallel
                tasks = [
                    self._execute_step(steps[step_name], context)
                    for step_name in executable
                ]
                await asyncio.gather(*tasks)
                
                # Remove completed steps
                remaining_steps -= set(executable)
            
            context.status = WorkflowStatus.COMPLETED
            self.logger.info(f"Workflow {context.workflow_id} completed successfully")
            
        except Exception as e:
            context.status = WorkflowStatus.FAILED
            context.errors["workflow_error"] = str(e)
            self.logger.error(f"Workflow {context.workflow_id} failed: {str(e)}")
            
        return context
    
    async def _execute_step(self, step: WorkflowStep, context: WorkflowContext):
        """Execute a single workflow step with retries."""
        context.current_step = step.name
        
        for attempt in range(step.retry_count):
            try:
                # Execute step with timeout if specified
                if step.timeout_seconds:
                    async with asyncio.timeout(step.timeout_seconds):
                        result = await asyncio.to_thread(
                            step.function, context.results
                        )
                else:
                    result = await asyncio.to_thread(
                        step.function, context.results
                    )
                
                # Store result
                context.results[step.name] = result
                self.logger.info(f"Step {step.name} completed successfully")
                return
                
            except Exception as e:
                error = f"Attempt {attempt + 1}/{step.retry_count} failed: {str(e)}"
                self.logger.warning(f"Step {step.name} - {error}")
                
                if attempt == step.retry_count - 1:
                    context.errors[step.name] = error
                    raise
                
                # Wait before retry
                await asyncio.sleep(2 ** attempt)  # Exponential backoff
    
    def get_workflow_status(self, workflow_id: str) -> Optional[WorkflowContext]:
        """Get the current status of a workflow execution."""
        return self.contexts.get(workflow_id)
    
    def cancel_workflow(self, workflow_id: str) -> bool:
        """Cancel a running workflow."""
        context = self.contexts.get(workflow_id)
        if not context or context.status != WorkflowStatus.RUNNING:
            return False
        
        context.status = WorkflowStatus.CANCELLED
        self.logger.info(f"Workflow {workflow_id} cancelled")
        return True
