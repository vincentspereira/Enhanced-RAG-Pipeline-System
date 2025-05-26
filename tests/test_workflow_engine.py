import pytest
from unittest.mock import Mock, patch
import asyncio
from Scripts.workflows.engine import (
    WorkflowEngine,
    WorkflowStep,
    WorkflowStatus,
    WorkflowContext
)

@pytest.fixture
def simple_workflow():
    return [
        WorkflowStep(
            name="step1",
            function=lambda x: {"step1_result": x["input"] + 1},
            dependencies=[]
        ),
        WorkflowStep(
            name="step2",
            function=lambda x: {"step2_result": x["step1_result"] * 2},
            dependencies=["step1"]
        )
    ]

class TestWorkflowEngine:
    def test_workflow_registration(self, simple_workflow):
        """Test registering a workflow"""
        engine = WorkflowEngine()
        engine.register_workflow("test-workflow", simple_workflow)
        
        assert "test-workflow" in engine.workflows
        assert len(engine.workflows["test-workflow"]) == 2
    
    def test_workflow_validation(self):
        """Test workflow validation"""
        engine = WorkflowEngine()
        
        # Test duplicate step names
        duplicate_steps = [
            WorkflowStep(name="step1", function=lambda x: x, dependencies=[]),
            WorkflowStep(name="step1", function=lambda x: x, dependencies=[])
        ]
        
        with pytest.raises(ValueError, match="Duplicate step names"):
            engine.register_workflow("test-workflow", duplicate_steps)
        
        # Test non-existent dependencies
        invalid_deps = [
            WorkflowStep(name="step1", function=lambda x: x, dependencies=["nonexistent"])
        ]
        
        with pytest.raises(ValueError, match="depends on non-existent step"):
            engine.register_workflow("test-workflow", invalid_deps)
    
    @pytest.mark.asyncio
    async def test_workflow_execution(self, simple_workflow):
        """Test workflow execution"""
        engine = WorkflowEngine()
        engine.register_workflow("test-workflow", simple_workflow)
        
        context = await engine.execute_workflow(
            "test-workflow",
            {"input": 1}
        )
        
        assert context.status == WorkflowStatus.COMPLETED
        assert context.results["step1_result"] == 2
        assert context.results["step2_result"] == 4
    
    @pytest.mark.asyncio
    async def test_workflow_error_handling(self):
        """Test workflow error handling"""
        def failing_function(x):
            raise ValueError("Test error")
        
        error_workflow = [
            WorkflowStep(
                name="failing_step",
                function=failing_function,
                dependencies=[],
                retry_count=2
            )
        ]
        
        engine = WorkflowEngine()
        engine.register_workflow("error-workflow", error_workflow)
        
        context = await engine.execute_workflow(
            "error-workflow",
            {"input": 1}
        )
        
        assert context.status == WorkflowStatus.FAILED
        assert "failing_step" in context.errors
    
    @pytest.mark.asyncio
    async def test_parallel_execution(self):
        """Test parallel step execution"""
        async def async_step(x):
            await asyncio.sleep(0.1)
            return {"result": x["input"]}
        
        parallel_workflow = [
            WorkflowStep(
                name="step1",
                function=async_step,
                dependencies=[]
            ),
            WorkflowStep(
                name="step2",
                function=async_step,
                dependencies=[]
            )
        ]
        
        engine = WorkflowEngine()
        engine.register_workflow("parallel-workflow", parallel_workflow)
        
        start_time = asyncio.get_event_loop().time()
        context = await engine.execute_workflow(
            "parallel-workflow",
            {"input": 1}
        )
        end_time = asyncio.get_event_loop().time()
        
        # Both steps should run in parallel, taking ~0.1s total
        assert end_time - start_time < 0.15
        assert context.status == WorkflowStatus.COMPLETED
    
    def test_workflow_cancellation(self):
        """Test workflow cancellation"""
        engine = WorkflowEngine()
        workflow_id = "test-workflow-1"
        
        # Create a running workflow context
        context = WorkflowContext(
            workflow_id=workflow_id,
            start_time=None,
            status=WorkflowStatus.RUNNING,
            results={},
            errors={}
        )
        engine.contexts[workflow_id] = context
        
        # Cancel the workflow
        result = engine.cancel_workflow(workflow_id)
        
        assert result is True
        assert context.status == WorkflowStatus.CANCELLED
