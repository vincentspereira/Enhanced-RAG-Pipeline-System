"""
Unit tests for Agent base class and specific agent types.
"""
import unittest
import asyncio
from unittest.mock import MagicMock, AsyncMock

from Scripts.agents.base_agent import Agent, Message, Task
from Scripts.agents.agent_types import OrchestratorAgent, SpecialistAgent, LearningAgent
from Scripts.agents.network import AgentNetwork

# Since Agent methods are async and involve asyncio.Queue,
# testing them in a synchronous unittest framework requires careful handling of the event loop.

class TestAgentBase(unittest.IsolatedAsyncioTestCase): # Use IsolatedAsyncioTestCase for async tests

    async def test_agent_initialization(self):
        agent = OrchestratorAgent(agent_id="test_orch", agent_name="TestOrchestrator")
        self.assertEqual(agent.agent_id, "test_orch")
        self.assertEqual(agent.agent_name, "TestOrchestrator")
        self.assertIsInstance(agent.mailbox, asyncio.Queue)
        self.assertEqual(agent.status, "idle")

    async def test_message_creation(self):
        msg = Message(sender_id="s1", recipient_id="r1", message_type="TEST", content={"data": "hello"})
        self.assertEqual(msg.sender_id, "s1")
        self.assertEqual(msg.recipient_id, "r1")
        self.assertEqual(msg.message_type, "TEST")
        self.assertEqual(msg.content, {"data": "hello"})
        self.assertIsNotNone(msg.message_id)
        self.assertEqual(msg.correlation_id, msg.message_id) # Default behavior

    async def test_message_reply_creation(self):
        original_msg = Message(sender_id="s1", recipient_id="r1", message_type="REQUEST", content={})
        reply_msg = original_msg.create_reply(reply_content={"result": "done"}, reply_type="REQUEST_REPLY")

        self.assertEqual(reply_msg.sender_id, "r1") # Recipient of original is sender of reply
        self.assertEqual(reply_msg.recipient_id, "s1") # Sender of original is recipient of reply
        self.assertEqual(reply_msg.message_type, "REQUEST_REPLY")
        self.assertEqual(reply_msg.correlation_id, original_msg.correlation_id)


class TestAgentNetwork(unittest.IsolatedAsyncioTestCase):

    async def asyncSetUp(self):
        self.network = AgentNetwork()
        # Start router for tests that need message delivery via bus
        asyncio.create_task(self.network.start_router())
        await asyncio.sleep(0.01) # give router a chance to start

    async def asyncTearDown(self):
        await self.network.stop_router()
        await asyncio.sleep(0.01) # give router a chance to stop

    async def test_agent_registration(self):
        agent = OrchestratorAgent(network=self.network) # Pass network
        await self.network.register_agent(agent)
        self.assertIn(agent.agent_id, self.network.agents)
        retrieved_agent = self.network.get_agent(agent.agent_id)
        self.assertEqual(agent, retrieved_agent)

    async def test_find_agent_by_capability(self):
        orchestrator = OrchestratorAgent(network=self.network)
        orchestrator.register_capability("test_capability_1")
        await self.network.register_agent(orchestrator)

        specialist = SpecialistAgent(expertise="test_expertise", network=self.network)
        # SpecialistAgent registers "execute_task:test_expertise" by default
        await self.network.register_agent(specialist)

        found_orchestrators = self.network.find_agents_with_capability("test_capability_1")
        self.assertIn(orchestrator, found_orchestrators)

        found_specialists = self.network.find_agents_with_capability("execute_task:test_expertise")
        self.assertIn(specialist, found_specialists)

    async def test_message_routing_via_bus(self):
        sender_agent = OrchestratorAgent(agent_id="sender", network=self.network)
        recipient_agent = SpecialistAgent(expertise="test", agent_id="recipient", network=self.network)

        await self.network.register_agent(sender_agent)
        await self.network.register_agent(recipient_agent)

        # Mock recipient's process_message to check if it's called
        recipient_agent.process_message = AsyncMock()

        # Start recipient agent's loop to process from its mailbox
        recipient_task = asyncio.create_task(recipient_agent.start())
        await asyncio.sleep(0.01) # Allow recipient to start

        test_content = {"data": "hello from bus"}
        await self.network.send_message_to_agent(
            sender_agent_id=sender_agent.agent_id,
            recipient_agent_id=recipient_agent.agent_id,
            message_type="TEST_BUS_MESSAGE",
            content=test_content,
            use_bus=True # Explicitly use the bus
        )

        await asyncio.sleep(0.1) # Allow time for routing and processing

        recipient_agent.process_message.assert_called_once()
        called_message = recipient_agent.process_message.call_args[0][0]
        self.assertEqual(called_message.sender_id, sender_agent.agent_id)
        self.assertEqual(called_message.content, test_content)

        # Cleanup
        recipient_task.cancel()
        try:
            await recipient_task
        except asyncio.CancelledError:
            pass


class TestSpecialistAgent(unittest.IsolatedAsyncioTestCase):
    async def test_specialist_handles_task(self):
        network_mock = MagicMock(spec=AgentNetwork)
        network_mock.send_message_to_agent = AsyncMock() # Mock the network call

        specialist = SpecialistAgent(expertise="research", network=network_mock)

        # Simulate Orchestrator sending a task
        task_message_content = {"task_data": {"topic": "AI"}, "task_id": "task123"}
        incoming_message = Message(
            sender_id="orchestrator_id",
            recipient_id=specialist.agent_id,
            message_type="PERFORM_RESEARCH_TASK",
            content=task_message_content,
            correlation_id="task123"
        )

        await specialist.process_message(incoming_message) # Call process directly for unit test

        # Check if specialist tried to send a reply via the mocked network
        network_mock.send_message_to_agent.assert_called_once()
        call_args = network_mock.send_message_to_agent.call_args[1] # Get kwargs

        self.assertEqual(call_args['recipient_agent_id'], "orchestrator_id")
        self.assertEqual(call_args['message_type'], "RESEARCH_TASK_RESULT")
        self.assertIn("status", call_args['content'])
        self.assertEqual(call_args['content']['status'], "completed")
        self.assertIn("result", call_args['content'])


class TestNewAgentTypes(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.network_mock = MagicMock(spec=AgentNetwork)
        # Define an async mock for send_message_to_agent for the network_mock
        self.network_mock.send_message_to_agent = AsyncMock()


    def test_research_agent_creation(self):
        agent = SpecialistAgent(expertise="research", network=self.network_mock, agent_name="TestResearchAgent") # Using SpecialistAgent directly as ResearchAgent is a subclass
        self.assertIn("execute_task:research", agent.capabilities)
        self.assertEqual(agent.agent_name, "TestResearchAgent")

    def test_creative_agent_creation(self):
        agent = SpecialistAgent(expertise="creative_writing", network=self.network_mock)
        self.assertIn("execute_task:creative_writing", agent.capabilities)
        self.assertEqual(agent.agent_name, "Specialist-Creative_writing") # Default name

    def test_analysis_agent_creation(self):
        agent = SpecialistAgent(expertise="data_analysis", network=self.network_mock)
        self.assertIn("execute_task:data_analysis", agent.capabilities)

    def test_translation_agent_creation(self):
        # TranslationAgent might have more specific init if it were more complex
        agent = SpecialistAgent(expertise="translation", network=self.network_mock)
        self.assertIn("execute_task:translation", agent.capabilities)
        # If TranslationAgent had specific capabilities like `translate_text:en,es`
        # those would be tested here. For now, it inherits from SpecialistAgent.

    def test_compliance_agent_creation(self):
        agent = SpecialistAgent(expertise="compliance_check", network=self.network_mock)
        self.assertIn("execute_task:compliance_check", agent.capabilities)

class TestBaseAgentHandlers(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.network_mock = AsyncMock(spec=AgentNetwork) # Use AsyncMock for network
        self.agent = Agent(agent_id="handler_test_agent", network=self.network_mock)
        # Manually register since Agent is ABC and _register_default_handlers is part of concrete usually
        # For this test, we assume it's callable or we directly call the handler.
        # If testing BaseAgent directly, and _handle_knowledge_share is defined, it's fine.
        # If Agent class itself doesn't call _register_default_handlers, this test needs adjustment
        # or we test via a concrete subclass that does call it.
        # The current BaseAgent's __init__ does call _register_default_handlers.

    async def test_handle_knowledge_share(self):
        self.assertNotIn("test_key", self.agent.knowledge_base)
        share_message_content = {"key": "test_key", "value": "test_value", "shared_by": "sharer_agent"}
        share_message = Message(
            sender_id="sharer_agent",
            recipient_id=self.agent.agent_id,
            message_type="KNOWLEDGE_SHARE",
            content=share_message_content
        )
        await self.agent._handle_knowledge_share(share_message) # Call handler directly
        self.assertIn("test_key", self.agent.knowledge_base)
        self.assertEqual(self.agent.get_knowledge("test_key"), "test_value")

class TestOrchestratorAgentHandlers(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.network_mock = AsyncMock(spec=AgentNetwork)
        self.orchestrator = OrchestratorAgent(agent_id="test_orch_handler", network=self.network_mock)

    async def test_handle_shared_insight(self):
        self.assertNotIn("test_insight", self.orchestrator.strategic_insights)
        insight_content = {"insight_key": "test_insight", "insight_value": "valuable_info", "source_agent": "learning_agent_id"}
        insight_message = Message(
            sender_id="learning_agent_id",
            recipient_id=self.orchestrator.agent_id,
            message_type="SHARED_INSIGHT",
            content=insight_content
        )
        await self.orchestrator._handle_shared_insight(insight_message) # Call handler directly
        self.assertIn("test_insight", self.orchestrator.strategic_insights)
        self.assertEqual(self.orchestrator.strategic_insights["test_insight"], "valuable_info")
        self.assertEqual(self.orchestrator.get_knowledge("insight_test_insight"), "valuable_info")


if __name__ == '__main__':
    unittest.main()
```
