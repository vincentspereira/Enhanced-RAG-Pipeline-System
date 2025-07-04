"""
Test script for the Collaborative Agent Network.
"""
import asyncio
import logging

# Configure basic logging for the test
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

from .base_agent import Message # Relative imports
from .agent_types import OrchestratorAgent, SpecialistAgent, LearningAgent, QualityAssuranceAgent
from .network import AgentNetwork

async def main():
    logger.info("Starting Agent Network Test Scenario...")

    # 1. Create the Agent Network
    network = AgentNetwork()
    asyncio.create_task(network.start_router()) # Start the message router

    # 2. Create Agents and provide them with the network reference
    orchestrator = OrchestratorAgent(network=network)
    research_specialist = SpecialistAgent(expertise="research", network=network, agent_name="Researcher-01")
    creative_specialist = SpecialistAgent(expertise="creative_writing", network=network, agent_name="Writer-01")
    qa_agent = QualityAssuranceAgent(network=network)
    learning_agent = LearningAgent(network=network)

    # 3. Register Agents with the Network
    await network.register_agent(orchestrator)
    await network.register_agent(research_specialist)
    await network.register_agent(creative_specialist)
    await network.register_agent(qa_agent)
    await network.register_agent(learning_agent)

    # 4. Start Agent Processing Loops
    # These tasks will run indefinitely until cancelled or an error occurs in their loop
    agent_tasks = [
        asyncio.create_task(orchestrator.start()),
        asyncio.create_task(research_specialist.start()),
        asyncio.create_task(creative_specialist.start()),
        asyncio.create_task(qa_agent.start()),
        asyncio.create_task(learning_agent.start()),
    ]

    await asyncio.sleep(0.1) # Allow agents to start up

    # 5. Simulate an initial request to the Orchestrator
    # This initial message is sent from a "UserProxy" or an external source.
    # For this test, we'll have the test script act as the initial sender.
    # The sender_id for such an initial message might be special, e.g., "EXTERNAL_REQUESTER"
    # or the network could assign a temporary ID if it needs to route replies back.

    # For simplicity, let's assume the orchestrator can handle a message without a real sender_id
    # or we can make one of the other agents (e.g. learning_agent) the initial requester.

    initial_requester_id = learning_agent.agent_id # Let learning agent be the requester for this test

    workflow_def = {
        "name": "SimpleResearchAndWriteWorkflow",
        "steps": [
            {"agent_capability": "execute_task:research", "details": {"topic": "AI in healthcare"}},
            {"agent_capability": "execute_task:creative_writing", "details": {"prompt": "Write a summary about AI in healthcare based on research."}}
        ]
    }

    logger.info(f"Sending initial workflow request from {initial_requester_id} to Orchestrator {orchestrator.agent_id}")

    # The Orchestrator's _handle_execute_workflow currently doesn't use delegate_task_to_capable_agent.
    # It simulates a workflow and sends a reply to the original message's sender_id.
    # So, the learning_agent should receive a reply if its ID is used as sender.

    initial_message_content = {
        "workflow": workflow_def,
        "data": {"initial_topic": "AI in healthcare"},
        "workflow_id": "test_workflow_001"
    }

    # Use network.send_message_to_agent to correctly put it on the bus or route it
    await network.send_message_to_agent(
        sender_agent_id=initial_requester_id, # So orchestrator knows where to reply
        recipient_agent_id=orchestrator.agent_id,
        message_type="EXECUTE_WORKFLOW",
        content=initial_message_content
    )

    # Let the system run for a bit to process messages
    logger.info("Test scenario initiated. Waiting for agents to process...")
    await asyncio.sleep(5) # Increased time for simulated tasks and replies

    # 6. Observe (logs will show interactions)
    # In a real test, you'd check agent states, mailboxes, or specific outputs.
    logger.info("--- Agent Performance Metrics ---")
    for agent_id, agent_instance in network.agents.items():
        logger.info(f"Agent: {agent_instance.agent_name} ({agent_id})")
        logger.info(f"  Status: {agent_instance.status}")
        logger.info(f"  Metrics: {agent_instance.performance_metrics}")
        logger.info(f"  Mailbox size: {agent_instance.mailbox.qsize()}")

    logger.info("--- Network Stats ---")
    logger.info(network.get_network_stats())

    # 7. Stop agents and network
    logger.info("Stopping agent tasks...")
    for task in agent_tasks:
        task.cancel()

    await asyncio.gather(*agent_tasks, return_exceptions=True) # Wait for tasks to finish cancelling

    await network.stop_router() # Stop the message router
    logger.info("Agent Network Test Scenario Complete.")

if __name__ == "__main__":
    # Ensure a new event loop for python versions < 3.10 on Windows if needed,
    # or if running multiple times in a Jupyter notebook.
    # For simple script, asyncio.run should be fine.
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Test scenario interrupted by user.")
    except Exception as e:
        logger.error(f"An error occurred in the test scenario: {e}", exc_info=True)
