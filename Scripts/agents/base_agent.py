"""
Base Agent class for the Collaborative Agent Network.
"""
import uuid
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field # Added dataclass and field
import logging
import asyncio
from datetime import datetime

logger = logging.getLogger(__name__)

# Forward declaration for Message class if it's in the same module or for type hinting
class Message:
    pass

# Forward declaration for AgentNetwork
class AgentNetwork:
    pass

class Agent(ABC):
    """
    Abstract Base Class for all agents in the network.
    """
    def __init__(self, agent_id: Optional[str] = None, agent_name: Optional[str] = None, network: Optional[AgentNetwork] = None):
        self.agent_id: str = agent_id or str(uuid.uuid4())
        self.agent_name: str = agent_name or self.__class__.__name__
        self.network: Optional[AgentNetwork] = network # Reference to the agent network/router
        self.mailbox: asyncio.Queue = asyncio.Queue()
        self.knowledge_base: Dict[str, Any] = {} # Simple KV store for agent's knowledge
        self.capabilities: List[str] = [] # List of capabilities this agent has
        self.performance_metrics: Dict[str, Any] = {"tasks_processed": 0, "successful_tasks": 0, "failed_tasks": 0, "avg_processing_time": 0.0}
        self.status: str = "idle" # idle, busy, error
        self.message_handlers: Dict[str, Callable[['Agent', Message], Any]] = {} # type -> handler

        self._task_history: List[Dict[str, Any]] = []
        self._current_tasks: Dict[str, asyncio.Task] = {} # task_id -> asyncio.Task
        logger.info(f"Agent {self.agent_name} ({self.agent_id}) initialized.")

    @abstractmethod
    async def process_message(self, message: Message):
        """Processes an incoming message from the mailbox."""
        pass

    async def send_message_to_network(self, recipient_agent_id: str, message_type: str, message_content: Dict[str, Any], requires_ack: bool = False) -> Optional[Message]:
        """Sends a message via the AgentNetwork."""
        if not self.network:
            logger.error(f"Agent {self.agent_name} ({self.agent_id}) has no network reference. Cannot send message.")
            return None

        # The AgentNetwork's send_message_to_agent handles message creation
        # and posting to the bus or direct routing.
        # requires_ack handling would need to be implemented at the network/protocol level
        # or by the agent waiting for a correlated reply.
        msg = await self.network.send_message_to_agent(
            sender_agent_id=self.agent_id,
            recipient_agent_id=recipient_agent_id,
            message_type=message_type,
            content=message_content
            # TODO: requires_ack needs a more robust implementation.
            # The Message object itself has `requires_acknowledgment`,
            # but the waiting logic is not in this base send method.
        )
        if msg and requires_ack:
             logger.debug(f"{self.agent_name} sent message {msg.message_id} that requires ack. Ack handling is conceptual.")
        return msg

    async def receive_message(self) -> Message:
        """Waits for and returns a message from the agent's mailbox."""
        message = await self.mailbox.get()
        logger.debug(f"Agent {self.agent_name} received message {message.message_id} of type '{message.message_type}' from {message.sender_id}")
        self.mailbox.task_done()
        return message

    async def start(self):
        """Starts the agent's main processing loop."""
        logger.info(f"Agent {self.agent_name} ({self.agent_id}) starting...")
        self.status = "idle"
        try:
            while True:
                message = await self.receive_message()
                self.status = "busy"
                await self.process_message(message)
                self.status = "idle"
        except asyncio.CancelledError:
            logger.info(f"Agent {self.agent_name} ({self.agent_id}) stopping...")
        except Exception as e:
            logger.error(f"Agent {self.agent_name} encountered an error in main loop: {e}", exc_info=True)
            self.status = "error"
        finally:
            logger.info(f"Agent {self.agent_name} ({self.agent_id}) stopped.")


    def update_knowledge(self, key: str, value: Any):
        """Updates the agent's internal knowledge base."""
        self.knowledge_base[key] = value
        logger.debug(f"Agent {self.agent_name} knowledge updated: {key} = {value}")

    def get_knowledge(self, key: str) -> Optional[Any]:
        """Retrieves knowledge from the agent's internal knowledge base."""
        return self.knowledge_base.get(key)

    def log_task(self, task_id: str, status: str, details: Optional[Dict] = None):
        self._task_history.append({
            "task_id": task_id,
            "status": status, # e.g., "received", "processing", "completed", "failed"
            "timestamp": datetime.utcnow().isoformat(),
            "details": details or {}
        })
        if status == "completed":
            self.performance_metrics["tasks_processed"] +=1
            self.performance_metrics["successful_tasks"] +=1
        elif status == "failed":
            self.performance_metrics["tasks_processed"] +=1
            self.performance_metrics["failed_tasks"] +=1


    def register_capability(self, capability: str):
        if capability not in self.capabilities:
            self.capabilities.append(capability)

    def register_message_handler(self, message_type: str, handler: Callable[['Agent', Message], Any]):
        """Registers a handler for a specific message type."""
        self.message_handlers[message_type] = handler
        logger.info(f"Agent {self.agent_name} registered handler for message type '{message_type}'.")

    # --- Methods for inter-agent learning (conceptual) ---
    def share_knowledge(self, recipient_agent: 'Agent', knowledge_key: str, knowledge_value: Any):
        """Shares a piece of knowledge with another agent."""
        # This would use the messaging system.
        content = {"key": knowledge_key, "value": knowledge_value, "type": "knowledge_sharing"}
        # await self.send_message(recipient_agent, content, "KNOWLEDGE_SHARE") # Conceptual
        logger.info(f"{self.agent_name} attempting to share knowledge '{knowledge_key}' with {recipient_agent.agent_name}")
        # Direct update for now, replace with messaging
        recipient_agent.update_knowledge(knowledge_key, knowledge_value)


    def learn_from_interaction(self, interaction_data: Dict[str, Any]):
        """Learns from an interaction or task outcome."""
        # Example: update internal models, heuristics, or knowledge_base
        logger.debug(f"Agent {self.agent_name} learning from interaction: {interaction_data}")
        # This is highly agent-specific.
        pass

class Message:
    """
    Represents a message passed between agents.
    """
    def __init__(self, sender_id: str, recipient_id: str, message_type: str,
                 content: Dict[str, Any], message_id: Optional[str] = None,
                 correlation_id: Optional[str] = None,
                 timestamp: Optional[str] = None,
                 requires_acknowledgment: bool = False,
                 status: Optional[str] = None, # e.g. "pending", "delivered", "processed", "failed"
                 error_info: Optional[Dict] = None):
        self.message_id: str = message_id or str(uuid.uuid4())
        self.correlation_id: str = correlation_id or self.message_id # For request-reply patterns
        self.sender_id: str = sender_id
        self.recipient_id: str = recipient_id
        self.message_type: str = message_type # e.g., "TASK_REQUEST", "TASK_RESULT", "KNOWLEDGE_QUERY"
        self.content: Dict[str, Any] = content # The actual payload
        self.timestamp: str = timestamp or datetime.utcnow().isoformat()
        self.requires_acknowledgment: bool = requires_acknowledgment
        self.status: Optional[str] = status
        self.error_info: Optional[Dict] = error_info

    def create_reply(self, reply_content: Dict[str, Any], reply_type: Optional[str] = None, status: Optional[str] = "processed") -> 'Message':
        """Creates a reply message based on this message."""
        return Message(
            sender_id=self.recipient_id, # Original recipient is now sender
            recipient_id=self.sender_id,  # Original sender is now recipient
            message_type=reply_type or f"{self.message_type}_REPLY",
            content=reply_content,
            correlation_id=self.correlation_id, # Link to original message
            status=status
        )

    def __repr__(self):
        return (f"Message(id={self.message_id}, type='{self.message_type}', "
                f"from={self.sender_id}, to={self.recipient_id}, status='{self.status}')")

# Example of a simple task structure that might be in message content
@dataclass
class Task:
    task_id: str
    task_type: str # Specific to the agent that will handle it
    data: Dict[str, Any]
    priority: int = 0
    deadline: Optional[datetime] = None
    # Other metadata: source_request_id, user_id, etc.
