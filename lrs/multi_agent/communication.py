"""Communication mechanisms for multi-agent LRS systems."""

from typing import Dict, Any, Optional, List, Callable  # Added List
from dataclasses import dataclass
from datetime import datetime

from lrs.core.lens import ToolLens, ExecutionResult
from lrs.multi_agent.social_precision import SocialPrecisionTracker

class MessageType(Enum):
    """Types of inter-agent messages"""
    QUERY = "query"              # Ask for information
    INFORM = "inform"            # Share information
    REQUEST = "request"          # Request action
    ACKNOWLEDGE = "acknowledge"  # Confirm receipt
    ERROR = "error"              # Report problem


@dataclass
class Message:
    """
    Inter-agent message.
    
    Attributes:
        from_agent: Sender ID
        to_agent: Receiver ID
        message_type: Type of message
        content: Message payload
        timestamp: When sent
        in_reply_to: Optional message ID this replies to
    """
    from_agent: str
    to_agent: str
    message_type: MessageType
    content: Any
    timestamp: str = None
    in_reply_to: Optional[str] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()


class CommunicationLens(ToolLens):
    """
    Tool for sending messages between agents.
    
    Communication is an information-seeking action:
    - Reduces social uncertainty (increases social precision)
    - Has a cost (time, attention)
    - Provides epistemic value
    
    Examples:
        >>> from lrs.multi_agent.shared_state import SharedWorldState
        >>> 
        >>> shared_state = SharedWorldState()
        >>> comm_tool = CommunicationLens("agent_a", shared_state)
        >>> 
        >>> # Send query to Agent B
        >>> result = comm_tool.get({
        ...     'to_agent': 'agent_b',
        ...     'message_type': 'query',
        ...     'content': 'What is your current task?'
        ... })
    """
    
    def __init__(
        self,
        agent_id: str,
        shared_state: 'SharedWorldState',
        message_cost: float = 0.1
    ):
        """
        Initialize communication tool.
        
        Args:
            agent_id: ID of agent using this tool
            shared_state: Shared world state for message passing
            message_cost: Cost of sending messages (for G calculation)
        """
        super().__init__(
            name=f"send_message_{agent_id}",
            input_schema={
                'type': 'object',
                'required': ['to_agent', 'message_type', 'content'],
                'properties': {
                    'to_agent': {'type': 'string'},
                    'message_type': {'type': 'string'},
                    'content': {'type': 'string'},
                    'in_reply_to': {'type': 'string'}
                }
            },
            output_schema={
                'type': 'object',
                'properties': {
                    'sent': {'type': 'boolean'},
                    'message_id': {'type': 'string'}
                }
            }
        )
        
        self.agent_id = agent_id
        self.shared_state = shared_state
        self.message_cost = message_cost
        self.sent_messages: Dict[str, Message] = {}
        self.received_messages: Dict[str, Message] = {}
    
    def get(self, state: dict) -> ExecutionResult:
        """
        Send a message to another agent.
        
        Args:
            state: Must contain 'to_agent', 'message_type', 'content'
        
        Returns:
            ExecutionResult with message confirmation
        """
        self.call_count += 1
        
        try:
            to_agent = state.get('to_agent')
            msg_type = state.get('message_type')
            content = state.get('content')
            in_reply_to = state.get('in_reply_to')
            
            # Validate
            if not to_agent or not msg_type or content is None:
                self.failure_count += 1
                return ExecutionResult(
                    success=False,
                    value=None,
                    error="Missing required fields",
                    prediction_error=0.9
                )
            
            # Create message
            message = Message(
                from_agent=self.agent_id,
                to_agent=to_agent,
                message_type=MessageType(msg_type),
                content=content,
                in_reply_to=in_reply_to
            )
            
            # Store in shared state
            message_id = f"{self.agent_id}_{len(self.sent_messages)}"
            self.sent_messages[message_id] = message
            
            # Update shared state
            self.shared_state.update(to_agent, {
                'incoming_message': {
                    'id': message_id,
                    'from': message.from_agent,
                    'type': message.message_type.value,
                    'content': message.content,
                    'timestamp': message.timestamp
                }
            })
            
            # Communication has epistemic value (reduces social uncertainty)
            # Prediction error reflects information gain
            prediction_error = 0.2  # Low error = high info gain
            
            return ExecutionResult(
                success=True,
                value={
                    'sent': True,
                    'message_id': message_id,
                    'timestamp': message.timestamp
                },
                error=None,
                prediction_error=prediction_error
            )
        
        except Exception as e:
            self.failure_count += 1
            return ExecutionResult(
                success=False,
                value=None,
                error=str(e),
                prediction_error=0.9
            )
    
    def set(self, state: dict, observation: dict) -> dict:
        """Update state with sent message"""
        return {
            **state,
            'last_message_sent': observation,
            'communication_count': state.get('communication_count', 0) + 1
        }
    
    def receive_messages(self) -> List[Message]:
        """
        Check for incoming messages.
        
        Returns:
            List of received messages
        """
        agent_state = self.shared_state.get_agent_state(self.agent_id)
        
        if not agent_state or 'incoming_message' not in agent_state:
            return []
        
        # Get incoming message
        msg_data = agent_state['incoming_message']
        
        # Convert to Message object
        message = Message(
            from_agent=msg_data['from'],
            to_agent=self.agent_id,
            message_type=MessageType(msg_data['type']),
            content=msg_data['content'],
            timestamp=msg_data['timestamp']
        )
        
        # Store
        msg_id = msg_data['id']
        if msg_id not in self.received_messages:
            self.received_messages[msg_id] = message
            return [message]
        
        return []
