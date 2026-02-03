"""
AI-Powered TUI Assistant: Natural language interface for agent control.

This component provides intelligent natural language processing capabilities
to control and monitor LRS agents through conversational interfaces.
"""

import asyncio
import json
import re
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum

from ...multi_agent.shared_state import SharedWorldState


class IntentType(Enum):
    """Types of user intents."""

    QUERY = "query"
    CONTROL = "control"
    MONITOR = "monitor"
    COORDINATE = "coordinate"
    OPTIMIZE = "optimize"
    DEBUG = "debug"
    EXPLAIN = "explain"
    PREDICT = "predict"


class Confidence(Enum):
    """Confidence levels for intent recognition."""

    VERY_HIGH = "very_high"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    VERY_LOW = "very_low"


@dataclass
class Intent:
    """Recognized user intent."""

    intent_type: IntentType
    entities: Dict[str, Any]
    confidence: Confidence
    original_text: str
    processed_at: datetime
    suggested_actions: List[str] = field(default_factory=list)


@dataclass
class ConversationContext:
    """Context for ongoing conversation."""

    conversation_id: str
    user_id: Optional[str]
    messages: List[Dict[str, Any]] = field(default_factory=list)
    entities: Dict[str, Any] = field(default_factory=dict)
    preferences: Dict[str, Any] = field(default_factory=dict)
    started_at: datetime = field(default_factory=datetime.now)
    last_activity: datetime = field(default_factory=datetime.now)


@dataclass
class AssistantResponse:
    """Response from AI assistant."""

    text: str
    intent: Intent
    actions_taken: List[str] = field(default_factory=list)
    data_provided: Dict[str, Any] = field(default_factory=dict)
    follow_up_questions: List[str] = field(default_factory=list)
    confidence: Confidence = Confidence.MEDIUM
    response_time: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)


class TUIAIAssistant:
    """
    AI-powered assistant for natural language agent control.

    Features:
    - Natural language intent recognition
    - Conversational context management
    - Multi-turn dialogue handling
    - Intelligent agent control suggestions
    - Real-time query processing
    - Adaptive learning from interactions

    Examples:
        >>> assistant = TUIAIAssistant(shared_state, tui_bridge)
        >>>
        >>> # Process natural language query
        >>> response = await assistant.process_query(
        ...     "Show me agents with low precision",
        ...     conversation_id="conv_123"
        ... )
        >>>
        >>> # Handle agent control command
        >>> response = await assistant.process_query(
        ...     "Restart agent_1 and reset its precision",
        ...     conversation_id="conv_123"
        ... )
    """

    def __init__(
        self, shared_state: SharedWorldState, tui_bridge, llm_client: Optional[Any] = None
    ):
        """
        Initialize AI Assistant.

        Args:
            shared_state: LRS shared world state
            tui_bridge: TUI bridge instance
            llm_client: Optional LLM client for advanced processing
        """
        self.shared_state = shared_state
        self.tui_bridge = tui_bridge
        self.llm_client = llm_client

        # Conversation management
        self.conversations: Dict[str, ConversationContext] = {}
        self.active_contexts: Dict[str, str] = {}  # user_id -> conversation_id

        # Intent patterns and rules
        self.intent_patterns = self._setup_intent_patterns()
        self.entity_extractors = self._setup_entity_extractors()

        # Response templates
        self.response_templates = self._setup_response_templates()

        # Learning and adaptation
        self.interaction_history: List[Dict[str, Any]] = []
        self.success_patterns: Dict[str, int] = {}

        self.logger = logging.getLogger(__name__)

        # Start background tasks
        self._start_background_tasks()

    async def process_query(
        self, query: str, conversation_id: Optional[str] = None, user_id: Optional[str] = None
    ) -> AssistantResponse:
        """
        Process natural language query.

        Args:
            query: User's natural language query
            conversation_id: Optional conversation ID
            user_id: Optional user identifier

        Returns:
            Assistant response with actions and data
        """
        start_time = datetime.now()

        try:
            # Get or create conversation context
            context = self._get_or_create_context(conversation_id, user_id)

            # Add user message to context
            context.messages.append(
                {"type": "user", "text": query, "timestamp": datetime.now().isoformat()}
            )
            context.last_activity = datetime.now()

            # Recognize intent
            intent = await self._recognize_intent(query, context)

            # Process intent and generate response
            response = await self._process_intent(intent, context)

            # Add assistant response to context
            context.messages.append(
                {
                    "type": "assistant",
                    "text": response.text,
                    "intent": intent.intent_type.value,
                    "timestamp": datetime.now().isoformat(),
                }
            )

            # Calculate response time
            response_time = (datetime.now() - start_time).total_seconds()
            response.response_time = response_time

            # Log interaction for learning
            self._log_interaction(query, intent, response, response_time)

            return response

        except Exception as e:
            self.logger.error(f"Error processing query '{query}': {e}")

            return AssistantResponse(
                text=f"I apologize, but I encountered an error processing your request: {str(e)}",
                intent=Intent(
                    intent_type=IntentType.QUERY,
                    entities={},
                    confidence=Confidence.VERY_LOW,
                    original_text=query,
                    processed_at=datetime.now(),
                ),
                confidence=Confidence.VERY_LOW,
                response_time=(datetime.now() - start_time).total_seconds(),
            )

    async def get_conversation_history(
        self, conversation_id: str, limit: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Get conversation history.

        Args:
            conversation_id: Conversation identifier
            limit: Maximum number of messages

        Returns:
            List of conversation messages
        """
        if conversation_id not in self.conversations:
            return []

        context = self.conversations[conversation_id]
        return context.messages[-limit:]

    async def clear_conversation(self, conversation_id: str) -> bool:
        """
        Clear conversation context.

        Args:
            conversation_id: Conversation identifier

        Returns:
            Success status
        """
        if conversation_id in self.conversations:
            del self.conversations[conversation_id]

            # Remove from active contexts
            for user_id, conv_id in list(self.active_contexts.items()):
                if conv_id == conversation_id:
                    del self.active_contexts[user_id]

            return True

        return False

    async def _recognize_intent(self, query: str, context: ConversationContext) -> Intent:
        """
        Recognize user intent from natural language.

        Args:
            query: User's query
            context: Conversation context

        Returns:
            Recognized intent with entities
        """
        query_lower = query.lower()

        # Try pattern-based recognition first
        for intent_type, patterns in self.intent_patterns.items():
            for pattern in patterns:
                match = re.search(pattern, query_lower, re.IGNORECASE)
                if match:
                    entities = self._extract_entities(query, match, intent_type)
                    confidence = self._calculate_confidence(query, pattern, entities)

                    return Intent(
                        intent_type=intent_type,
                        entities=entities,
                        confidence=confidence,
                        original_text=query,
                        processed_at=datetime.now(),
                    )

        # Fallback to LLM-based recognition if available
        if self.llm_client:
            return await self._llm_intent_recognition(query, context)

        # Default fallback
        return Intent(
            intent_type=IntentType.QUERY,
            entities={"raw_query": query},
            confidence=Confidence.LOW,
            original_text=query,
            processed_at=datetime.now(),
        )

    async def _process_intent(
        self, intent: Intent, context: ConversationContext
    ) -> AssistantResponse:
        """
        Process recognized intent and generate response.

        Args:
            intent: Recognized intent
            context: Conversation context

        Returns:
            Assistant response
        """
        if intent.intent_type == IntentType.QUERY:
            return await self._handle_query_intent(intent, context)

        elif intent.intent_type == IntentType.CONTROL:
            return await self._handle_control_intent(intent, context)

        elif intent.intent_type == IntentType.MONITOR:
            return await self._handle_monitor_intent(intent, context)

        elif intent.intent_type == IntentType.COORDINATE:
            return await self._handle_coordinate_intent(intent, context)

        elif intent.intent_type == IntentType.OPTIMIZE:
            return await self._handle_optimize_intent(intent, context)

        elif intent.intent_type == IntentType.DEBUG:
            return await self._handle_debug_intent(intent, context)

        elif intent.intent_type == IntentType.EXPLAIN:
            return await self._handle_explain_intent(intent, context)

        elif intent.intent_type == IntentType.PREDICT:
            return await self._handle_predict_intent(intent, context)

        else:
            return self._generate_fallback_response(intent)

    async def _handle_query_intent(
        self, intent: Intent, context: ConversationContext
    ) -> AssistantResponse:
        """Handle query-type intents."""

        query_type = intent.entities.get("query_type", "general")

        if query_type == "agent_status":
            return await self._query_agent_status(intent.entities)

        elif query_type == "precision_info":
            return await self._query_precision_info(intent.entities)

        elif query_type == "system_health":
            return await self._query_system_health(intent.entities)

        elif query_type == "recent_activity":
            return await self._query_recent_activity(intent.entities)

        else:
            return await self._general_query(intent.entities)

    async def _handle_control_intent(
        self, intent: Intent, context: ConversationContext
    ) -> AssistantResponse:
        """Handle control-type intents."""

        action = intent.entities.get("action")
        agent_id = intent.entities.get("agent_id")

        if not agent_id:
            return AssistantResponse(
                text="Which agent would you like me to control?",
                intent=intent,
                confidence=Confidence.MEDIUM,
                follow_up_questions=["Please specify the agent ID"],
            )

        actions_taken = []

        try:
            if action == "restart":
                # Restart agent
                if hasattr(self.tui_bridge, "restart_agent"):
                    await self.tui_bridge.restart_agent(agent_id)
                    actions_taken.append(f"Restarted agent {agent_id}")
                else:
                    # Simulate restart via state update
                    self.shared_state.update(
                        agent_id,
                        {"status": "restarting", "restarted_at": datetime.now().isoformat()},
                    )
                    actions_taken.append(f"Initiated restart for agent {agent_id}")

            elif action == "stop":
                self.shared_state.update(
                    agent_id, {"status": "stopped", "stopped_at": datetime.now().isoformat()}
                )
                actions_taken.append(f"Stopped agent {agent_id}")

            elif action == "reset_precision":
                self.shared_state.update(
                    agent_id,
                    {
                        "precision": {"alpha": 1.0, "beta": 1.0, "value": 0.5},
                        "precision_reset_at": datetime.now().isoformat(),
                    },
                )
                actions_taken.append(f"Reset precision for agent {agent_id}")

            elif action == "pause":
                self.shared_state.update(
                    agent_id, {"status": "paused", "paused_at": datetime.now().isoformat()}
                )
                actions_taken.append(f"Paused agent {agent_id}")

            elif action == "resume":
                self.shared_state.update(
                    agent_id, {"status": "active", "resumed_at": datetime.now().isoformat()}
                )
                actions_taken.append(f"Resumed agent {agent_id}")

            else:
                return AssistantResponse(
                    text=f"I'm not sure how to '{action}' agent {agent_id}. Available actions: restart, stop, reset_precision, pause, resume.",
                    intent=intent,
                    confidence=Confidence.LOW,
                )

            # Verify action was successful
            agent_state = self.shared_state.get_agent_state(agent_id)

            response_text = f"I've {', '.join(actions_taken)}. "
            response_text += (
                f"The agent's current status is {agent_state.get('status', 'unknown')}."
            )

            return AssistantResponse(
                text=response_text,
                intent=intent,
                actions_taken=actions_taken,
                data_provided={"agent_state": agent_state},
                confidence=Confidence.HIGH,
            )

        except Exception as e:
            return AssistantResponse(
                text=f"I encountered an error trying to control agent {agent_id}: {str(e)}",
                intent=intent,
                confidence=Confidence.LOW,
            )

    async def _handle_monitor_intent(
        self, intent: Intent, context: ConversationContext
    ) -> AssistantResponse:
        """Handle monitor-type intents."""

        monitor_type = intent.entities.get("monitor_type", "general")

        if monitor_type == "precision":
            return await self._monitor_precision(intent.entities)

        elif monitor_type == "performance":
            return await self._monitor_performance(intent.entities)

        elif monitor_type == "resources":
            return await self._monitor_resources(intent.entities)

        else:
            return await self._general_monitoring(intent.entities)

    async def _query_agent_status(self, entities: Dict[str, Any]) -> AssistantResponse:
        """Query agent status."""

        agent_id = entities.get("agent_id")

        if agent_id:
            # Query specific agent
            agent_state = self.shared_state.get_agent_state(agent_id)

            if not agent_state:
                return AssistantResponse(
                    text=f"Agent {agent_id} not found.", confidence=Confidence.HIGH
                )

            status = agent_state.get("status", "unknown")
            precision = agent_state.get("precision", {})
            last_update = agent_state.get("last_update", "unknown")

            response_text = f"Agent {agent_id} status:\n"
            response_text += f"- Status: {status}\n"
            response_text += f"- Precision: {precision.get('value', 'N/A'):.2f}\n"
            response_text += f"- Last Update: {last_update}"

            return AssistantResponse(
                text=response_text,
                data_provided={"agent_state": agent_state},
                confidence=Confidence.HIGH,
            )

        else:
            # Query all agents
            all_states = self.shared_state.get_all_states()

            if not all_states:
                return AssistantResponse(
                    text="No agents are currently running.", confidence=Confidence.HIGH
                )

            response_text = f"Found {len(all_states)} active agents:\n"

            for agent_id, state in all_states.items():
                status = state.get("status", "unknown")
                precision = state.get("precision", {}).get("value", "N/A")
                response_text += f"- {agent_id}: {status} (precision: {precision})\n"

            return AssistantResponse(
                text=response_text,
                data_provided={"agent_count": len(all_states), "agents": list(all_states.keys())},
                confidence=Confidence.HIGH,
            )

    def _setup_intent_patterns(self) -> Dict[IntentType, List[str]]:
        """Setup regex patterns for intent recognition."""

        return {
            IntentType.QUERY: [
                r"(show|tell me|what is|how many|list).*agent",
                r"(show|tell me).*precision",
                r"(what|how).*status",
                r"(show|tell me).*system",
                r"(recent|latest).*activity",
            ],
            IntentType.CONTROL: [
                r"(restart|reboot|reset).*agent",
                r"(stop|shutdown).*agent",
                r"(pause|suspend).*agent",
                r"(resume|continue).*agent",
                r"(reset|clear).*precision",
            ],
            IntentType.MONITOR: [
                r"(monitor|watch|track).*agent",
                r"(monitor|watch).*precision",
                r"(monitor|watch).*performance",
                r"(monitor|watch).*resources",
            ],
            IntentType.COORDINATE: [
                r"(coordinate|orchestrate).*agent",
                r"(work together|collaborate).*agent",
                r"(team up|group).*task",
            ],
            IntentType.OPTIMIZE: [
                r"(optimize|improve|tune).*agent",
                r"(optimize|improve).*precision",
                r"(optimize|improve).*performance",
            ],
            IntentType.DEBUG: [
                r"(debug|troubleshoot).*agent",
                r"(what.*wrong|why.*failing).*agent",
                r"(investigate|diagnose).*agent",
            ],
            IntentType.EXPLAIN: [
                r"(explain|why).*agent",
                r"(how.*work).*agent",
                r"(tell me about).*agent",
            ],
            IntentType.PREDICT: [
                r"(predict|forecast).*agent",
                r"(what.*next|will.*agent).*",
                r"(expect|anticipate).*agent",
            ],
        }

    def _setup_entity_extractors(self) -> Dict[str, Any]:
        """Setup entity extraction patterns."""

        return {
            "agent_id": [r"agent[_\s](\w+)", r"id[_\s](\w+)", r"(\w+)(?=\s+agent)"],
            "action": [
                r"(restart|reboot|reset|stop|shutdown|pause|suspend|resume|continue)",
                r"(coordinate|orchestrate|collaborate|optimize|debug)",
            ],
            "query_type": [
                r"(status|health|performance|precision|resources)",
                r"(recent|latest|current|overall)",
            ],
            "monitor_type": [r"(precision|performance|resources|activity)", r"(system|agent|tool)"],
        }

    def _setup_response_templates(self) -> Dict[str, str]:
        """Setup response templates."""

        return {
            "agent_not_found": "I couldn't find agent {agent_id}. Available agents: {available_agents}",
            "action_successful": "I've successfully {action} agent {agent_id}.",
            "action_failed": "I encountered an error trying to {action} agent {agent_id}: {error}",
            "unclear_intent": "I'm not sure what you want me to do. Could you please rephrase that?",
            "no_agents": "No agents are currently running.",
            "general_info": "Here's what I found: {info}",
        }

    def _extract_entities(
        self, query: str, match: re.Match, intent_type: IntentType
    ) -> Dict[str, Any]:
        """Extract entities from query."""

        entities = {"raw_query": query}

        # Extract agent IDs
        for pattern in self.entity_extractors["agent_id"]:
            agent_matches = re.findall(pattern, query, re.IGNORECASE)
            if agent_matches:
                entities["agent_id"] = agent_matches[0]
                break

        # Extract actions
        for pattern in self.entity_extractors["action"]:
            action_matches = re.findall(pattern, query, re.IGNORECASE)
            if action_matches:
                entities["action"] = action_matches[0].lower()
                break

        # Extract query types
        for pattern in self.entity_extractors["query_type"]:
            query_matches = re.findall(pattern, query, re.IGNORECASE)
            if query_matches:
                entities["query_type"] = query_matches[0].lower()
                break

        # Extract monitor types
        for pattern in self.entity_extractors["monitor_type"]:
            monitor_matches = re.findall(pattern, query, re.IGNORECASE)
            if monitor_matches:
                entities["monitor_type"] = monitor_matches[0].lower()
                break

        return entities

    def _calculate_confidence(
        self, query: str, pattern: str, entities: Dict[str, Any]
    ) -> Confidence:
        """Calculate confidence level for intent recognition."""

        confidence_score = 0.5  # Base confidence

        # Higher confidence for more specific patterns
        if len(pattern) > 20:
            confidence_score += 0.2

        # Higher confidence for extracted entities
        if entities.get("agent_id"):
            confidence_score += 0.2

        if entities.get("action"):
            confidence_score += 0.1

        # Convert to confidence enum
        if confidence_score >= 0.9:
            return Confidence.VERY_HIGH
        elif confidence_score >= 0.7:
            return Confidence.HIGH
        elif confidence_score >= 0.5:
            return Confidence.MEDIUM
        elif confidence_score >= 0.3:
            return Confidence.LOW
        else:
            return Confidence.VERY_LOW

    def _get_or_create_context(
        self, conversation_id: Optional[str], user_id: Optional[str]
    ) -> ConversationContext:
        """Get existing or create new conversation context."""

        if conversation_id and conversation_id in self.conversations:
            return self.conversations[conversation_id]

        # Create new conversation
        new_conversation_id = conversation_id or f"conv_{datetime.now().timestamp()}"

        context = ConversationContext(conversation_id=new_conversation_id, user_id=user_id)

        self.conversations[new_conversation_id] = context

        if user_id:
            self.active_contexts[user_id] = new_conversation_id

        return context

    def _log_interaction(
        self, query: str, intent: Intent, response: AssistantResponse, response_time: float
    ):
        """Log interaction for learning and analysis."""

        interaction = {
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "intent_type": intent.intent_type.value,
            "confidence": intent.confidence.value,
            "response_time": response_time,
            "success": response.confidence in [Confidence.HIGH, Confidence.VERY_HIGH],
        }

        self.interaction_history.append(interaction)

        # Keep only recent interactions
        if len(self.interaction_history) > 10000:
            self.interaction_history = self.interaction_history[-5000:]

        # Update success patterns
        intent_key = intent.intent_type.value
        if interaction["success"]:
            self.success_patterns[intent_key] = self.success_patterns.get(intent_key, 0) + 1

    async def _llm_intent_recognition(self, query: str, context: ConversationContext) -> Intent:
        """Use LLM for advanced intent recognition."""

        if not self.llm_client:
            # Fallback to pattern-based
            return Intent(
                intent_type=IntentType.QUERY,
                entities={"raw_query": query},
                confidence=Confidence.MEDIUM,
                original_text=query,
                processed_at=datetime.now(),
            )

        # Prepare prompt for LLM
        prompt = f"""
        Analyze this user query and classify the intent:
        
        Query: "{query}"
        
        Possible intents:
        - QUERY: Asking for information
        - CONTROL: Telling agents what to do
        - MONITOR: Watching agent activity
        - COORDINATE: Managing multiple agents
        - OPTIMIZE: Improving agent performance
        - DEBUG: Troubleshooting issues
        - EXPLAIN: Understanding agent behavior
        - PREDICT: Forecasting agent behavior
        
        Previous conversation context:
        {json.dumps(context.messages[-3:], indent=2)}
        
        Respond with JSON format:
        {{
            "intent_type": "INTENT_NAME",
            "entities": {{"agent_id": "...", "action": "...", ...}},
            "confidence": "high|medium|low"
        }}
        """

        try:
            response = await self.llm_client.generate(prompt)
            result = json.loads(response)

            return Intent(
                intent_type=IntentType(result["intent_type"]),
                entities=result["entities"],
                confidence=Confidence(result["confidence"]),
                original_text=query,
                processed_at=datetime.now(),
            )

        except Exception as e:
            self.logger.error(f"LLM intent recognition failed: {e}")

            # Fallback to pattern-based
            return Intent(
                intent_type=IntentType.QUERY,
                entities={"raw_query": query},
                confidence=Confidence.MEDIUM,
                original_text=query,
                processed_at=datetime.now(),
            )

    def _start_background_tasks(self):
        """Start background tasks for maintenance and learning."""

        # Task to clean up old conversations
        asyncio.create_task(self._cleanup_conversations())

        # Task to analyze interaction patterns
        asyncio.create_task(self._analyze_patterns())

    async def _cleanup_conversations(self):
        """Clean up old conversation contexts."""

        while True:
            try:
                cutoff_time = datetime.now() - timedelta(hours=24)

                # Remove old conversations
                old_conversations = [
                    conv_id
                    for conv_id, context in self.conversations.items()
                    if context.last_activity < cutoff_time
                ]

                for conv_id in old_conversations:
                    del self.conversations[conv_id]

                if old_conversations:
                    self.logger.info(f"Cleaned up {len(old_conversations)} old conversations")

                await asyncio.sleep(3600)  # Check every hour

            except Exception as e:
                self.logger.error(f"Error in conversation cleanup: {e}")
                await asyncio.sleep(300)  # Retry in 5 minutes

    async def _analyze_patterns(self):
        """Analyze interaction patterns for improvement."""

        while True:
            try:
                if len(self.interaction_history) >= 100:
                    # Analyze recent patterns
                    recent_interactions = self.interaction_history[-100:]

                    # Calculate success rates by intent type
                    intent_stats = {}
                    for interaction in recent_interactions:
                        intent_type = interaction["intent_type"]
                        if intent_type not in intent_stats:
                            intent_stats[intent_type] = {"success": 0, "total": 0}

                        intent_stats[intent_type]["total"] += 1
                        if interaction["success"]:
                            intent_stats[intent_type]["success"] += 1

                    # Log insights
                    for intent_type, stats in intent_stats.items():
                        if stats["total"] >= 10:
                            success_rate = stats["success"] / stats["total"]
                            if success_rate < 0.7:
                                self.logger.warning(
                                    f"Low success rate for {intent_type}: {success_rate:.2%}"
                                )

                await asyncio.sleep(1800)  # Analyze every 30 minutes

            except Exception as e:
                self.logger.error(f"Error in pattern analysis: {e}")
                await asyncio.sleep(300)
