"""
OpenAI Assistants API integration for LRS-Agents.

Allows LRS agents to use OpenAI Assistants as policy generators while
maintaining Active Inference dynamics for selection and adaptation.
"""

from typing import Dict, List, Optional, Any
import json
import time
from openai import OpenAI
from openai.types.beta import Assistant, Thread
from openai.types.beta.threads import Run

from lrs.core.lens import ToolLens, ExecutionResult
from lrs.inference.prompts import MetaCognitivePrompter


class OpenAIAssistantLens(ToolLens):
    """
    Wraps OpenAI Assistant as a ToolLens for LRS integration.
    
    The assistant generates policy proposals, while LRS evaluates them
    via Expected Free Energy and tracks precision.
    
    Examples:
        >>> from openai import OpenAI
        >>> 
        >>> client = OpenAI(api_key="...")
        >>> assistant = client.beta.assistants.create(
        ...     name="Policy Generator",
        ...     instructions="Generate diverse policy proposals",
        ...     model="gpt-4-turbo-preview"
        ... )
        >>> 
        >>> lens = OpenAIAssistantLens(client, assistant.id)
        >>> result = lens.get({"query": "Fetch data from API"})
    """
    
    def __init__(
        self,
        client: OpenAI,
        assistant_id: str,
        thread_id: Optional[str] = None,
        temperature: float = 0.7,
        max_wait: int = 30
    ):
        """
        Initialize OpenAI Assistant wrapper.
        
        Args:
            client: OpenAI client instance
            assistant_id: ID of the assistant to use
            thread_id: Optional existing thread ID (creates new if None)
            temperature: Sampling temperature (will be adapted by precision)
            max_wait: Maximum seconds to wait for assistant response
        """
        super().__init__(
            name="openai_assistant",
            input_schema={
                'type': 'object',
                'required': ['query'],
                'properties': {
                    'query': {'type': 'string'},
                    'precision': {'type': 'number'}
                }
            },
            output_schema={
                'type': 'object',
                'properties': {
                    'proposals': {'type': 'array'}
                }
            }
        )
        
        self.client = client
        self.assistant_id = assistant_id
        self.base_temperature = temperature
        self.max_wait = max_wait
        
        # Create or use existing thread
        if thread_id:
            self.thread_id = thread_id
        else:
            thread = self.client.beta.threads.create()
            self.thread_id = thread.id
    
    def get(self, state: dict) -> ExecutionResult:
        """
        Query OpenAI Assistant for policy proposals.
        
        Args:
            state: Must contain 'query' and optionally 'precision'
        
        Returns:
            ExecutionResult with proposals or error
        """
        self.call_count += 1
        
        try:
            query = state.get('query', 'Generate policy proposals')
            precision = state.get('precision', 0.5)
            
            # Adapt temperature based on precision
            adapted_temp = self._adapt_temperature(precision)
            
            # Create message in thread
            self.client.beta.threads.messages.create(
                thread_id=self.thread_id,
                role="user",
                content=query
            )
            
            # Run assistant
            run = self.client.beta.threads.runs.create(
                thread_id=self.thread_id,
                assistant_id=self.assistant_id,
                temperature=adapted_temp
            )
            
            # Wait for completion
            response = self._wait_for_completion(run.id)
            
            # Parse proposals
            proposals = self._parse_proposals(response)
            
            return ExecutionResult(
                success=True,
                value={'proposals': proposals},
                error=None,
                prediction_error=0.1
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
        """Update state with assistant proposals"""
        return {
            **state,
            'assistant_proposals': observation.get('proposals', []),
            'last_assistant_query': state.get('query')
        }
    
    def _adapt_temperature(self, precision: float) -> float:
        """Adapt temperature based on precision"""
        return self.base_temperature * (1.0 / (precision + 0.1))
    
    def _wait_for_completion(self, run_id: str) -> str:
        """Wait for assistant run to complete"""
        start_time = time.time()
        
        while time.time() - start_time < self.max_wait:
            run = self.client.beta.threads.runs.retrieve(
                thread_id=self.thread_id,
                run_id=run_id
            )
            
            if run.status == 'completed':
                messages = self.client.beta.threads.messages.list(
                    thread_id=self.thread_id,
                    order='desc',
                    limit=1
                )
                
                if messages.data:
                    return messages.data[0].content[0].text.value
                else:
                    raise ValueError("No messages returned")
            
            elif run.status in ['failed', 'cancelled', 'expired']:
                raise RuntimeError(f"Run failed with status: {run.status}")
            
            time.sleep(1)
        
        raise TimeoutError(f"Assistant didn't respond within {self.max_wait}s")
    
    def _parse_proposals(self, response: str) -> List[Dict]:
        """Parse assistant response into structured proposals"""
        try:
            data = json.loads(response)
            if isinstance(data, dict) and 'proposals' in data:
                return data['proposals']
            elif isinstance(data, list):
                return data
            else:
                raise ValueError("Unexpected response format")
        except json.JSONDecodeError:
            return [{
                'policy_id': 1,
                'description': response,
                'estimated_success_prob': 0.5,
                'strategy': 'unknown'
            }]


class OpenAIAssistantPolicyGenerator:
    """
    High-level interface for using OpenAI Assistants as policy generators.
    
    Examples:
        >>> from openai import OpenAI
        >>> 
        >>> client = OpenAI()
        >>> generator = OpenAIAssistantPolicyGenerator(
        ...     client=client,
        ...     model="gpt-4-turbo-preview"
        ... )
        >>> 
        >>> proposals = generator.generate_proposals(
        ...     state={'goal': 'Fetch data'},
        ...     precision=0.3,
        ...     tool_registry=registry
        ... )
    """
    
    def __init__(
        self,
        client: OpenAI,
        model: str = "gpt-4-turbo-preview",
        assistant_id: Optional[str] = None
    ):
        """Initialize policy generator"""
        self.client = client
        self.model = model
        
        if assistant_id:
            self.assistant_id = assistant_id
        else:
            self.assistant_id = self._create_assistant()
        
        self.lens = OpenAIAssistantLens(client, self.assistant_id)
    
    def _create_assistant(self) -> str:
        """Create assistant with LRS-specific instructions"""
        instructions = """You are a Bayesian policy generator for an Active Inference agent.

Your role is to PROPOSE diverse policy candidates, not to DECIDE which is best.
The agent will evaluate your proposals using Expected Free Energy (G).

Generate 3-5 policy proposals in JSON format:

{
  "proposals": [
    {
      "policy_id": 1,
      "tools": ["tool_name_1", "tool_name_2"],
      "description": "Brief strategy description",
      "estimated_success_prob": 0.8,
      "expected_information_gain": 0.3,
      "strategy": "exploit|explore|balanced",
      "failure_modes": ["What could go wrong"]
    }
  ]
}

Adapt your strategy based on the agent's precision (confidence):
- HIGH precision (>0.7): Focus on exploitation (proven patterns)
- LOW precision (<0.4): Focus on exploration (gather information)
- MEDIUM precision: Balance both

Ensure diversity across the exploration-exploitation spectrum."""
        
        assistant = self.client.beta.assistants.create(
            name="LRS Policy Generator",
            instructions=instructions,
            model=self.model,
            response_format={"type": "json_object"}
        )
        
        return assistant.id
    
    def generate_proposals(
        self,
        state: Dict,
        precision: float,
        tool_registry: Dict[str, Any]
    ) -> List[Dict]:
        """Generate policy proposals using assistant"""
        tool_list = "\n".join([
            f"- {name}: {tool.get('description', 'No description')}"
            for name, tool in tool_registry.items()
        ])
        
        query = f"""Goal: {state.get('goal', 'Unknown')}

Available Tools:
{tool_list}

Current Precision: {precision:.3f}

Generate policy proposals appropriate for this precision level."""
        
        result = self.lens.get({
            'query': query,
            'precision': precision
        })
        
        if result.success:
            return result.value.get('proposals', [])
        else:
            return []


def create_openai_lrs_agent(
    client: OpenAI,
    tools: List[ToolLens],
    model: str = "gpt-4-turbo-preview",
    **kwargs
) -> Any:
    """
    Create LRS agent using OpenAI Assistant for policy generation.
    
    Examples:
        >>> from openai import OpenAI
        >>> 
        >>> client = OpenAI(api_key="...")
        >>> tools = [ShellTool(), PythonTool()]
        >>> 
        >>> agent = create_openai_lrs_agent(client, tools)
        >>> result = agent.invoke({
        ...     "messages": [{"role": "user", "content": "Task"}]
        ... })
    """
    from lrs import create_lrs_agent
    from lrs.core.registry import ToolRegistry
    
    registry = ToolRegistry()
    for tool in tools:
        registry.register(tool)
    
    generator = OpenAIAssistantPolicyGenerator(client, model)
    
    from lrs.integration.langgraph import LRSGraphBuilder
    
    builder = LRSGraphBuilder(
        llm=generator,
        registry=registry,
        **kwargs
    )
    
    return builder.build()
