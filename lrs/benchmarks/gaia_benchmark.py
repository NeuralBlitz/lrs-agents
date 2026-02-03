"""
GAIA (General AI Assistants) benchmark integration.

Tests LRS agents on real-world tasks requiring:
- Multi-step reasoning
- Tool use
- File handling
- Web search
"""

import json
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from pathlib import Path
import time

from lrs.core.lens import ToolLens, ExecutionResult
from lrs.monitoring.structured_logging import LRSLogger


@dataclass
class GAIATask:
    """
    Single GAIA benchmark task.
    
    Attributes:
        task_id: Unique task identifier
        question: Task question
        level: Difficulty level (1=easy, 2=medium, 3=hard)
        final_answer: Expected answer
        file_name: Optional attached file name
        file_path: Optional attached file path
        annotator_metadata: Optional metadata from human annotators
    """
    task_id: str
    question: str
    level: int
    final_answer: str
    file_name: Optional[str] = None
    file_path: Optional[str] = None
    annotator_metadata: Optional[Dict] = None


# Tool implementations for GAIA

class FileReadTool(ToolLens):
    """Read file contents"""
    
    def __init__(self):
        super().__init__(
            name="read_file",
            input_schema={
                'type': 'object',
                'required': ['path'],
                'properties': {
                    'path': {'type': 'string'}
                }
            },
            output_schema={'type': 'string'}
        )
    
    def get(self, state: dict) -> ExecutionResult:
        self.call_count += 1
        path = state.get('path', '')
        
        # Validate path
        if not path or '..' in path:
            self.failure_count += 1
            return ExecutionResult(
                success=False,
                value=None,
                error="Invalid path",
                prediction_error=0.9
            )
        
        try:
            content = Path(path).read_text()
            return ExecutionResult(
                success=True,
                value=content,
                error=None,
                prediction_error=0.05
            )
        except Exception as e:
            self.failure_count += 1
            return ExecutionResult(
                success=False,
                value=None,
                error=str(e),
                prediction_error=0.8
            )
    
    def set(self, state: dict, observation: str) -> dict:
        return {**state, 'file_content': observation}


class WebSearchTool(ToolLens):
    """Web search (mock implementation)"""
    
    def __init__(self):
        super().__init__(
            name="web_search",
            input_schema={
                'type': 'object',
                'required': ['query'],
                'properties': {
                    'query': {'type': 'string'}
                }
            },
            output_schema={'type': 'string'}
        )
    
    def get(self, state: dict) -> ExecutionResult:
        self.call_count += 1
        query = state.get('query', '')
        
        # Mock search (would integrate with real API in production)
        # For now, simulate occasional rate limiting
        import random
        if random.random() < 0.1:  # 10% rate limit
            self.failure_count += 1
            return ExecutionResult(
                success=False,
                value=None,
                error="Rate limited",
                prediction_error=0.7
            )
        
        # Mock results
        results = f"Search results for '{query}': [Mock data]"
        
        return ExecutionResult(
            success=True,
            value=results,
            error=None,
            prediction_error=0.2
        )
    
    def set(self, state: dict, observation: str) -> dict:
        return {**state, 'search_results': observation}


class CalculatorTool(ToolLens):
    """Evaluate mathematical expressions"""
    
    def __init__(self):
        super().__init__(
            name="calculator",
            input_schema={
                'type': 'object',
                'required': ['expression'],
                'properties': {
                    'expression': {'type': 'string'}
                }
            },
            output_schema={'type': 'number'}
        )
    
    def get(self, state: dict) -> ExecutionResult:
        self.call_count += 1
        expression = state.get('expression', '')
        
        try:
            # Safe eval (restrict to math operations)
            allowed_names = {
                'abs': abs, 'round': round, 'min': min, 'max': max,
                'sum': sum, 'pow': pow
            }
            result = eval(expression, {"__builtins__": {}}, allowed_names)
            
            return ExecutionResult(
                success=True,
                value=result,
                error=None,
                prediction_error=0.0  # Math is deterministic
            )
        except Exception as e:
            self.failure_count += 1
            return ExecutionResult(
                success=False,
                value=None,
                error=str(e),
                prediction_error=0.9
            )
    
    def set(self, state: dict, observation: float) -> dict:
        return {**state, 'calculation_result': observation}


class PythonExecutorTool(ToolLens):
    """Execute Python code"""
    
    def __init__(self):
        super().__init__(
            name="python_exec",
            input_schema={
                'type': 'object',
                'required': ['code'],
                'properties': {
                    'code': {'type': 'string'}
                }
            },
            output_schema={'type': 'string'}
        )
    
    def get(self, state: dict) -> ExecutionResult:
        self.call_count += 1
        code = state.get('code', '')
        
        try:
            # Execute in restricted environment
            namespace = {'__builtins__': __builtins__}
            exec(code, namespace)
            result = namespace.get('result', 'Executed')
            
            return ExecutionResult(
                success=True,
                value=str(result),
                error=None,
                prediction_error=0.1
            )
        except Exception as e:
            self.failure_count += 1
            return ExecutionResult(
                success=False,
                value=None,
                error=str(e),
                prediction_error=0.8
            )
    
    def set(self, state: dict, observation: str) -> dict:
        return {**state, 'python_output': observation}


class WikipediaTool(ToolLens):
    """Wikipedia search (mock)"""
    
    def __init__(self):
        super().__init__(
            name="wikipedia",
            input_schema={
                'type': 'object',
                'required': ['query'],
                'properties': {
                    'query': {'type': 'string'}
                }
            },
            output_schema={'type': 'string'}
        )
    
    def get(self, state: dict) -> ExecutionResult:
        self.call_count += 1
        query = state.get('query', '')
        
        # Mock Wikipedia lookup
        summary = f"Wikipedia summary for '{query}': [Mock article content]"
        
        return ExecutionResult(
            success=True,
            value=summary,
            error=None,
            prediction_error=0.15
        )
    
    def set(self, state: dict, observation: str) -> dict:
        return {**state, 'wiki_content': observation}


class GAIAToolkit:
    """Standard toolkit for GAIA benchmark"""
    
    @staticmethod
    def create_tools() -> List[ToolLens]:
        """Create standard GAIA tool set"""
        return [
            FileReadTool(),
            WebSearchTool(),
            CalculatorTool(),
            PythonExecutorTool(),
            WikipediaTool()
        ]


class GAIABenchmark:
    """
    GAIA benchmark runner for LRS agents.
    
    Examples:
        >>> from langchain_anthropic import ChatAnthropic
        >>> 
        >>> llm = ChatAnthropic(model="claude-sonnet-4-20250514")
        >>> benchmark = GAIABenchmark(llm=llm, log_dir="logs/gaia")
        >>> 
        >>> results = benchmark.run(
        ...     tasks_file="gaia_validation.jsonl",
        ...     level_filter=1  # Only level 1 tasks
        ... )
        >>> 
        >>> print(f"Overall: {results['overall']['success_rate']:.1%}")
    """
    
    def __init__(
        self,
        llm: Any,
        log_dir: str = "logs/gaia",
        max_steps: int = 20
    ):
        """
        Initialize GAIA benchmark.
        
        Args:
            llm: Language model for agent
            log_dir: Directory for logs
            max_steps: Max steps per task
        """
        self.llm = llm
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.max_steps = max_steps
    
    def load_tasks(self, tasks_file: str) -> List[GAIATask]:
        """
        Load tasks from JSONL file.
        
        Args:
            tasks_file: Path to GAIA tasks file
        
        Returns:
            List of GAIATask objects
        """
        tasks = []
        
        with open(tasks_file, 'r') as f:
            for line in f:
                data = json.loads(line)
                task = GAIATask(
                    task_id=data['task_id'],
                    question=data['Question'],
                    level=data['Level'],
                    final_answer=data['Final answer'],
                    file_name=data.get('file_name'),
                    file_path=data.get('file_path'),
                    annotator_metadata=data.get('Annotator Metadata')
                )
                tasks.append(task)
        
        return tasks
    
    def run_task(self, task: GAIATask) -> Dict[str, Any]:
        """
        Run single GAIA task.
        
        Args:
            task: GAIA task to run
        
        Returns:
            Task results dict
        """
        # Create logger
        logger = LRSLogger(
            agent_id=task.task_id,
            log_file=str(self.log_dir / f"{task.task_id}.jsonl")
        )
        
        # Create tools
        tools = GAIAToolkit.create_tools()
        
        # Create LRS agent
        from lrs import create_lrs_agent
        
        agent = create_lrs_agent(
            llm=self.llm,
            tools=tools,
            preferences={
                'answer_correct': 10.0,
                'step_taken': -0.1,
                'error': -2.0
            }
        )
        
        # Run agent
        start_time = time.time()
        
        state = {
            'messages': [{
                'role': 'user',
                'content': task.question
            }],
            'belief_state': {
                'task_id': task.task_id,
                'level': task.level,
                'file_path': task.file_path
            },
            'max_iterations': self.max_steps
        }
        
        try:
            result = agent.invoke(state)
            execution_time = time.time() - start_time
            
            # Extract answer
            predicted_answer = self._extract_answer(result)
            
            # Check correctness
            correct = self._check_answer(predicted_answer, task.final_answer)
            
            # Log performance
            logger.log_performance_metrics(
                total_steps=len(result.get('tool_history', [])),
                success_rate=1.0 if correct else 0.0,
                avg_precision=sum(result['precision'].values()) / len(result['precision']),
                adaptation_count=result.get('adaptation_count', 0),
                execution_time=execution_time
            )
            
            return {
                'task_id': task.task_id,
                'level': task.level,
                'correct': correct,
                'predicted_answer': predicted_answer,
                'expected_answer': task.final_answer,
                'steps': len(result.get('tool_history', [])),
                'adaptations': result.get('adaptation_count', 0),
                'precision_trajectory': result.get('precision', {}),
                'execution_time': execution_time
            }
        
        except Exception as e:
            logger.log_error('task_execution', str(e))
            return {
                'task_id': task.task_id,
                'level': task.level,
                'correct': False,
                'error': str(e)
            }
    
    def _extract_answer(self, result: Dict) -> str:
        """Extract final answer from agent output"""
        belief_state = result.get('belief_state', {})
        
        # Check for explicit answer
        if 'final_answer' in belief_state:
            return str(belief_state['final_answer'])
        
        # Check tool history for answer
        tool_history = result.get('tool_history', [])
        if tool_history:
            last_result = tool_history[-1].get('result', '')
            return str(last_result)
        
        return ""
    
    def _check_answer(self, predicted: str, expected: str) -> bool:
        """
        Check if predicted answer matches expected.
        
        Uses fuzzy matching for numerical answers and substrings.
        """
        # Normalize
        predicted = str(predicted).strip().lower()
        expected = str(expected).strip().lower()
        
        # Exact match
        if predicted == expected:
            return True
        
        # Numerical fuzzy match
        try:
            pred_num = float(predicted)
            exp_num = float(expected)
            
            # Within 1% tolerance
            if abs(pred_num - exp_num) / abs(exp_num) < 0.01:
                return True
        except:
            pass
        
        # Substring match (predicted contains expected)
        if expected in predicted:
            return True
        
        return False
    
    def run(
        self,
        tasks_file: str,
        level_filter: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Run full GAIA benchmark.
        
        Args:
            tasks_file: Path to tasks JSONL file
            level_filter: Optional level filter (1, 2, or 3)
        
        Returns:
            Aggregate results by level
        """
        # Load tasks
        tasks = self.load_tasks(tasks_file)
        
        # Filter by level
        if level_filter:
            tasks = [t for t in tasks if t.level == level_filter]
        
        print(f"Running GAIA benchmark ({len(tasks)} tasks)...")
        
        # Run tasks
        results = []
        for i, task in enumerate(tasks):
            print(f"  Task {i+1}/{len(tasks)}: {task.task_id}...")
            result = self.run_task(task)
            results.append(result)
        
        # Aggregate by level
        by_level = {1: [], 2: [], 3: []}
        for r in results:
            by_level[r['level']].append(r)
        
        # Calculate stats
        def calc_stats(task_results):
            if not task_results:
                return {'total': 0, 'correct': 0, 'success_rate': 0.0}
            
            correct = sum(1 for r in task_results if r.get('correct', False))
            return {
                'total': len(task_results),
                'correct': correct,
                'success_rate': correct / len(task_results)
            }
        
        overall_stats = calc_stats(results)
        level1_stats = calc_stats(by_level[1])
        level2_stats = calc_stats(by_level[2])
        level3_stats = calc_stats(by_level[3])
        
        # Print summary
        print("\n" + "="*60)
        print("GAIA BENCHMARK RESULTS")
        print("="*60)
        print(f"Overall: {overall_stats['correct']}/{overall_stats['total']} ({overall_stats['success_rate']:.1%})")
        print(f"Level 1: {level1_stats['correct']}/{level1_stats['total']} ({level1_stats['success_rate']:.1%})")
        print(f"Level 2: {level2_stats['correct']}/{level2_stats['total']} ({level2_stats['success_rate']:.1%})")
        print(f"Level 3: {level3_stats['correct']}/{level3_stats['total']} ({level3_stats['success_rate']:.1%})")
        print("="*60)
        
        return {
            'overall': overall_stats,
            'level_1': level1_stats,
            'level_2': level2_stats,
            'level_3': level3_stats,
            'all_results': results
        }


# Allow running as script
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python gaia_benchmark.py <tasks_file.jsonl>")
        sys.exit(1)
    
    try:
        from langchain_anthropic import ChatAnthropic
        llm = ChatAnthropic(model="claude-sonnet-4-20250514")
    except:
        print("Error: Install langchain-anthropic to run benchmark")
        sys.exit(1)
    
    benchmark = GAIABenchmark(llm)
    results = benchmark.run(sys.argv[1])
