AutoGPT Integration
===================

Integrate LRS-Agents with AutoGPT for resilient autonomous agents with automatic adaptation.

Overview
--------

AutoGPT provides:

* Task decomposition
* Autonomous execution loops
* Command-based architecture

LRS-Agents adds:

* Automatic adaptation when commands fail
* Precision tracking across steps
* Smart fallback strategies
* No manual error handling

Quick Start
-----------

Basic AutoGPT Agent
^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from langchain_anthropic import ChatAnthropic
   from lrs.integration.autogpt_adapter import LRSAutoGPTAgent

   # Define AutoGPT-style commands
   def search_web(query: str) -> dict:
       """Search the web for information"""
       # Implementation
       return {'results': [...]}

   def write_file(filename: str, content: str) -> dict:
       """Write content to a file"""
       with open(filename, 'w') as f:
           f.write(content)
       return {'status': 'success', 'filename': filename}

   # Create LRS-powered AutoGPT agent
   llm = ChatAnthropic(model="claude-sonnet-4-20250514")
   
   agent = LRSAutoGPTAgent(
       name="ResearchAgent",
       role="AI research assistant",
       commands={
           'search_web': search_web,
           'write_file': write_file
       },
       llm=llm,
       goals=[
           "Research the given topic",
           "Synthesize findings",
           "Create a report"
       ]
   )

   # Run task
   result = agent.run(
       task="Research Active Inference and create a summary",
       max_iterations=50
   )

Converting AutoGPT Commands to LRS Tools
-----------------------------------------

Automatic Conversion
^^^^^^^^^^^^^^^^^^^^

AutoGPT commands are automatically converted to LRS tools:

.. code-block:: python

   from lrs.integration.autogpt_adapter import AutoGPTCommand

   # AutoGPT command function
   def browse_website(url: str) -> dict:
       """Browse a website and extract content"""
       response = requests.get(url)
       soup = BeautifulSoup(response.text, 'html.parser')
       return {
           'status': 'success',
           'content': soup.get_text()[:1000]
       }

   # Automatically wrapped as LRS tool
   browse_tool = AutoGPTCommand(
       name="browse_website",
       func=browse_website,
       description="Browse a website"
   )

   # Now has:
   # - Automatic error handling
   # - Prediction error calculation
   # - Statistics tracking

Custom Prediction Errors
^^^^^^^^^^^^^^^^^^^^^^^^^

For better adaptation, provide custom error calculation:

.. code-block:: python

   def api_command(endpoint: str) -> dict:
       """Call external API"""
       try:
           response = requests.get(f"https://api.example.com/{endpoint}")
           response.raise_for_status()
           return {'status': 'success', 'data': response.json()}
       except requests.exceptions.Timeout:
           return {'status': 'error', 'error': 'timeout'}
       except requests.exceptions.HTTPError as e:
           return {'status': 'error', 'error': str(e)}

   def calculate_error(result: dict) -> float:
       """Custom prediction error calculation"""
       if result['status'] == 'success':
           return 0.1  # Expected success
       elif 'timeout' in result.get('error', ''):
           return 0.7  # Somewhat expected
       else:
           return 0.9  # Unexpected error

   api_tool = AutoGPTCommand(
       name="api_call",
       func=api_command,
       description="Call external API",
       error_fn=calculate_error
   )

Common AutoGPT Commands
-----------------------

File Operations
^^^^^^^^^^^^^^^

.. code-block:: python

   def read_file(filename: str) -> dict:
       """Read file contents"""
       try:
           with open(filename, 'r') as f:
               content = f.read()
           return {'status': 'success', 'content': content}
       except FileNotFoundError:
           return {'status': 'error', 'error': 'File not found'}
       except Exception as e:
           return {'status': 'error', 'error': str(e)}

   def write_file(filename: str, content: str) -> dict:
       """Write content to file"""
       try:
           with open(filename, 'w') as f:
               f.write(content)
           return {'status': 'success', 'filename': filename}
       except Exception as e:
           return {'status': 'error', 'error': str(e)}

   def append_to_file(filename: str, content: str) -> dict:
       """Append content to file"""
       try:
           with open(filename, 'a') as f:
               f.write(content)
           return {'status': 'success'}
       except Exception as e:
           return {'status': 'error', 'error': str(e)}

Web Operations
^^^^^^^^^^^^^^

.. code-block:: python

   def search_web(query: str) -> dict:
       """Search the web"""
       try:
           # Using DuckDuckGo as example
           from duckduckgo_search import DDGS
           
           with DDGS() as ddgs:
               results = list(ddgs.text(query, max_results=5))
           
           return {
               'status': 'success',
               'results': results
           }
       except Exception as e:
           return {'status': 'error', 'error': str(e)}

   def browse_website(url: str) -> dict:
       """Browse a website"""
       try:
           import requests
           from bs4 import BeautifulSoup
           
           response = requests.get(url, timeout=10)
           soup = BeautifulSoup(response.text, 'html.parser')
           
           # Extract main content
           paragraphs = soup.find_all('p')
           content = '\n\n'.join([p.get_text() for p in paragraphs[:10]])
           
           return {
               'status': 'success',
               'url': url,
               'content': content
           }
       except Exception as e:
           return {'status': 'error', 'error': str(e)}

Code Execution
^^^^^^^^^^^^^^

.. code-block:: python

   def execute_python(code: str) -> dict:
       """Execute Python code safely"""
       try:
           # Create restricted namespace
           namespace = {
               '__builtins__': {
                   'print': print,
                   'len': len,
                   'range': range,
                   'sum': sum,
                   # Add safe builtins only
               }
           }
           
           # Capture output
           from io import StringIO
           import sys
           
           old_stdout = sys.stdout
           sys.stdout = StringIO()
           
           exec(code, namespace)
           
           output = sys.stdout.getvalue()
           sys.stdout = old_stdout
           
           return {
               'status': 'success',
               'output': output
           }
       except Exception as e:
           return {'status': 'error', 'error': str(e)}

Complete Research Agent Example
--------------------------------

Here's a complete autonomous research agent:

.. code-block:: python

   from langchain_anthropic import ChatAnthropic
   from lrs.integration.autogpt_adapter import LRSAutoGPTAgent
   from lrs.monitoring.tracker import LRSStateTracker
   from lrs.monitoring.structured_logging import create_logger_for_agent
   import requests
   from bs4 import BeautifulSoup
   from duckduckgo_search import DDGS

   # Initialize monitoring
   tracker = LRSStateTracker()
   logger = create_logger_for_agent("research_agent")

   # Define commands
   def search_web(query: str) -> dict:
       """Search the web for information"""
       try:
           with DDGS() as ddgs:
               results = list(ddgs.text(query, max_results=5))
           
           logger.log_tool_execution(
               tool_name="search_web",
               success=True,
               execution_time=0.5,
               prediction_error=0.1
           )
           
           return {'status': 'success', 'results': results}
       except Exception as e:
           logger.log_tool_execution(
               tool_name="search_web",
               success=False,
               execution_time=0.5,
               prediction_error=0.9,
               error_message=str(e)
           )
           return {'status': 'error', 'error': str(e)}

   def browse_website(url: str) -> dict:
       """Browse and extract website content"""
       try:
           response = requests.get(url, timeout=10)
           soup = BeautifulSoup(response.text, 'html.parser')
           
           # Extract content
           paragraphs = soup.find_all('p')
           content = '\n\n'.join([p.get_text() for p in paragraphs[:15]])
           
           return {
               'status': 'success',
               'url': url,
               'content': content
           }
       except requests.exceptions.Timeout:
           return {'status': 'error', 'error': 'Connection timeout'}
       except Exception as e:
           return {'status': 'error', 'error': str(e)}

   def write_report(filename: str, content: str) -> dict:
       """Write research report to file"""
       try:
           with open(filename, 'w') as f:
               f.write(content)
           return {'status': 'success', 'filename': filename}
       except Exception as e:
           return {'status': 'error', 'error': str(e)}

   def read_notes(filename: str = "research_notes.txt") -> dict:
       """Read research notes"""
       try:
           with open(filename, 'r') as f:
               content = f.read()
           return {'status': 'success', 'content': content}
       except FileNotFoundError:
           return {'status': 'success', 'content': ''}  # Empty notes
       except Exception as e:
           return {'status': 'error', 'error': str(e)}

   def append_notes(content: str, filename: str = "research_notes.txt") -> dict:
       """Append to research notes"""
       try:
           with open(filename, 'a') as f:
               f.write(content + '\n\n')
           return {'status': 'success'}
       except Exception as e:
           return {'status': 'error', 'error': str(e)}

   # Create agent
   llm = ChatAnthropic(model="claude-sonnet-4-20250514")

   agent = LRSAutoGPTAgent(
       name="ResearchAgent",
       role="Autonomous research assistant",
       commands={
           'search_web': search_web,
           'browse_website': browse_website,
           'write_report': write_report,
           'read_notes': read_notes,
           'append_notes': append_notes
       },
       llm=llm,
       goals=[
           "Research the given topic thoroughly",
           "Collect information from multiple sources",
           "Synthesize findings into a coherent report",
           "Ensure accuracy and cite sources"
       ],
       tracker=tracker
   )

   # Run research task
   result = agent.run(
       task="Research the latest developments in Active Inference and its applications to AI agents",
       max_iterations=30
   )

   # Analyze performance
   print("=" * 60)
   print("RESEARCH RESULTS")
   print("=" * 60)
   
   print(f"\nTotal steps: {len(result['tool_history'])}")
   print(f"Adaptations: {result.get('adaptation_count', 0)}")
   print(f"Final precision: {result['precision']['execution']:.2f}")
   
   print("\nCommand usage:")
   for cmd, stats in tracker.get_tool_usage_stats().items():
       print(f"  {cmd}: {stats['calls']} calls, {stats['success_rate']:.1%} success")
   
   print("\nAdaptation events:")
   for event in tracker.get_adaptation_events():
       print(f"  Step {event['step']}: {event['trigger']}")
       print(f"    Precision: {event['old_precision']:.2f} → {event['new_precision']:.2f}")

Handling AutoGPT Loops
-----------------------

Traditional AutoGPT Loop
^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   # Traditional AutoGPT (simplified)
   while not task_complete:
       # Get next action from LLM
       action = llm.decide_next_action(task, memory)
       
       # Execute command
       result = execute_command(action)
       
       # Update memory
       memory.append(result)
       
       # Check if done
       task_complete = check_completion(task, memory)

Problems:

* No learning from failures
* Fixed retry logic
* Can loop on same failed command
* No automatic adaptation

LRS-Enhanced Loop
^^^^^^^^^^^^^^^^^

.. code-block:: python

   # LRS-enhanced AutoGPT
   while not task_complete and precision > 0.2:
       # Generate policy proposals (with precision-adaptive temperature)
       proposals = generate_proposals(task, memory, precision)
       
       # Evaluate via Expected Free Energy
       G_values = [calculate_G(p, precision) for p in proposals]
       
       # Select policy (precision-weighted)
       policy = select_policy(proposals, G_values, precision)
       
       # Execute command
       result = execute_command(policy)
       
       # Update precision based on prediction error
       precision = update_precision(precision, result.prediction_error)
       
       # Adapt if precision drops
       if precision < threshold:
           explore_alternatives()
       
       # Update memory
       memory.append(result)

Benefits:

* Learns from failures (precision tracking)
* Adapts automatically (no fixed retry logic)
* Explores alternatives when stuck
* Balances explore/exploit naturally

Precision Dynamics in Research
-------------------------------

Example Execution Trace
^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: text

   Step 1: search_web("Active Inference")
           ✓ Success (error: 0.1)
           Precision: 0.50 → 0.52
   
   Step 2: browse_website(result[0].url)
           ✗ Timeout (error: 0.9)
           Precision: 0.52 → 0.42
   
   Step 3: browse_website(result[1].url)
           ✗ Timeout (error: 0.9)
           Precision: 0.42 → 0.32
   
   [ADAPTATION TRIGGERED - Precision < 0.4]
   
   Step 4: append_notes("Search results available")
           ✓ Success (error: 0.05)
           Precision: 0.32 → 0.37
           
           Strategy shift: Bypass browsing, use search results directly
   
   Step 5: search_web("Active Inference applications")
           ✓ Success (error: 0.1)
           Precision: 0.37 → 0.42
   
   Step 6: write_report("report.txt", content)
           ✓ Success (error: 0.05)
           Precision: 0.42 → 0.48

Key Insight: Agent automatically shifted strategy when browsing failed repeatedly.

Monitoring and Logging
----------------------

Real-time Monitoring
^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from lrs.monitoring.tracker import LRSStateTracker
   from lrs.monitoring.dashboard import create_dashboard
   import threading

   tracker = LRSStateTracker()

   # Run dashboard in background
   dashboard_thread = threading.Thread(
       target=create_dashboard,
       args=(tracker,),
       daemon=True
   )
   dashboard_thread.start()

   # Create agent with tracker
   agent = LRSAutoGPTAgent(
       name="Agent",
       commands=commands,
       llm=llm,
       tracker=tracker
   )

   # Dashboard available at http://localhost:8501

Structured Logging
^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from lrs.monitoring.structured_logging import create_logger_for_agent

   logger = create_logger_for_agent(
       agent_id="research_agent_1",
       log_file="logs/agent.jsonl"
   )

   # Logs automatically include:
   # - Timestamp
   # - Agent ID
   # - Command executed
   # - Success/failure
   # - Prediction error
   # - Precision value
   # - Adaptation events

   # Query logs later
   import json

   with open("logs/agent.jsonl") as f:
       logs = [json.loads(line) for line in f]
   
   # Analyze failures
   failures = [l for l in logs if l['event'] == 'tool_execution' and not l['success']]
   print(f"Total failures: {len(failures)}")

Best Practices
--------------

1. Define Clear Goals
^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   agent = LRSAutoGPTAgent(
       name="Agent",
       goals=[
           "Specific goal 1 with clear success criteria",
           "Specific goal 2 that's measurable",
           "Specific goal 3 with stopping condition"
       ]
   )

   # Bad goals (too vague):
   # - "Do research"
   # - "Be helpful"
   # - "Complete the task"

2. Provide Diverse Commands
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   commands = {
       # Fast, reliable commands
       'read_cache': read_from_cache,
       'check_notes': read_notes,
       
       # Medium speed, medium reliability
       'search_web': search_web,
       'execute_code': execute_python,
       
       # Slow, potentially unreliable
       'browse_website': browse_website,
       'download_file': download_file
   }

   # Agent can adapt based on reliability

3. Set Reasonable Iteration Limits
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   # Simple tasks
   result = agent.run(task="Quick lookup", max_iterations=10)

   # Complex research
   result = agent.run(task="Deep research", max_iterations=50)

   # Very complex tasks
   result = agent.run(task="Multi-day analysis", max_iterations=200)

4. Monitor Command Performance
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   # After execution
   stats = tracker.get_tool_usage_stats()
   
   for cmd, cmd_stats in stats.items():
       if cmd_stats['success_rate'] < 0.5:
           print(f"Warning: {cmd} has low success rate: {cmd_stats['success_rate']:.1%}")
           print(f"  Consider improving or providing alternatives")

Troubleshooting
---------------

Agent Gets Stuck
^^^^^^^^^^^^^^^^

If agent loops on same command:

.. code-block:: python

   # Check prediction errors are set correctly
   def problematic_command(arg: str) -> dict:
       try:
           result = operation()
           return {'status': 'success', 'result': result}
       except Exception as e:
           # IMPORTANT: Return high prediction error for failures
           return {'status': 'error', 'error': str(e), '_prediction_error': 0.9}

   # Without high prediction error, precision won't drop
   # Without precision drop, no adaptation triggered

Agent Doesn't Adapt
^^^^^^^^^^^^^^^^^^^

Check adaptation threshold:

.. code-block:: python

   agent = LRSAutoGPTAgent(
       commands=commands,
       llm=llm,
       adaptation_threshold=0.4  # Default
   )

   # Lower threshold = more adaptation
   agent = LRSAutoGPTAgent(
       commands=commands,
       llm=llm,
       adaptation_threshold=0.3  # More sensitive
   )

High Iteration Count
^^^^^^^^^^^^^^^^^^^^

If agent takes too many steps:

.. code-block:: python

   # Increase step cost
   agent = LRSAutoGPTAgent(
       commands=commands,
       llm=llm,
       preferences={
           'success': 5.0,
           'error': -3.0,
           'step_cost': -0.5  # Higher cost per step
       }
   )

   # Or provide more efficient commands
   commands['quick_summary'] = quick_summary_command

Comparison with Standard AutoGPT
---------------------------------

.. list-table::
   :header-rows: 1
   :widths: 30 35 35

   * - Feature
     - Standard AutoGPT
     - LRS AutoGPT
   * - Error Handling
     - Manual try/except
     - Automatic via precision
   * - Retry Logic
     - Fixed (e.g., 3 retries)
     - Adaptive based on confidence
   * - Alternative Commands
     - Must be explicitly programmed
     - Explored automatically
   * - Learning
     - No learning from failures
     - Updates precision continuously
   * - Stuck Detection
     - Manual timeout/iteration limit
     - Precision collapse triggers adaptation
   * - Exploration
     - Random or none
     - Principled via free energy

Next Steps
----------

* Try the research agent example
* Read :doc:`../tutorials/05_llm_integration`
* See :doc:`production_deployment` for scaling
* Explore :doc:`../api/integration` for API details

