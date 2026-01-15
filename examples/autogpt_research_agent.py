from langchain_anthropic import ChatAnthropic
from lrs.integration.autogpt_adapter import LRSAutoGPTAgent
import requests
from bs4 import BeautifulSoup
import json
from pathlib import Path


# Define AutoGPT-style commands as functions
def browse_website(url: str) -> dict:
    """
    Browse a website and extract content.
    
    AutoGPT command signature.
    """
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Extract main content
        paragraphs = soup.find_all('p')
        content = '\n\n'.join([p.get_text() for p in paragraphs[:10]])
        
        return {
            'status': 'success',
            'content': content,
            'url': url
        }
    
    except Exception as e:
        return {
            'status': 'error',
            'error': str(e)
        }


def search_web(query: str) -> dict:
    """
    Search the web (mock - would use real API).
    
    AutoGPT command signature.
    """
    # Mock search results
    results = [
        {
            'title': f'Result for {query} - Article 1',
            'url': 'https://example.com/article1',
            'snippet': 'This article discusses...'
        },
        {
            'title': f'Result for {query} - Paper 2',
            'url': 'https://example.com/paper2',
            'snippet': 'Research shows that...'
        }
    ]
    
    return {
        'status': 'success',
        'results': results,
        'query': query
    }


def write_file(filename: str, content: str) -> dict:
    """
    Write content to a file.
    
    AutoGPT command signature.
    """
    try:
        Path(filename).write_text(content)
        return {
            'status': 'success',
            'filename': filename,
            'bytes_written': len(content)
        }
    
    except Exception as e:
        return {
            'status': 'error',
            'error': str(e)
        }


def read_file(filename: str) -> dict:
    """
    Read file content.
    
    AutoGPT command signature.
    """
    try:
        content = Path(filename).read_text()
        return {
            'status': 'success',
            'content': content,
            'filename': filename
        }
    
    except Exception as e:
        return {
            'status': 'error',
            'error': str(e)
        }


def append_to_file(filename: str, content: str) -> dict:
    """
    Append content to file.
    
    AutoGPT command signature.
    """
    try:
        with open(filename, 'a') as f:
            f.write(content)
        
        return {
            'status': 'success',
            'filename': filename
        }
    
    except Exception as e:
        return {
            'status': 'error',
            'error': str(e)
        }


def main():
    print("=" * 60)
    print("AUTOGPT RESEARCH AGENT WITH LRS")
    print("=" * 60)
    print("""
This example shows how LRS enhances AutoGPT with:
- Automatic adaptation when commands fail
- Precision tracking across research steps
- Smart fallback strategies
- No manual error handling needed
    """)
    
    # Initialize LLM
    llm = ChatAnthropic(model="claude-sonnet-4-20250514")
    
    # Define agent with AutoGPT-style commands
    print("\n→ Creating LRS-powered AutoGPT agent...")
    
    agent = LRSAutoGPTAgent(
        name="ResearchAgent",
        role="AI research assistant",
        commands={
            'browse_website': browse_website,
            'search_web': search_web,
            'write_file': write_file,
            'read_file': read_file,
            'append_to_file': append_to_file
        },
        llm=llm,
        goals=[
            "Research the given topic thoroughly",
            "Synthesize findings from multiple sources",
            "Create a comprehensive report"
        ]
    )
    
    print("  ✓ Agent created with 5 commands")
    print("\nAvailable commands:")
    print("  - browse_website: Extract content from URLs")
    print("  - search_web: Find relevant sources")
    print("  - write_file: Create research reports")
    print("  - read_file: Review existing notes")
    print("  - append_to_file: Update reports")
    
    # Run research task
    print("\n" + "-" * 60)
    print("RUNNING RESEARCH TASK")
    print("-" * 60)
    
    task = "Research Active Inference and write a summary report"
    
    print(f"\nTask: {task}")
    print("\n→ Agent executing...")
    
    # Note: This is a simplified demonstration
    # Full implementation would actually execute the task
    
    print("\nExecution trace (simulated):")
    print("  1. ✓ search_web('Active Inference')")
    print("     Precision: 0.50 → 0.55 (successful search)")
    print()
    print("  2. ✗ browse_website('https://example.com/article1')")
    print("     Precision: 0.55 → 0.45 (connection timeout)")
    print("     → Adaptation triggered!")
    print()
    print("  3. ✓ browse_website('https://example.com/article2')")
    print("     Precision: 0.45 → 0.60 (fallback succeeded)")
    print()
    print("  4. ✓ write_file('active_inference_summary.txt', ...)")
    print("     Precision: 0.60 → 0.65")
    print()
    print("  5. ✓ read_file('active_inference_summary.txt')")
    print("     Precision: 0.65 → 0.70")
    
    # Simulated results
    results = {
        'success': True,
        'precision_trajectory': [0.50, 0.55, 0.45, 0.60, 0.65, 0.70],
        'adaptations': 1,
        'tool_usage': [
            {'tool': 'search_web', 'success': True},
            {'tool': 'browse_website', 'success': False},
            {'tool': 'browse_website', 'success': True},
            {'tool': 'write_file', 'success': True},
            {'tool': 'read_file', 'success': True}
        ],
        'final_state': {
            'task': task,
            'completed': True,
            'report_written': True
        }
    }
    
    # Display results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    
    print(f"\nTask completed: {results['success']}")
    print(f"Total adaptations: {results['adaptations']}")
    print(f"Final precision: {results['precision_trajectory'][-1]:.2f}")
    
    print("\nTool usage:")
    for entry in results['tool_usage']:
        status = "✓" if entry['success'] else "✗"
        print(f"  {status} {entry['tool']}")
    
    # Comparison with standard AutoGPT
    print("\n" + "=" * 60)
    print("LRS vs STANDARD AUTOGPT")
    print("=" * 60)
    
    print("""
Standard AutoGPT:
  ✗ Manual error handling needed
  ✗ Fixed retry logic
  ✗ No learning from failures
  ✗ Can loop on same failed command
  
LRS-Enhanced AutoGPT:
  ✓ Automatic adaptation on errors
  ✓ Precision-driven strategy selection
  ✓ Learns which commands are reliable
  ✓ Explores alternatives when stuck
  ✓ Graceful degradation
    """)
    
    # Show precision trajectory
    print("\n" + "=" * 60)
    print("PRECISION TRAJECTORY")
    print("=" * 60)
    
    try:
        import matplotlib.pyplot as plt
        
        steps = range(len(results['precision_trajectory']))
        precision = results['precision_trajectory']
        
        plt.figure(figsize=(10, 6))
        plt.plot(steps, precision, 'o-', linewidth=2, markersize=8)
        plt.axhline(y=0.7, color='green', linestyle='--', alpha=0.5, label='High confidence')
        plt.axhline(y=0.4, color='orange', linestyle='--', alpha=0.5, label='Adaptation threshold')
        plt.xlabel('Step')
        plt.ylabel('Precision')
        plt.title('Research Agent Precision Over Time')
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig('autogpt_precision_trajectory.png', dpi=150)
        
        print("\n✓ Visualization saved to autogpt_precision_trajectory.png")
        print("\nPrecision behavior:")
        print("  • Starts at neutral (0.5)")
        print("  • Increases with successful commands")
        print("  • Drops sharply on failures")
        print("  • Recovers as agent adapts")
    
    except ImportError:
        print("\n(Install matplotlib for visualization)")
    
    # Real-world applications
    print("\n" + "=" * 60)
    print("REAL-WORLD APPLICATIONS")
    print("=" * 60)
    print("""
LRS-enhanced AutoGPT is ideal for:

1. Research Automation
   - Literature reviews
   - Data collection
   - Report generation
   
2. Content Creation
   - Blog post research
   - Fact-checking
   - Citation gathering
   
3. Data Pipeline Orchestration
   - ETL workflows
   - API integration
   - Error-resilient processing
   
4. Competitive Intelligence
   - Market research
   - Competitor analysis
   - Trend monitoring
   
Key Advantage:
  AutoGPT provides the task decomposition
  LRS provides the resilient execution
    """)


if __name__ == "__main__":
    print("\n[Note: This is a demonstration with simulated execution]")
    print("[Full implementation would execute actual AutoGPT tasks]")
    print("[Commands shown are illustrative of the pattern]\n")
    
    main()
