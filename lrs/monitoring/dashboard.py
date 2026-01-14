"""
Real-time Streamlit dashboard for LRS agents.

Provides visualization of:
- Precision trajectories (3-level hierarchy)
- G-space map (epistemic vs pragmatic)
- Prediction error stream
- Adaptation timeline
- Tool usage statistics
"""

import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Any
from datetime import datetime

from lrs.monitoring.tracker import LRSStateTracker


def create_dashboard(tracker: LRSStateTracker):
    """
    Create Streamlit dashboard for LRS agent monitoring.
    
    Args:
        tracker: LRSStateTracker instance with execution history
    
    Examples:
        >>> import streamlit as st
        >>> from lrs.monitoring import create_dashboard
        >>> 
        >>> tracker = LRSStateTracker()
        >>> # ... run agent with tracker ...
        >>> 
        >>> create_dashboard(tracker)
    """
    st.set_page_config(
        page_title="LRS Agent Monitor",
        page_icon="🧠",
        layout="wide"
    )
    
    st.title("🧠 LRS Agent Monitoring Dashboard")
    st.markdown("Real-time Active Inference agent observability")
    
    # Sidebar with summary stats
    _render_sidebar(tracker)
    
    # Main content
    col1, col2 = st.columns(2)
    
    with col1:
        _render_precision_trajectories(tracker)
        _render_prediction_error_stream(tracker)
    
    with col2:
        _render_g_space_map(tracker)
        _render_tool_usage(tracker)
    
    # Full-width sections
    _render_adaptation_timeline(tracker)
    _render_detailed_history(tracker)


def _render_sidebar(tracker: LRSStateTracker):
    """Render sidebar with summary statistics"""
    st.sidebar.header("📊 Summary Statistics")
    
    summary = tracker.get_summary()
    
    st.sidebar.metric("Total Steps", summary['total_steps'])
    st.sidebar.metric("Adaptations", summary['total_adaptations'])
    st.sidebar.metric("Avg Precision", f"{summary['avg_precision']:.3f}")
    
    if summary['final_precision']:
        st.sidebar.subheader("Current Precision")
        for level, value in summary['final_precision'].items():
            st.sidebar.metric(
                level.capitalize(),
                f"{value:.3f}",
                delta=None
            )
    
    # Export button
    if st.sidebar.button("Export History"):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = f"lrs_history_{timestamp}.json"
        tracker.export_history(filepath)
        st.sidebar.success(f"Exported to {filepath}")


def _render_precision_trajectories(tracker: LRSStateTracker):
    """Render precision trajectory chart"""
    st.subheader("📈 Precision Trajectories")
    
    trajectories = tracker.get_all_precision_trajectories()
    
    if not trajectories or not trajectories['execution']:
        st.info("No data yet. Run agent to see precision trajectories.")
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    steps = range(len(trajectories['execution']))
    
    ax.plot(steps, trajectories['abstract'], 
            label='Abstract', linewidth=2, alpha=0.8, color='blue')
    ax.plot(steps, trajectories['planning'], 
            label='Planning', linewidth=2, alpha=0.8, color='orange')
    ax.plot(steps, trajectories['execution'], 
            label='Execution', linewidth=2, alpha=0.8, color='green')
    
    # Threshold lines
    ax.axhline(y=0.7, color='green', linestyle='--', alpha=0.3, label='High confidence')
    ax.axhline(y=0.4, color='orange', linestyle='--', alpha=0.3, label='Adaptation threshold')
    
    ax.set_xlabel('Step')
    ax.set_ylabel('Precision (γ)')
    ax.set_title('Hierarchical Precision Over Time')
    ax.legend(loc='best')
    ax.grid(alpha=0.3)
    ax.set_ylim([0, 1])
    
    st.pyplot(fig)
    plt.close()
    
    # Current values
    current = tracker.get_current_state()
    if current:
        cols = st.columns(3)
        for i, (level, value) in enumerate(current.precision.items()):
            with cols[i]:
                st.metric(
                    level.capitalize(),
                    f"{value:.3f}",
                    delta=None
                )


def _render_g_space_map(tracker: LRSStateTracker):
    """Render G-space visualization"""
    st.subheader("🎯 G-Space Map")
    
    # This requires G values from candidate policies
    # For now, show a placeholder
    st.info("G-space map shows epistemic vs pragmatic values for candidate policies.")
    st.markdown("""
    **Coming soon**: Scatter plot of:
    - X-axis: Epistemic value (information gain)
    - Y-axis: Pragmatic value (expected reward)
    - Points: Candidate policies
    - Highlight: Selected policy
    """)


def _render_prediction_error_stream(tracker: LRSStateTracker):
    """Render prediction error timeline"""
    st.subheader("⚠️ Prediction Error Stream")
    
    errors = tracker.get_prediction_errors()
    
    if not errors:
        st.info("No prediction errors recorded yet.")
        return
    
    fig, ax = plt.subplots(figsize=(10, 4))
    
    ax.bar(range(len(errors)), errors, color='red', alpha=0.6)
    ax.axhline(y=0.7, color='orange', linestyle='--', alpha=0.5, label='High surprise')
    ax.set_xlabel('Execution Step')
    ax.set_ylabel('Prediction Error (ε)')
    ax.set_title('Surprise Events Over Time')
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_ylim([0, 1])
    
    st.pyplot(fig)
    plt.close()
    
    # Statistics
    avg_error = sum(errors) / len(errors)
    high_errors = [e for e in errors if e > 0.7]
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Avg Error", f"{avg_error:.3f}")
    col2.metric("High Surprise Events", len(high_errors))
    col3.metric("Max Error", f"{max(errors):.3f}")


def _render_tool_usage(tracker: LRSStateTracker):
    """Render tool usage statistics"""
    st.subheader("🔧 Tool Usage Statistics")
    
    stats = tracker.get_tool_usage_stats()
    
    if not stats:
        st.info("No tool executions yet.")
        return
    
    # Create dataframe
    df = pd.DataFrame([
        {
            'Tool': tool_name,
            'Calls': data['calls'],
            'Success Rate': data['success_rate'],
            'Avg Error': data['avg_error']
        }
        for tool_name, data in stats.items()
    ])
    
    # Display table
    st.dataframe(
        df.style.format({
            'Success Rate': '{:.1%}',
            'Avg Error': '{:.3f}'
        }),
        use_container_width=True
    )
    
    # Visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    
    # Success rates
    ax1.barh(df['Tool'], df['Success Rate'], color='green', alpha=0.7)
    ax1.set_xlabel('Success Rate')
    ax1.set_title('Tool Reliability')
    ax1.set_xlim([0, 1])
    
    # Call counts
    ax2.barh(df['Tool'], df['Calls'], color='blue', alpha=0.7)
    ax2.set_xlabel('Number of Calls')
    ax2.set_title('Tool Usage Frequency')
    
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()


def _render_adaptation_timeline(tracker: LRSStateTracker):
    """Render adaptation events timeline"""
    st.subheader("🔄 Adaptation Timeline")
    
    events = tracker.get_adaptation_events()
    
    if not events:
        st.info("No adaptations occurred yet.")
        return
    
    for i, event in enumerate(events, 1):
        with st.expander(f"Adaptation #{i} - {event.get('timestamp', 'Unknown time')}"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Trigger**")
                st.write(f"Tool: `{event.get('trigger_tool', 'Unknown')}`")
                st.write(f"Error: {event.get('trigger_error', 0.0):.3f}")
            
            with col2:
                st.markdown("**Precision Change**")
                before = event.get('precision_before', {})
                after = event.get('precision_after', {})
                
                for level in ['execution', 'planning', 'abstract']:
                    b = before.get(level, 0.5)
                    a = after.get(level, 0.5)
                    delta = a - b
                    st.write(f"{level}: {b:.3f} → {a:.3f} ({delta:+.3f})")


def _render_detailed_history(tracker: LRSStateTracker):
    """Render detailed execution history"""
    st.subheader("📜 Execution History")
    
    if not tracker.history:
        st.info("No execution history yet.")
        return
    
    # Create detailed log
    history_data = []
    
    for snapshot in tracker.history:
        for entry in snapshot.tool_history:
            history_data.append({
                'Timestamp': snapshot.timestamp.strftime("%H:%M:%S"),
                'Tool': entry.get('tool', 'Unknown'),
                'Success': '✓' if entry.get('success') else '✗',
                'Error': f"{entry.get('prediction_error', 0.0):.3f}",
                'Precision': f"{snapshot.precision.get('execution', 0.5):.3f}"
            })
    
    if history_data:
        df = pd.DataFrame(history_data)
        st.dataframe(df, use_container_width=True)


def run_dashboard(tracker: Optional[LRSStateTracker] = None):
    """
    Run dashboard as standalone Streamlit app.
    
    Args:
        tracker: Optional pre-populated tracker
    
    Examples:
        >>> # In terminal:
        >>> # streamlit run lrs/monitoring/dashboard.py
        >>> 
        >>> # Or programmatically:
        >>> from lrs.monitoring import run_dashboard
        >>> run_dashboard()
    """
    if tracker is None:
        # Create demo tracker with sample data
        tracker = _create_demo_tracker()
    
    create_dashboard(tracker)


def _create_demo_tracker() -> LRSStateTracker:
    """Create demo tracker with sample data for testing"""
    tracker = LRSStateTracker()
    
    # Simulate some execution history
    import random
    
    for i in range(20):
        # Simulate precision changing
        precision = {
            'execution': max(0.2, min(0.9, 0.5 + random.gauss(0, 0.1))),
            'planning': max(0.3, min(0.8, 0.5 + random.gauss(0, 0.08))),
            'abstract': max(0.4, min(0.7, 0.5 + random.gauss(0, 0.05)))
        }
        
        # Simulate tool execution
        tool_name = random.choice(['api_fetch', 'cache_fetch', 'parse_json'])
        success = random.random() > 0.3
        pred_error = random.random() * (0.3 if success else 1.0)
        
        state = {
            'precision': precision,
            'tool_history': [{
                'tool': tool_name,
                'success': success,
                'prediction_error': pred_error
            }],
            'adaptation_count': i // 5,  # Adapt every 5 steps
            'belief_state': {}
        }
        
        tracker.track_state(state)
    
    return tracker


# Allow running as standalone app
if __name__ == "__main__":
    run_dashboard()
