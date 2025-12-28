"""
Data Lineage Dashboard for RAG System

Streamlit UI for visualizing data flow, pipeline statistics, and asset lineage.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import json
from datetime import datetime, timedelta
from typing import Dict, Any, List
from data_lineage import get_lineage_tracker

# Page config
st.set_page_config(
    page_title="Data Lineage Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("📊 RAG Pipeline Data Lineage")
st.markdown("""
Monitor your document ingestion, embedding, retrieval, and LLM response generation pipeline.
Built on OpenLineage standard for data governance.
""")

tracker = get_lineage_tracker()

# Sidebar filters
st.sidebar.header("Filters")
view_mode = st.sidebar.radio("View", ["Overview", "Events Timeline", "Asset Lineage", "Performance Analysis"])
events_limit = st.sidebar.slider("Events to display", 10, 500, 100)

# Get data
stats = tracker.get_stats()
events = tracker.get_events(limit=events_limit)


# ============================================================================
# VIEW 1: OVERVIEW
# ============================================================================
if view_mode == "Overview":
    st.header("Pipeline Overview")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Events", stats["total_events"])
    
    with col2:
        st.metric("Total Assets", stats["total_assets"])
    
    # Count successes
    success_count = sum(op["count"] for op in stats["status_breakdown"] 
                       if op.get("status") == "SUCCESS")
    with col3:
        st.metric("Successful Operations", success_count)
    
    # Calculate success rate
    total_ops = sum(op["count"] for op in stats["status_breakdown"])
    success_rate = (success_count / total_ops * 100) if total_ops > 0 else 0
    with col4:
        st.metric("Success Rate", f"{success_rate:.1f}%")
    
    # Operations breakdown
    st.subheader("Operations")
    if stats["operations"]:
        ops_df = pd.DataFrame(stats["operations"])
        ops_df.columns = ["Operation", "Count", "Avg Duration (ms)"]
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.bar(ops_df, x="Operation", y="Count", 
                        title="Operation Counts",
                        color="Count", color_continuous_scale="blues")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.bar(ops_df, x="Operation", y="Avg Duration (ms)",
                        title="Average Operation Duration",
                        color="Avg Duration (ms)", color_continuous_scale="reds")
            st.plotly_chart(fig, use_container_width=True)
        
        st.dataframe(ops_df, use_container_width=True)
    else:
        st.info("No operations recorded yet. Start processing documents to see data lineage.")
    
    # Status breakdown
    st.subheader("Status Breakdown")
    if stats["status_breakdown"]:
        status_df = pd.DataFrame(stats["status_breakdown"])
        status_df.columns = ["Status", "Count"]
        
        fig = px.pie(status_df, names="Status", values="Count",
                    title="Operation Status Distribution")
        st.plotly_chart(fig, use_container_width=True)


# ============================================================================
# VIEW 2: EVENTS TIMELINE
# ============================================================================
elif view_mode == "Events Timeline":
    st.header("Events Timeline")
    
    # Filter by operation
    if events:
        operations = sorted(set(e["operation"] for e in events if e["operation"]))
        selected_op = st.selectbox("Filter by operation", ["All"] + operations)
        
        # Process events for display
        display_events = []
        for event in events:
            try:
                display_events.append({
                    "Timestamp": event["timestamp"],
                    "Operation": event["operation"],
                    "Status": event["status"],
                    "Duration (ms)": event["duration_ms"],
                    "Producer": event["producer"],
                    "Event ID": event["event_id"][:8] + "..."
                })
            except Exception as e:
                st.warning(f"Error processing event: {e}")
        
        if selected_op != "All":
            display_events = [e for e in display_events if e["Operation"] == selected_op]
        
        # Display timeline
        if display_events:
            events_df = pd.DataFrame(display_events)
            st.dataframe(events_df, use_container_width=True)
            
            # Timeline visualization
            st.subheader("Timeline Visualization")
            fig = px.timeline(
                events_df.head(50),
                x_start="Timestamp",
                x_end="Timestamp",
                y="Operation",
                color="Status",
                hover_data=["Duration (ms)"],
                title="Recent Events Timeline"
            )
            fig.update_yaxes(autorange="reversed")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No events found with selected filters.")
    else:
        st.info("No events recorded yet.")


# ============================================================================
# VIEW 3: ASSET LINEAGE
# ============================================================================
elif view_mode == "Asset Lineage":
    st.header("Asset Lineage Explorer")
    
    if events:
        # Get unique asset names
        assets = set()
        for event in events:
            try:
                input_assets = json.loads(event["input_assets"])
                output_assets = json.loads(event["output_assets"])
                for a in input_assets + output_assets:
                    assets.add(a["name"])
            except:
                pass
        
        if assets:
            selected_asset = st.selectbox("Select asset to trace", sorted(assets))
            
            if selected_asset:
                lineage = tracker.get_asset_lineage(selected_asset)
                
                if "error" not in lineage:
                    asset_info = lineage["asset"]
                    
                    # Asset details
                    st.subheader(f"Asset: {selected_asset}")
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Type", asset_info.get("type", "N/A"))
                    with col2:
                        st.metric("Format", asset_info.get("format", "N/A"))
                    with col3:
                        size = asset_info.get("size_bytes")
                        if size:
                            st.metric("Size", f"{size:,} bytes")
                        else:
                            st.metric("Size", "N/A")
                    with col4:
                        st.metric("Events", lineage["event_count"])
                    
                    # Timeline of events involving this asset
                    st.subheader("Events Involving This Asset")
                    if lineage["events"]:
                        event_list = []
                        for event in lineage["events"]:
                            event_list.append({
                                "Timestamp": event["timestamp"],
                                "Operation": event["operation"],
                                "Producer": event["producer"],
                                "Status": event["status"],
                                "Duration (ms)": event["duration_ms"]
                            })
                        
                        events_df = pd.DataFrame(event_list)
                        st.dataframe(events_df, use_container_width=True)
                    else:
                        st.info("No events found for this asset.")
                else:
                    st.error(lineage["error"])
        else:
            st.info("No assets recorded yet.")
    else:
        st.info("No data available yet.")


# ============================================================================
# VIEW 4: PERFORMANCE ANALYSIS
# ============================================================================
elif view_mode == "Performance Analysis":
    st.header("Performance Analysis")
    
    if events:
        # Parse events into dataframe
        perf_data = []
        for event in events:
            perf_data.append({
                "Operation": event["operation"],
                "Duration (ms)": event["duration_ms"],
                "Timestamp": event["timestamp"],
                "Status": event["status"]
            })
        
        if perf_data:
            perf_df = pd.DataFrame(perf_data)
            
            # Box plot of operation durations
            st.subheader("Operation Duration Distribution")
            fig = px.box(perf_df, x="Operation", y="Duration (ms)",
                        title="Duration Distribution by Operation",
                        color="Operation")
            st.plotly_chart(fig, use_container_width=True)
            
            # Scatter plot over time
            st.subheader("Duration Trends Over Time")
            fig = px.scatter(perf_df, x="Timestamp", y="Duration (ms)",
                            color="Operation", size="Duration (ms)",
                            title="Operation Duration Over Time",
                            hover_data=["Status"])
            st.plotly_chart(fig, use_container_width=True)
            
            # Statistics by operation
            st.subheader("Operation Statistics")
            stats_by_op = perf_df.groupby("Operation")["Duration (ms)"].agg([
                ("Count", "count"),
                ("Mean (ms)", "mean"),
                ("Median (ms)", "median"),
                ("Min (ms)", "min"),
                ("Max (ms)", "max"),
                ("Std Dev", "std")
            ]).round(2)
            st.dataframe(stats_by_op)
            
            # Slowest operations
            st.subheader("Slowest Operations")
            slowest = perf_df.nlargest(10, "Duration (ms)")[
                ["Operation", "Duration (ms)", "Timestamp", "Status"]
            ]
            st.dataframe(slowest, use_container_width=True)
    else:
        st.info("No performance data available yet.")


# ============================================================================
# FOOTER
# ============================================================================
st.divider()
st.markdown("""
---
**Data Lineage Dashboard** | OpenLineage Standard
- Track document ingestion, embedding, retrieval, and response generation
- Monitor pipeline health and performance metrics
- Analyze data asset lineage and dependencies
""")
