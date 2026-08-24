"""
Interactive Streamlit Dashboard for Kumbh Mela Crowd Detection System.

Features:
- Interactive Plotly charts (zoom, hover, pan)
- Real-time risk monitoring
- Frame-by-frame analysis with color coding
- Download capabilities
"""

import streamlit as st  # Streamlit web framework
import yaml  # YAML parsing
import json  # JSON handling
import tempfile  # Temporary files
import sys  # System functions
from pathlib import Path  # File paths

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np  # Numerical operations
import pandas as pd  # Data manipulation
import cv2  # OpenCV
import plotly.express as px  # Interactive charts
import plotly.graph_objects as go  # Advanced charts
from plotly.subplots import make_subplots  # Subplots


def load_config():
    """Load configuration from config.yaml."""
    config_path = Path(__file__).parent.parent / "config.yaml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def inject_custom_css():
    """Inject custom CSS for styled elements."""
    st.markdown("""
    <style>
    .risk-card {
        padding: 1.2rem;
        border-radius: 12px;
        margin-bottom: 0.5rem;
        font-weight: 600;
        text-align: center;
        font-size: 1.3rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .risk-safe { background: linear-gradient(135deg, #d4edda, #c3e6cb); color: #155724; border: 2px solid #28a745; }
    .risk-low { background: linear-gradient(135deg, #fff3cd, #ffeeba); color: #856404; border: 2px solid #ffc107; }
    .risk-moderate { background: linear-gradient(135deg, #ffe0b2, #ffcc80); color: #e65100; border: 2px solid #ff9800; }
    .risk-high { background: linear-gradient(135deg, #ffcdd2, #ef9a9a); color: #b71c1c; border: 2px solid #f44336; }
    .risk-critical { background: linear-gradient(135deg, #f44336, #e53935); color: white; border: 2px solid #b71c1c; animation: pulse 1s infinite; }
    @keyframes pulse {
        0% { opacity: 1; transform: scale(1); }
        50% { opacity: 0.8; transform: scale(1.02); }
        100% { opacity: 1; transform: scale(1); }
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: #1a1a2e;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #666;
    }

    .section-header {
        background: linear-gradient(90deg, #f8f9fa, #e9ecef);
        padding: 0.8rem 1.5rem;
        border-radius: 8px;
        border-left: 4px solid #007bff;
        margin: 1rem 0;
    }
    </style>
    """, unsafe_allow_html=True)


def render_risk_card(risk_level, message):
    """Render a styled risk card."""
    css_class = f"risk-{risk_level}"
    return f'<div class="risk-card {css_class}">{message}</div>'


def get_risk_color(risk_level):
    """Get color for risk level."""
    colors = {
        "safe": "#28a745",
        "low": "#ffc107",
        "moderate": "#ff9800",
        "high": "#f44336",
        "critical": "#b71c1c"
    }
    return colors.get(risk_level, "#999")


def get_risk_numeric(risk_level):
    """Convert risk level to numeric score."""
    risk_map = {"safe": 0, "low": 25, "moderate": 50, "high": 75, "critical": 100}
    return risk_map.get(risk_level, 0)


def create_interactive_count_chart(frame_data):
    """Create interactive person count line chart with Plotly."""
    if not frame_data:
        return None
    
    timestamps = [d.get("timestamp", d.get("time", 0)) for d in frame_data]
    counts = [d.get("exact_count", d.get("count", 0)) for d in frame_data]
    
    fig = go.Figure()
    
    # Add line trace
    fig.add_trace(go.Scatter(
        x=timestamps,
        y=counts,
        mode='lines+markers',
        name='Person Count',
        line=dict(color='#007bff', width=3),
        marker=dict(size=6, color='#007bff'),
        hovertemplate='<b>Time:</b> %{x:.1f}s<br><b>Count:</b> %{y}<extra></extra>'
    ))
    
    # Add area fill
    fig.add_trace(go.Scatter(
        x=timestamps,
        y=counts,
        fill='tozeroy',
        fillcolor='rgba(0,123,255,0.2)',
        line=dict(color='rgba(0,0,0,0)'),
        showlegend=False,
        hoverinfo='skip'
    ))
    
    # Add average line
    avg_count = np.mean(counts) if counts else 0
    fig.add_hline(y=avg_count, line_dash="dash", line_color="red", 
                  annotation_text=f"Avg: {avg_count:.1f}",
                  annotation_position="top right")
    
    fig.update_layout(
        title=dict(text="Person Count Over Time", font=dict(size=16)),
        xaxis_title="Time (seconds)",
        yaxis_title="Person Count",
        template="plotly_white",
        hovermode='x unified',
        height=350,
        margin=dict(l=40, r=40, t=50, b=40)
    )
    
    return fig


def create_interactive_risk_chart(frame_data):
    """Create interactive risk level bar chart with Plotly."""
    if not frame_data:
        return None
    
    timestamps = [d.get("timestamp", d.get("time", 0)) for d in frame_data]
    risk_scores = []
    colors = []
    
    for d in frame_data:
        risk_info = d.get("risk_info", {})
        risk_level = risk_info.get("risk_level", "safe") if risk_info else "safe"
        risk_scores.append(get_risk_numeric(risk_level))
        colors.append(get_risk_color(risk_level))
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=timestamps,
        y=risk_scores,
        marker_color=colors,
        name='Risk Score',
        hovertemplate='<b>Time:</b> %{x:.1f}s<br><b>Risk:</b> %{y}<extra></extra>'
    ))
    
    # Add threshold lines
    fig.add_hline(y=50, line_dash="dash", line_color="orange", 
                  annotation_text="Moderate", annotation_position="top right")
    fig.add_hline(y=75, line_dash="dash", line_color="red", 
                  annotation_text="High", annotation_position="top right")
    
    fig.update_layout(
        title=dict(text="Risk Level Over Time", font=dict(size=16)),
        xaxis_title="Time (seconds)",
        yaxis_title="Risk Score",
        yaxis_range=[0, 110],
        template="plotly_white",
        height=350,
        margin=dict(l=40, r=40, t=50, b=40)
    )
    
    return fig


def create_risk_distribution_pie(risk_dist):
    """Create interactive risk distribution pie chart."""
    if not risk_dist:
        return None
    
    # Filter out zero values
    labels = []
    values = []
    colors = []
    
    for level in ["safe", "low", "moderate", "high", "critical"]:
        if level in risk_dist and risk_dist[level] > 0:
            labels.append(level.capitalize())
            values.append(risk_dist[level])
            colors.append(get_risk_color(level))
    
    if not labels:
        return None
    
    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        marker=dict(colors=colors),
        hole=0.3,
        textinfo='label+percent',
        textposition='inside',
        hovertemplate='<b>%{label}</b><br>Frames: %{value}<br>Percentage: %{percent}<extra></extra>'
    )])
    
    fig.update_layout(
        title=dict(text="Risk Level Distribution", font=dict(size=16)),
        template="plotly_white",
        height=350,
        margin=dict(l=40, r=40, t=50, b=40),
        showlegend=True
    )
    
    return fig


def create_risk_distribution_bar(risk_dist):
    """Create interactive risk distribution bar chart."""
    if not risk_dist:
        return None
    
    levels = ["Safe", "Low", "Moderate", "High", "Critical"]
    values = [risk_dist.get(level.lower(), 0) for level in levels]
    colors = [get_risk_color(level.lower()) for level in levels]
    
    fig = go.Figure(data=[go.Bar(
        x=levels,
        y=values,
        marker_color=colors,
        text=values,
        textposition='auto',
        hovertemplate='<b>%{x}</b><br>Frames: %{y}<extra></extra>'
    )])
    
    fig.update_layout(
        title=dict(text="Risk Level Frame Count", font=dict(size=16)),
        xaxis_title="Risk Level",
        yaxis_title="Number of Frames",
        template="plotly_white",
        height=350,
        margin=dict(l=40, r=40, t=50, b=40)
    )
    
    return fig


def create_cluster_analysis_chart(frame_data):
    """Create interactive cluster analysis chart."""
    if not frame_data:
        return None
    
    timestamps = [d.get("timestamp", d.get("time", 0)) for d in frame_data]
    close_pairs = []
    people_in_clusters = []
    
    for d in frame_data:
        risk_info = d.get("risk_info", {})
        if risk_info:
            close_pairs.append(risk_info.get("total_close_pairs", 0))
            people_in_clusters.append(risk_info.get("total_in_clusters", 0))
        else:
            close_pairs.append(0)
            people_in_clusters.append(0)
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Close Pairs Over Time", "People in Risk Clusters")
    )
    
    fig.add_trace(
        go.Scatter(
            x=timestamps, y=close_pairs,
            mode='lines+markers',
            name='Close Pairs',
            line=dict(color='#ffc107', width=2),
            marker=dict(size=5),
            fill='tozeroy',
            fillcolor='rgba(255,193,7,0.2)'
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=timestamps, y=people_in_clusters,
            mode='lines+markers',
            name='In Clusters',
            line=dict(color='#f44336', width=2),
            marker=dict(size=5),
            fill='tozeroy',
            fillcolor='rgba(244,67,54,0.2)'
        ),
        row=1, col=2
    )
    
    fig.update_layout(
        title=dict(text="Proximity Analysis", font=dict(size=16)),
        template="plotly_white",
        height=350,
        margin=dict(l=40, r=40, t=60, b=40),
        showlegend=True
    )
    
    fig.update_xaxes(title_text="Time (seconds)", row=1, col=1)
    fig.update_xaxes(title_text="Time (seconds)", row=1, col=2)
    fig.update_yaxes(title_text="Count", row=1, col=1)
    fig.update_yaxes(title_text="People", row=1, col=2)
    
    return fig


def create_risk_heatmap(frame_data):
    """Create a heatmap showing risk levels over time."""
    if not frame_data:
        return None
    
    timestamps = [d.get("timestamp", d.get("time", 0)) for d in frame_data]
    risk_levels = []
    
    for d in frame_data:
        risk_info = d.get("risk_info", {})
        level = risk_info.get("risk_level", "safe") if risk_info else "safe"
        risk_levels.append(get_risk_numeric(level))
    
    # Create heatmap data (single row)
    fig = go.Figure(data=go.Heatmap(
        z=[risk_levels],
        x=timestamps,
        y=['Risk Level'],
        colorscale=[
            [0, '#28a745'],    # Safe - Green
            [0.25, '#ffc107'], # Low - Yellow
            [0.5, '#ff9800'],  # Moderate - Orange
            [0.75, '#f44336'], # High - Red
            [1, '#b71c1c']     # Critical - Dark Red
        ],
        colorbar=dict(
            title='Risk',
            tickvals=[0, 25, 50, 75, 100],
            ticktext=['Safe', 'Low', 'Moderate', 'High', 'Critical']
        ),
        hovertemplate='<b>Time:</b> %{x:.1f}s<br><b>Risk Score:</b> %{z}<extra></extra>'
    ))
    
    fig.update_layout(
        title=dict(text="Risk Level Heatmap", font=dict(size=16)),
        xaxis_title="Time (seconds)",
        height=150,
        margin=dict(l=40, r=40, t=50, b=40)
    )
    
    return fig


def create_risk_gauge(current_risk_score, risk_level):
    """Create a gauge chart showing current risk level."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=current_risk_score,
        title={'text': f"Current Risk: {risk_level.upper()}", 'font': {'size': 20}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1},
            'bar': {'color': get_risk_color(risk_level.lower())},
            'steps': [
                {'range': [0, 25], 'color': '#d4edda'},
                {'range': [25, 50], 'color': '#fff3cd'},
                {'range': [50, 75], 'color': '#ffe0b2'},
                {'range': [75, 100], 'color': '#ffcdd2'}
            ],
            'threshold': {
                'line': {'color': 'red', 'width': 4},
                'thickness': 0.75,
                'value': current_risk_score
            }
        }
    ))
    
    fig.update_layout(
        height=250,
        margin=dict(l=40, r=40, t=60, b=20)
    )
    
    return fig


def create_risk_level_timeline(frame_data):
    """
    Create a prominent risk level timeline chart.
    Shows risk level changes over time with color-coded background zones.
    """
    if not frame_data:
        return None
    
    timestamps = [d.get("timestamp", d.get("time", 0)) for d in frame_data]
    risk_scores = []
    risk_levels = []
    hover_texts = []
    
    for d in frame_data:
        risk_info = d.get("risk_info", {})
        if risk_info:
            level = risk_info.get("risk_level", "safe")
            message = risk_info.get("risk_message", "")
            close_pairs = risk_info.get("total_close_pairs", 0)
            in_clusters = risk_info.get("total_in_clusters", 0)
        else:
            level = "safe"
            message = "No data"
            close_pairs = 0
            in_clusters = 0
        
        risk_scores.append(get_risk_numeric(level))
        risk_levels.append(level.capitalize())
        hover_texts.append(
            f"<b>Time:</b> {d.get('timestamp', 0):.1f}s<br>"
            f"<b>Risk:</b> {level.capitalize()}<br>"
            f"<b>Message:</b> {message}<br>"
            f"<b>Close Pairs:</b> {close_pairs}<br>"
            f"<b>In Clusters:</b> {in_clusters}"
        )
    
    fig = go.Figure()
    
    # Add colored background zones
    fig.add_vrect(x0=0, x1=1, y0=0, y1=25, fillcolor="green", opacity=0.1, line_width=0, layer="below")
    fig.add_vrect(x0=0, x1=1, y0=25, y1=50, fillcolor="yellow", opacity=0.1, line_width=0, layer="below")
    fig.add_vrect(x0=0, x1=1, y0=50, y1=75, fillcolor="orange", opacity=0.1, line_width=0, layer="below")
    fig.add_vrect(x0=0, x1=1, y0=75, y1=100, fillcolor="red", opacity=0.1, line_width=0, layer="below")
    
    # Add risk level line
    fig.add_trace(go.Scatter(
        x=timestamps,
        y=risk_scores,
        mode='lines+markers',
        name='Risk Level',
        line=dict(color='#1a1a2e', width=3),
        marker=dict(
            size=10,
            color=risk_scores,
            colorscale=[[0, '#28a745'], [0.25, '#ffc107'], [0.5, '#ff9800'], [0.75, '#f44336'], [1, '#b71c1c']],
            line=dict(width=2, color='white')
        ),
        text=hover_texts,
        hoverinfo='text'
    ))
    
    # Add threshold lines with labels
    fig.add_hline(y=25, line_dash="dot", line_color="#ffc107", line_width=1,
                  annotation_text="LOW", annotation_position="left")
    fig.add_hline(y=50, line_dash="dot", line_color="#ff9800", line_width=1,
                  annotation_text="MODERATE", annotation_position="left")
    fig.add_hline(y=75, line_dash="dot", line_color="#f44336", line_width=1,
                  annotation_text="HIGH", annotation_position="left")
    
    # Add risk level labels on right side
    fig.add_annotation(x=timestamps[-1] if timestamps else 0, y=0, text="SAFE", 
                       showarrow=False, font=dict(color="green", size=10), xanchor="left")
    fig.add_annotation(x=timestamps[-1] if timestamps else 0, y=25, text="LOW", 
                       showarrow=False, font=dict(color="#ffc107", size=10), xanchor="left")
    fig.add_annotation(x=timestamps[-1] if timestamps else 0, y=50, text="MODERATE", 
                       showarrow=False, font=dict(color="#ff9800", size=10), xanchor="left")
    fig.add_annotation(x=timestamps[-1] if timestamps else 0, y=75, text="HIGH", 
                       showarrow=False, font=dict(color="#f44336", size=10), xanchor="left")
    fig.add_annotation(x=timestamps[-1] if timestamps else 0, y=100, text="CRITICAL", 
                       showarrow=False, font=dict(color="#b71c1c", size=10), xanchor="left")
    
    fig.update_layout(
        title=dict(
            text="RISK LEVEL TIMELINE", 
            font=dict(size=18, color="#1a1a2e"),
            x=0.5
        ),
        xaxis_title="Time (seconds)",
        yaxis_title="Risk Score",
        yaxis=dict(
            range=[-5, 110],
            tickvals=[0, 25, 50, 75, 100],
            ticktext=["Safe", "Low", "Moderate", "High", "Critical"]
        ),
        template="plotly_white",
        height=400,
        margin=dict(l=60, r=80, t=60, b=40),
        hovermode='x unified',
        showlegend=False
    )
    
    return fig


def create_count_histogram(frame_data):
    """Create interactive count histogram."""
    if not frame_data:
        return None
    
    counts = [d.get("exact_count", d.get("count", 0)) for d in frame_data]
    
    fig = go.Figure(data=[go.Histogram(
        x=counts,
        nbinsx=max(1, max(counts) - min(counts) + 1) if counts else 10,
        marker_color='#007bff',
        opacity=0.7,
        hovertemplate='<b>Count:</b> %{x}<br><b>Frames:</b> %{y}<extra></extra>'
    )])
    
    fig.update_layout(
        title=dict(text="Person Count Distribution", font=dict(size=16)),
        xaxis_title="Person Count",
        yaxis_title="Frequency (Frames)",
        template="plotly_white",
        height=350,
        margin=dict(l=40, r=40, t=50, b=40)
    )
    
    return fig


def run_analysis(video_path, config, progress_bar=None):
    """Run video analysis with progress reporting."""
    from src.video.processor import VideoProcessor

    def streamlit_progress(pct, msg):
        """Update Streamlit progress bar."""
        if progress_bar:
            progress_bar.progress(min(pct, 1.0), text=msg)

    # Use absolute path for outputs
    output_dir = Path(__file__).parent.parent / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    processor = VideoProcessor(config)
    results = processor.process_video(
        video_path,
        output_dir=output_dir,
        progress_callback=streamlit_progress,
    )
    return results


def video_analysis_page(config):
    """Video analysis page with interactive charts."""
    st.header("Upload & Analyze Video")
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose a video file",
        type=["mp4", "avi", "mov", "mkv"],
        help="Upload a crowd video for person counting with risk detection",
    )

    if uploaded_file is None:
        st.info("Upload a video file to start analysis.")
        return

    # Save uploaded file
    temp_dir = Path(tempfile.mkdtemp())
    temp_video = temp_dir / uploaded_file.name
    with open(temp_video, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # Verify file was saved
    if not temp_video.exists():
        st.error("Failed to save uploaded video file.")
        return

    # Display uploaded video
    st.video(str(temp_video))
    st.success(f"Video uploaded: {uploaded_file.name} ({uploaded_file.size/1024/1024:.1f} MB)")

    # Run analysis button
    if st.button("Run Analysis", type="primary", use_container_width=True):
        progress_bar = st.progress(0, text="Initializing...")
        try:
            results = run_analysis(str(temp_video), config, progress_bar=progress_bar)
            if results and results.get("processed_frames", 0) > 0:
                progress_bar.progress(1.0, text="Done!")
                display_video_results(results, temp_video, config)
            else:
                st.error("No frames were processed. Please check the video file.")
        except Exception as e:
            st.error(f"Analysis failed: {str(e)}")
            import traceback
            st.code(traceback.format_exc())
        finally:
            progress_bar.empty()


def display_video_results(results, temp_video, config):
    """Display video analysis results with interactive charts."""
    stats = results.get("statistics", {})
    duration = results.get("video_duration_seconds", 0)
    frame_data = results.get("frame_data", [])
    risk_dist = stats.get("risk_level_distribution", {})

    # ===== PERSON COUNT (PROMINENT) =====
    st.markdown('<div class="section-header"><h3>Person Count</h3></div>', unsafe_allow_html=True)
    
    avg_count = stats.get("avg_count", 0)
    peak_count = stats.get("peak_count", 0)
    min_count = stats.get("min_count", 0)
    processed_frames = results.get("processed_frames", 0)
    
    # Large person count display
    count_col1, count_col2, count_col3, count_col4 = st.columns(4)
    with count_col1:
        st.metric(
            label="Average Persons Per Frame",
            value=f"{avg_count:.0f}",
            delta=None,
            help="Average number of people detected across all frames"
        )
    with count_col2:
        st.metric(
            label="Peak Persons Detected",
            value=f"{peak_count}",
            delta=None,
            help="Maximum people detected in a single frame"
        )
    with count_col3:
        st.metric(
            label="Min Persons Detected",
            value=f"{min_count}",
            delta=None,
            help="Minimum people detected in a single frame"
        )
    with count_col4:
        st.metric(
            label="Total Frames Analyzed",
            value=f"{processed_frames}",
            delta=None,
            help="Number of frames processed"
        )
    
    # Person count timeline
    if frame_data:
        timestamps = [d.get("timestamp", 0) for d in frame_data]
        counts = [d.get("exact_count", 0) for d in frame_data]
        
        fig_count = go.Figure()
        fig_count.add_trace(go.Scatter(
            x=timestamps,
            y=counts,
            mode='lines+markers',
            name='Person Count',
            line=dict(color='#007bff', width=3),
            marker=dict(size=8, color='#007bff'),
            hovertemplate='<b>Time:</b> %{x:.1f}s<br><b>Persons:</b> %{y}<extra></extra>'
        ))
        fig_count.add_trace(go.Scatter(
            x=timestamps,
            y=counts,
            fill='tozeroy',
            fillcolor='rgba(0,123,255,0.15)',
            line=dict(color='rgba(0,0,0,0)'),
            showlegend=False,
            hoverinfo='skip'
        ))
        fig_count.add_hline(y=avg_count, line_dash="dash", line_color="red",
                          annotation_text=f"Average: {avg_count:.0f}",
                          annotation_position="top right")
        fig_count.update_layout(
            title=dict(text="Person Count Per Frame", font=dict(size=16)),
            xaxis_title="Time (seconds)",
            yaxis_title="Number of Persons",
            template="plotly_white",
            hovermode='x unified',
            height=350,
            margin=dict(l=40, r=40, t=50, b=40)
        )
        st.plotly_chart(fig_count, use_container_width=True)

    # ===== OVERALL RISK ASSESSMENT =====
    st.markdown('<div class="section-header"><h3>Overall Risk Assessment</h3></div>', unsafe_allow_html=True)
    
    risk_frames = stats.get("risk_frames", 0)
    risk_pct = stats.get("risk_frame_pct", 0)
    
    # Calculate overall risk score
    risk_dist = stats.get("risk_level_distribution", {})
    total_frames = sum(risk_dist.values()) if risk_dist else 1
    
    # Weighted risk score
    risk_weights = {"safe": 0, "low": 25, "moderate": 50, "high": 75, "critical": 100}
    overall_score = sum(risk_weights.get(k, 0) * v for k, v in risk_dist.items()) / total_frames if total_frames > 0 else 0
    
    # Determine overall risk level
    if risk_pct > 20:
        overall_risk = "critical"
        overall_message = f"CRITICAL: Risk in {risk_pct:.1f}% of frames ({risk_frames} frames)"
    elif risk_pct > 10:
        overall_risk = "high"
        overall_message = f"HIGH RISK: {risk_pct:.1f}% of frames ({risk_frames} frames)"
    elif risk_pct > 0:
        overall_risk = "moderate"
        overall_message = f"MODERATE: Risk in {risk_pct:.1f}% of frames ({risk_frames} frames)"
    else:
        overall_risk = "safe"
        overall_message = "SAFE: No risk clusters detected"
    
    st.markdown(render_risk_card(overall_risk, overall_message), unsafe_allow_html=True)
    
    # Risk Gauge
    fig_gauge = create_risk_gauge(overall_score, overall_risk)
    st.plotly_chart(fig_gauge, use_container_width=True)

    # ===== KEY METRICS CARDS =====
    st.markdown('<div class="section-header"><h3>Key Metrics</h3></div>', unsafe_allow_html=True)
    
    # Create metric cards in columns
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Processed Frames",
            value=f"{results.get('processed_frames', 0)}",
            delta=None
        )
    with col2:
        st.metric(
            label="Total Frames in Video",
            value=f"{results.get('total_frames', 0)}",
            delta=None
        )
    with col3:
        st.metric(
            label="Max Frames Limit",
            value=f"{results.get('max_frames', 'All')}",
            delta=None
        )
    with col4:
        st.metric(
            label="Frame Skip",
            value=f"Every {config['video'].get('frame_skip', 1)} frame(s)",
            delta=None
        )

    col5, col6, col7, col8 = st.columns(4)
    
    with col5:
        st.metric(label="Duration", value=f"{duration:.1f}s")
    with col6:
        res = results.get("resolution", [0, 0])
        st.metric(label="Resolution", value=f"{res[0]}x{res[1]}")
    with col7:
        st.metric(
            label="Risk Frames",
            value=f"{risk_frames}",
            delta=f"{risk_pct:.1f}%"
        )
    with col8:
        high_crit = risk_dist.get("critical", 0) + risk_dist.get("high", 0)
        st.metric(label="High/Critical", value=f"{high_crit}")

    # ===== RISK LEVEL TIMELINE (PROMINENT) =====
    st.markdown('<div class="section-header"><h3>RISK LEVEL TIMELINE</h3></div>', unsafe_allow_html=True)
    
    fig_risk_timeline = create_risk_level_timeline(frame_data)
    if fig_risk_timeline:
        st.plotly_chart(fig_risk_timeline, use_container_width=True)
    
    # Risk level legend
    st.markdown("""
    <div style="display: flex; justify-content: center; gap: 20px; margin: 10px 0; padding: 10px; background: #f8f9fa; border-radius: 8px;">
        <span style="color: #28a745; font-weight: bold;">● SAFE (0-25)</span>
        <span style="color: #ffc107; font-weight: bold;">● LOW (25-50)</span>
        <span style="color: #ff9800; font-weight: bold;">● MODERATE (50-75)</span>
        <span style="color: #f44336; font-weight: bold;">● HIGH (75-100)</span>
        <span style="color: #b71c1c; font-weight: bold;">● CRITICAL (100)</span>
    </div>
    """, unsafe_allow_html=True)
    
    # Risk Heatmap
    fig_heatmap = create_risk_heatmap(frame_data)
    if fig_heatmap:
        st.plotly_chart(fig_heatmap, use_container_width=True)

    # ===== RISK DISTRIBUTION CHARTS =====
    st.markdown('<div class="section-header"><h3>Risk Distribution</h3></div>', unsafe_allow_html=True)
    
    risk_col1, risk_col2 = st.columns(2)
    
    with risk_col1:
        fig_bar = create_risk_distribution_bar(risk_dist)
        if fig_bar:
            st.plotly_chart(fig_bar, use_container_width=True)
    
    with risk_col2:
        fig_pie = create_risk_distribution_pie(risk_dist)
        if fig_pie:
            st.plotly_chart(fig_pie, use_container_width=True)

    # ===== TIME SERIES CHARTS =====
    st.markdown('<div class="section-header"><h3>Time Series Analysis</h3></div>', unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs(["Person Count", "Risk Level", "Proximity Analysis"])
    
    with tab1:
        fig_count = create_interactive_count_chart(frame_data)
        if fig_count:
            st.plotly_chart(fig_count, use_container_width=True)
    
    with tab2:
        fig_risk = create_interactive_risk_chart(frame_data)
        if fig_risk:
            st.plotly_chart(fig_risk, use_container_width=True)
    
    with tab3:
        fig_cluster = create_cluster_analysis_chart(frame_data)
        if fig_cluster:
            st.plotly_chart(fig_cluster, use_container_width=True)

    # ===== DISTRIBUTION HISTOGRAM =====
    st.markdown('<div class="section-header"><h3>Count Distribution</h3></div>', unsafe_allow_html=True)
    
    fig_hist = create_count_histogram(frame_data)
    if fig_hist:
        st.plotly_chart(fig_hist, use_container_width=True)

    # ===== VIDEO COMPARISON =====
    st.markdown('<div class="section-header"><h3>Video Comparison</h3></div>', unsafe_allow_html=True)
    
    video_col1, video_col2 = st.columns(2)
    with video_col1:
        st.subheader("Original Video")
        st.video(str(temp_video))
    with video_col2:
        st.subheader("Annotated Output")
        output_video = results.get("output_video")
        if output_video and Path(output_video).exists():
            with open(output_video, "rb") as vf:
                st.video(vf.read(), format="video/mp4")
        else:
            st.warning("Processed video not available")

    # ===== FRAME-BY-FRAME DATA =====
    st.markdown('<div class="section-header"><h3>Frame-by-Frame Data</h3></div>', unsafe_allow_html=True)
    
    if frame_data:
        # Build dataframe
        frame_rows = []
        for d in frame_data:
            risk_info = d.get("risk_info", {})
            frame_rows.append({
                "Frame": d.get("frame_idx", d.get("frame", 0)),
                "Time (s)": d.get("timestamp", d.get("time", 0)),
                "Count": d.get("exact_count", d.get("count", 0)),
                "Risk Level": risk_info.get("risk_level", "safe") if risk_info else "safe",
                "Risk Message": risk_info.get("risk_message", "No data") if risk_info else "No data",
                "Close Pairs": risk_info.get("total_close_pairs", 0) if risk_info else 0,
                "In Clusters": risk_info.get("total_in_clusters", 0) if risk_info else 0,
            })
        
        df = pd.DataFrame(frame_rows)
        
        # Style the dataframe
        def highlight_risk(val):
            color = get_risk_color(str(val).lower())
            return f'background-color: {color}20; color: {color}; font-weight: bold'
        
        st.dataframe(
            df.style.applymap(highlight_risk, subset=["Risk Level"]),
            use_container_width=True,
            height=400
        )
        
        # Summary stats
        st.subheader("Summary Statistics")
        sum_col1, sum_col2, sum_col3 = st.columns(3)
        
        with sum_col1:
            safe_frames = len(df[df["Risk Level"] == "safe"])
            st.metric("Safe Frames", f"{safe_frames}", f"{safe_frames/len(df)*100:.1f}%")
        
        with sum_col2:
            risk_count = len(df[df["Risk Level"] != "safe"])
            st.metric("Risk Frames", f"{risk_count}", f"{risk_count/len(df)*100:.1f}%")
        
        with sum_col3:
            max_cluster = df["In Clusters"].max()
            st.metric("Max in Cluster", f"{max_cluster}")

    # ===== DOWNLOAD SECTION =====
    st.markdown('<div class="section-header"><h3>Download Results</h3></div>', unsafe_allow_html=True)
    
    dl_col1, dl_col2 = st.columns(2)
    
    with dl_col1:
        analysis_summary = {
            "video_path": results.get("video_path", ""),
            "total_frames": results.get("total_frames", 0),
            "processed_frames": results.get("processed_frames", 0),
            "duration_seconds": duration,
            "resolution": results.get("resolution", []),
            "statistics": stats,
            "frame_data": frame_rows if frame_data else []
        }
        st.download_button(
            "Download Analysis (JSON)",
            data=json.dumps(analysis_summary, indent=2),
            file_name="crowd_analysis.json",
            mime="application/json",
            use_container_width=True
        )
    
    with dl_col2:
        output_video = results.get("output_video")
        if output_video and Path(output_video).exists():
            with open(output_video, "rb") as f:
                st.download_button(
                    "Download Annotated Video",
                    data=f.read(),
                    file_name=Path(output_video).name,
                    mime="video/mp4",
                    use_container_width=True
                )



def main():
    """Main dashboard function."""
    config = load_config()

    st.set_page_config(
        page_title=config["dashboard"]["title"],
        page_icon="crowd",
        layout=config["dashboard"]["layout"],
    )

    inject_custom_css()

    st.title("Kumbh Mela Crowd Detection System")
    st.markdown("---")

    st.warning(
        "**DISCLAIMER:** Risk scores are PROTOTYPE/MODEL-DERIVED indicators only. "
        "They are NOT official safety assessments."
    )

    # Sidebar
    with st.sidebar:
        st.header("Settings")
        st.markdown(f"- **Detector:** YOLOv8 (COCO pretrained)")
        st.markdown(f"- **Confidence:** {config['video'].get('confidence', 0.35)}")
        st.markdown(f"- **Model:** {config['video'].get('yolo_model', 'yolov8n')}")
        st.markdown("---")
        st.header("How It Works")
        st.markdown("""
        1. Upload a video
        2. YOLOv8 detects persons
        3. System analyzes proximity
        4. Risk clusters are flagged
        5. Results displayed with charts
        """)

    video_analysis_page(config)


if __name__ == "__main__":
    main()
