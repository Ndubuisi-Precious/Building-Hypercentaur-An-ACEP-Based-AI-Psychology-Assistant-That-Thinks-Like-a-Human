"""
Visualization components for Hypercentaur reasoning chains and concept networks.
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import networkx as nx
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple
import altair as alt
from pyvis.network import Network
import tempfile
import os

class HypercentaurVisualizer:
    """Handles all visualization components for Hypercentaur."""
    
    def __init__(self):
        self.color_scheme = {
            'primary': '#000000',
            'secondary': '#4682B4', 
            'accent': '#FF6347',
            'background': '#F8F9FA',
            'text': '#2F4F4F'
        }
    
    def visualize_reasoning_chain(self, reasoning_steps: List[Dict[str, Any]]) -> go.Figure:
        """Create an interactive reasoning chain visualization."""
        if not reasoning_steps:
            return self._create_empty_figure("No reasoning steps to display")
        
        # Prepare data for visualization
        step_data = []
        for i, step in enumerate(reasoning_steps):
            step_data.append({
                'step': i + 1,
                'type': step.get('reasoning_type', 'unknown'),
                'confidence': step.get('confidence', 0),
                'explanation': step.get('explanation', ''),
                'input_concepts': len(step.get('input_concepts', [])),
                'output_concepts': len(step.get('output_concepts', []))
            })
        
        df = pd.DataFrame(step_data)
        
        # Create subplot figure
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Reasoning Flow', 'Confidence Progression', 
                          'Concept Flow', 'Reasoning Types'),
            specs=[[{"colspan": 2}, None],
                   [{"type": "scatter"}, {"type": "bar"}]]
        )
        
        # 1. Main reasoning flow (Sankey-like visualization)
        self._add_reasoning_flow(fig, reasoning_steps, row=1, col=1)
        
        # 2. Confidence progression
        fig.add_trace(
            go.Scatter(
                x=df['step'],
                y=df['confidence'],
                mode='lines+markers',
                name='Confidence',
                line=dict(color=self.color_scheme['primary'], width=3),
                marker=dict(size=8, color=self.color_scheme['accent'])
            ),
            row=2, col=1
        )
        
        # 3. Reasoning types distribution
        type_counts = df['type'].value_counts()
        fig.add_trace(
            go.Bar(
                x=type_counts.index,
                y=type_counts.values,
                name='Reasoning Types',
                marker_color=self.color_scheme['secondary']
            ),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            title="Reasoning Chain Analysis",
            height=700,
            showlegend=True,
            font=dict(family="Arial", size=12)
        )
        
        # Update axis labels
        fig.update_xaxes(title_text="Reasoning Step", row=2, col=1)
        fig.update_yaxes(title_text="Confidence", row=2, col=1)
        fig.update_xaxes(title_text="Reasoning Type", row=2, col=2)
        fig.update_yaxes(title_text="Count", row=2, col=2)
        
        return fig
    
    def _add_reasoning_flow(self, fig: go.Figure, reasoning_steps: List[Dict[str, Any]], row: int, col: int):
        """Add reasoning flow visualization to subplot."""
        # Create a flow chart representation
        x_positions = []
        y_positions = []
        text_labels = []
        colors = []
        
        for i, step in enumerate(reasoning_steps):
            x_positions.append(i)
            y_positions.append(0)
            text_labels.append(f"Step {i+1}<br>{step.get('reasoning_type', 'unknown')}")
            
            # Color based on confidence
            confidence = step.get('confidence', 0)
            if confidence > 0.8:
                colors.append(self.color_scheme['primary'])
            elif confidence > 0.6:
                colors.append(self.color_scheme['secondary'])
            else:
                colors.append(self.color_scheme['accent'])
        
        # Add scatter plot for steps
        fig.add_trace(
            go.Scatter(
                x=x_positions,
                y=y_positions,
                mode='markers+text',
                text=text_labels,
                textposition="top center",
                marker=dict(
                    size=[step.get('confidence', 0.5) * 30 + 20 for step in reasoning_steps],
                    color=colors,
                    line=dict(width=2, color='white')
                ),
                name='Reasoning Steps',
                showlegend=False
            ),
            row=row, col=col
        )
        
        # Add arrows between steps
        for i in range(len(reasoning_steps) - 1):
            fig.add_annotation(
                x=i + 0.5, y=0,
                ax=i, ay=0,
                xref=f'x{col}', yref=f'y{row}',
                axref=f'x{col}', ayref=f'y{row}',
                arrowhead=2,
                arrowsize=1,
                arrowwidth=2,
                arrowcolor=self.color_scheme['text'],
                showarrow=True,
                row=row, col=col
            )
    
    def visualize_concept_network(self, concept_graph: nx.Graph, 
                                activated_concepts: List[str] = None) -> go.Figure:
        """Create an interactive concept network visualization."""
        if not concept_graph or len(concept_graph.nodes()) == 0:
            return self._create_empty_figure("No concept network to display")
        
        activated_concepts = activated_concepts or []
        
        # Use spring layout for positioning
        pos = nx.spring_layout(concept_graph, k=1, iterations=50)
        
        # Prepare edge traces
        edge_x = []
        edge_y = []
        edge_info = []
        
        for edge in concept_graph.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            
            weight = concept_graph[edge[0]][edge[1]].get('weight', 1)
            edge_info.append(f"{edge[0]} ↔ {edge[1]}<br>Weight: {weight}")
        
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=0.5, color='#888'),
            hoverinfo='none',
            mode='lines'
        )
        
        # Prepare node traces
        node_x = []
        node_y = []
        node_text = []
        node_colors = []
        node_sizes = []
        
        for node in concept_graph.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            
            # Node information
            node_info = concept_graph.nodes[node]
            frequency = node_info.get('frequency', 1)
            category = node_info.get('category', 'unknown')
            
            node_text.append(f"{node}<br>Frequency: {frequency}<br>Category: {category}")
            
            # Color based on activation status
            if node in activated_concepts:
                node_colors.append(self.color_scheme['accent'])
                node_sizes.append(max(frequency * 5 + 20, 25))
            else:
                node_colors.append(self.color_scheme['primary'])
                node_sizes.append(max(frequency * 3 + 15, 20))
        
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            hoverinfo='text',
            text=[node.replace('_', ' ') for node in concept_graph.nodes()],
            textposition="middle center",
            hovertext=node_text,
            marker=dict(
                size=node_sizes,
                color=node_colors,
                line=dict(width=2, color='white')
            )
        )
        
        # Create figure
        fig = go.Figure(data=[edge_trace, node_trace],
                       layout=go.Layout(
                        title=dict(text='Concept Network', font=dict(size=16)),
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=20,l=5,r=5,t=40),
                        annotations=[ dict(
                            text="Larger nodes = higher frequency<br>Red nodes = activated concepts",
                            showarrow=False,
                            xref="paper", yref="paper",
                            x=0.005, y=-0.002,
                            xanchor="left", yanchor="bottom",
                            font=dict(color="gray", size=10)
                        )],
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        height=600
                        ))
        
        return fig
    
    def visualize_confidence_breakdown(self, reasoning_result: Dict[str, Any]) -> go.Figure:
        """Create a confidence breakdown visualization."""
        concept_certainties = reasoning_result.get('concept_certainties', {})
        overall_confidence = reasoning_result.get('confidence', 0)
        
        if not concept_certainties:
            return self._create_empty_figure("No confidence data available")
        
        # Prepare data
        concepts = list(concept_certainties.keys())
        certainties = list(concept_certainties.values())
        
        # Create color scale based on certainty
        colors = ['red' if c < 0.5 else 'orange' if c < 0.7 else 'green' for c in certainties]
        
        fig = go.Figure()
        
        # Add bar chart
        fig.add_trace(go.Bar(
            x=[c.replace('_', ' ') for c in concepts],
            y=certainties,
            marker_color=colors,
            text=[f"{c:.2f}" for c in certainties],
            textposition='auto',
            name='Concept Certainty'
        ))
        
        # Add overall confidence line
        fig.add_hline(
            y=overall_confidence,
            line_dash="dash",
            line_color=self.color_scheme['accent'],
            annotation_text=f"Overall Confidence: {overall_confidence:.2f}"
        )
        
        fig.update_layout(
            title="Concept Certainty Analysis",
            xaxis_title="Concepts",
            yaxis_title="Certainty Score",
            yaxis=dict(range=[0, 1]),
            height=400
        )
        
        return fig
    
    def create_reasoning_timeline(self, reasoning_steps: List[Dict[str, Any]]) -> alt.Chart:
        """Create an Altair timeline visualization of reasoning steps."""
        if not reasoning_steps:
            return alt.Chart(pd.DataFrame()).mark_text(text="No data")
        
        # Prepare timeline data
        timeline_data = []
        for i, step in enumerate(reasoning_steps):
            timeline_data.append({
                'step': i + 1,
                'type': step.get('reasoning_type', 'unknown'),
                'confidence': step.get('confidence', 0),
                'explanation': step.get('explanation', ''),
                'start_time': i,
                'end_time': i + 1
            })
        
        df = pd.DataFrame(timeline_data)
        
        # Create Gantt-like chart
        chart = alt.Chart(df).mark_bar(height=30).encode(
            x=alt.X('start_time:O', title='Reasoning Step'),
            x2='end_time:O',
            y=alt.Y('type:N', title='Reasoning Type'),
            color=alt.Color('confidence:Q', 
                          scale=alt.Scale(range=['red', 'orange', 'green']),
                          title='Confidence'),
            tooltip=['step:O', 'type:N', 'confidence:Q', 'explanation:N']
        ).properties(
            width=600,
            height=200,
            title="Reasoning Timeline"
        )
        
        return chart
    
    def create_pyvis_network(self, concept_graph: nx.Graph, 
                           activated_concepts: List[str] = None) -> str:
        """Create an interactive PyVis network visualization."""
        activated_concepts = activated_concepts or []
        
        # Create PyVis network
        net = Network(height="400px", width="100%", bgcolor="#ffffff", font_color="black")
        
        # Configure physics
        net.set_options("""
        var options = {
          "physics": {
            "enabled": true,
            "stabilization": {"iterations": 100}
          }
        }
        """)
        
        # Add nodes
        for node in concept_graph.nodes():
            node_info = concept_graph.nodes[node]
            frequency = node_info.get('frequency', 1)
            category = node_info.get('category', 'unknown')
            
            # Node properties
            size = max(frequency * 10 + 20, 25)
            color = self.color_scheme['accent'] if node in activated_concepts else self.color_scheme['primary']
            
            title = f"Concept: {node.replace('_', ' ')}\nFrequency: {frequency}\nCategory: {category}"
            
            net.add_node(node, 
                        label=node.replace('_', ' '), 
                        size=size, 
                        color=color,
                        title=title)
        
        # Add edges
        for edge in concept_graph.edges():
            weight = concept_graph[edge[0]][edge[1]].get('weight', 1)
            net.add_edge(edge[0], edge[1], width=weight)
        
        # Generate HTML
        with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False) as f:
            net.save_graph(f.name)
            return f.name
    
    def display_metrics_dashboard(self, analytics_data: Dict[str, Any]):
        """Display a metrics dashboard using Streamlit."""
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="Total Queries",
                value=analytics_data.get('total_queries', 0)
            )
        
        with col2:
            confidence_stats = analytics_data.get('confidence_stats', {})
            avg_confidence = confidence_stats.get('avg_confidence', 0)
            st.metric(
                label="Avg Confidence",
                value=f"{avg_confidence:.2f}" if avg_confidence else "N/A"
            )
        
        with col3:
            st.metric(
                label="Total Concepts",
                value=analytics_data.get('total_concepts', 0)
            )
        
        with col4:
            most_used = analytics_data.get('most_used_concepts', [])
            top_concept = most_used[0]['concept'] if most_used else "N/A"
            st.metric(
                label="Top Concept",
                value=top_concept.replace('_', ' ')
            )
        
        # Most used concepts chart
        if most_used:
            st.subheader("Most Used Concepts")
            concepts_df = pd.DataFrame(most_used)
            
            fig = px.bar(
                concepts_df.head(10),
                x='usage_count',
                y='concept',
                orientation='h',
                color='category',
                title="Top 10 Most Used Concepts"
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        # Category distribution
        category_dist = analytics_data.get('category_distribution', [])
        if category_dist:
            st.subheader("Concept Categories")
            
            categories_df = pd.DataFrame(category_dist)
            fig = px.pie(
                categories_df,
                values='count',
                names='_id',
                title="Distribution of Concept Categories"
            )
            st.plotly_chart(fig, use_container_width=True)
    
    def _create_empty_figure(self, message: str) -> go.Figure:
        """Create an empty figure with a message."""
        fig = go.Figure()
        fig.add_annotation(
            text=message,
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=16, color="gray")
        )
        fig.update_layout(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            height=400
        )
        return fig