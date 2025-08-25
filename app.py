"""
Hypercentaur - ACEP-based Psychological Reasoning Assistant
Main Streamlit application interface.
"""

import streamlit as st
import pandas as pd
import numpy as np
import time
from datetime import datetime
import logging
import traceback
import os
from typing import Dict, Any, List

# Import Hypercentaur modules
from data_loader import EnhancedPsych101DataLoader
from reasoning_engine import EnhancedACEPReasoningEngine
from db_manager import HypercentaurDBManager
from visualizer import HypercentaurVisualizer
from utils import (
    generate_session_id, hash_query, format_concept_name,
    calculate_reasoning_quality, create_concept_summary,
    get_system_stats
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Hypercentaur - ACEP Reasoning Assistant",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        text-align: center;
        color: #2E8B57;
        font-size: 2.5rem;
        font-weight: bold;
        margin-bottom: 2rem;
    }
    .reasoning-card {
        background-color: #000000;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #2E8B57;
        margin: 1rem 0;
    }
    .concept-tag {
        background-color: #000000;
        color: #2F4F4F;
        padding: 0.2rem 0.5rem;
        border-radius: 1rem;
        font-size: 0.8rem;
        margin: 0.2rem;
        display: inline-block;
    }
    .confidence-high { background-color: #000000; }
    .confidence-medium { background-color: #000000; }
    .confidence-low { background-color: #000000; }
    @keyframes blink {
        0%, 50% { opacity: 1; }
        51%, 100% { opacity: 0; }
    }
</style>
""", unsafe_allow_html=True)

class HypercentaurApp:
    """Main application class."""
    
    def __init__(self):
        self.data_loader = None
        self.reasoning_engine = None
        self.db_manager = None
        self.visualizer = HypercentaurVisualizer()
        self.session_id = self._get_session_id()
        
    def _get_session_id(self) -> str:
        """Get or create session ID."""
        if 'session_id' not in st.session_state:
            st.session_state.session_id = generate_session_id()
        return st.session_state.session_id
    
    def initialize_components(self) -> bool:
        """Initialize all Hypercentaur components."""
        try:
            with st.spinner("Initializing Hypercentaur components..."):
                # Initialize data loader
                if 'data_loader' not in st.session_state:
                    self.data_loader = EnhancedPsych101DataLoader()
                    processed_data = self.data_loader.process_dataset()
                    st.session_state.data_loader = self.data_loader
                    st.session_state.processed_data = processed_data
                else:
                    self.data_loader = st.session_state.data_loader
                
                # Initialize reasoning engine
                if 'reasoning_engine' not in st.session_state:
                    self.reasoning_engine = EnhancedACEPReasoningEngine(self.data_loader)
                    st.session_state.reasoning_engine = self.reasoning_engine
                else:
                    self.reasoning_engine = st.session_state.reasoning_engine
                
                # Initialize database manager
                if 'db_manager' not in st.session_state:
                    self.db_manager = HypercentaurDBManager()
                    if self.db_manager.connect():
                        st.session_state.db_manager = self.db_manager
                        st.session_state.db_connected = True
                    else:
                        st.session_state.db_connected = False
                        st.warning("⚠️ Database connection failed. Running without persistence.")
                else:
                    self.db_manager = st.session_state.db_manager
                
                return True
                
        except Exception as e:
            st.error(f"❌ Failed to initialize components: {e}")
            logger.error(f"Initialization error: {traceback.format_exc()}")
            return False
    
    def render_header(self):
        """Render the application header."""
        st.markdown('<h1 class="main-header">🧠 Hypercentaur</h1>', unsafe_allow_html=True)
        st.markdown("**ACEP-based Psychological Reasoning Assistant**")
        
        # Status indicators
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if hasattr(st.session_state, 'data_loader'):
                st.success("✅ Data Loaded")
            else:
                st.error("❌ Data Loading")
        
        with col2:
            if hasattr(st.session_state, 'reasoning_engine'):
                st.success("✅ Reasoning Engine")
            else:
                st.error("❌ Reasoning Engine")
        
        with col3:
            if st.session_state.get('db_connected', False):
                st.success("✅ Database")
            else:
                st.warning("⚠️ No Database")
        
        with col4:
            concept_count = 0
            if hasattr(st.session_state, 'processed_data'):
                concept_count = len(st.session_state.processed_data.get('concepts', []))
            st.info(f"📊 {concept_count} Concepts")
    
    def render_sidebar(self):
        """Render the sidebar with navigation and settings."""
        with st.sidebar:
            st.title("🎛️ Control Panel")
            
            # Navigation
            page = st.selectbox(
                "Navigation",
                ["🔍 Query Interface", "📊 Analytics", "🌐 Concept Network", "⚙️ Settings"],
                key="navigation"
            )
            
            st.divider()
            
            # Quick stats
            if hasattr(st.session_state, 'processed_data'):
                st.subheader("📈 Quick Stats")
                processed_data = st.session_state.processed_data
                
                st.metric("Examples", processed_data.get('num_examples', 0))
                st.metric("Concepts", len(processed_data.get('concepts', [])))
                
                # Get actual query count from reasoning engine
                query_count = 0
                if self.reasoning_engine and hasattr(self.reasoning_engine, 'reasoning_history'):
                    query_count = len(self.reasoning_engine.reasoning_history)
                elif 'query_counter' in st.session_state:
                    query_count = st.session_state.query_counter
                
                st.metric("Queries", query_count)
            
            st.divider()
            
            # Session info
            st.subheader("🔗 Session")
            st.text(f"ID: {self.session_id[:8]}...")
            st.text(f"Time: {datetime.now().strftime('%H:%M')}")
            
            # Settings
            st.subheader("⚙️ Settings")
            
            confidence_threshold = st.slider(
                "Confidence Threshold",
                min_value=0.0,
                max_value=1.0,
                value=0.5,
                step=0.1,
                help="Minimum confidence for concept activation"
            )
            
            max_reasoning_depth = st.slider(
                "Max Reasoning Depth",
                min_value=1,
                max_value=20,
                value=10,
                help="Maximum number of reasoning steps"
            )
            
            # Update engine settings
            if self.reasoning_engine:
                self.reasoning_engine.certainty_threshold = confidence_threshold
                self.reasoning_engine.max_reasoning_depth = max_reasoning_depth
            
            return page
    
    def render_query_interface(self):
        """Render the main query interface."""
        st.header("🔍 Query Interface")
        
        # Query input
        col1, col2 = st.columns([4, 1])
        
        with col1:
            # Use a key that changes to clear the input
            query_key = f"user_query_{st.session_state.get('query_counter', 0)}"
            query = st.text_input(
                "Enter your psychology question:",
                placeholder="e.g., What is cognitive dissonance?",
                key=query_key
            )
        
        with col2:
            query_button = st.button("🚀 Reason", type="primary")
        
        # Example queries
        st.subheader("💡 Example Queries")
        example_queries = [
            "What is cognitive dissonance?",
            "Explain classical conditioning",
            "What causes confirmation bias?",
            "How does operant conditioning work?",
            "What is the fundamental attribution error?"
        ]
        
        cols = st.columns(len(example_queries))
        for i, example in enumerate(example_queries):
            with cols[i]:
                if st.button(example, key=f"example_{i}"):
                    query = example
                    query_button = True
        
        # Process query
        if (query_button and query) or st.session_state.get('auto_submit', False):
            if query:  # Only process if there's actually a query
                self.process_query_with_animation(query)
        
        # Display recent queries
        if self.db_manager and st.session_state.get('db_connected', False):
            self.render_recent_queries()
    
    def process_query_with_animation(self, query: str):
        """Process a user query with typing animation and thinking phases."""
        if not self.reasoning_engine:
            st.error("❌ Reasoning engine not initialized")
            return
        
        # Initialize query counter if not exists
        if 'query_counter' not in st.session_state:
            st.session_state.query_counter = 0
        
        try:
            # Increment query counter
            st.session_state.query_counter += 1
            
            # Start timing
            start_time = time.time()
            
            # Create containers for animation
            thinking_container = st.empty()
            status_container = st.empty()
            
            # Phase 1: Thinking animation
            thinking_container.info("🧠 **Thinking...** Analyzing your psychology question")
            time.sleep(1)
            
            # Phase 2: Processing
            status_container.info("🔍 **Processing...** Extracting psychological concepts")
            time.sleep(1)
            
            # Phase 3: Reasoning
            thinking_container.info("⚡ **Reasoning...** Applying ACEP protocol")
            time.sleep(1)
            
            # Execute reasoning
            result = self.reasoning_engine.reason(query)
            
            # Phase 4: Finalizing
            status_container.info("✨ **Finalizing...** Generating response")
            time.sleep(0.5)
            
            # Clear thinking indicators
            thinking_container.empty()
            status_container.empty()
            
            # Phase 5: Typing animation for answer
            answer = result.get('answer', 'No answer generated')
            self.display_typing_animation(answer, result)
            
            # Store results in database
            if self.db_manager and st.session_state.get('db_connected', False):
                result['processing_time'] = time.time() - start_time
                result['user_session'] = self.session_id
                self.db_manager.store_query_result(result)
            
            # Display full results
            self.display_reasoning_result(result)
            
        except Exception as e:
            st.error(f"❌ Error processing query: {e}")
            logger.error(f"Query processing error: {traceback.format_exc()}")
    
    def display_typing_animation(self, answer: str, result: Dict[str, Any]):
        """Display typing animation for the answer."""
        # Create container for typing effect
        typing_container = st.empty()
        
        # Split answer into words
        words = answer.split(' ')
        displayed_text = ""
        
        # Typing effect
        for i, word in enumerate(words):
            displayed_text += word + " "
            
            # Display progress
            confidence = result.get('confidence', 0)
            confidence_class = (
                "confidence-high" if confidence > 0.7 
                else "confidence-medium" if confidence > 0.5 
                else "confidence-low"
            )
            
            typing_container.markdown(f"""
            <div class="reasoning-card {confidence_class}">
                <h4>🎯 Answer</h4>
                <p>{displayed_text}<span style="animation: blink 1s infinite;">|</span></p>
                <small>Confidence: {confidence:.2%}</small>
            </div>
            """, unsafe_allow_html=True)
            
            # Adjust speed based on word length
            if len(word) > 8:
                time.sleep(0.15)
            elif len(word) > 4:
                time.sleep(0.1)
            else:
                time.sleep(0.08)
        
        # Final display without cursor
        typing_container.markdown(f"""
        <div class="reasoning-card {confidence_class}">
            <h4>🎯 Answer</h4>
            <p>{displayed_text}</p>
            <small>Confidence: {confidence:.2%}</small>
        </div>
        """, unsafe_allow_html=True)
        
        time.sleep(0.5)  # Brief pause before showing full results
    
    def display_reasoning_result(self, result: Dict[str, Any]):
        """Display the reasoning result with visualizations."""
        
        # Tabs for detailed analysis
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Analysis", "🌐 Concepts", "🔗 Reasoning Chain", "📈 Metrics"])
        
        with tab1:
            self.render_analysis_tab(result)
        
        with tab2:
            self.render_concepts_tab(result)
        
        with tab3:
            self.render_reasoning_chain_tab(result)
        
        with tab4:
            self.render_metrics_tab(result)
    
    def render_analysis_tab(self, result: Dict[str, Any]):
        """Render the analysis tab."""
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📋 Summary")
            
            st.write(f"**Query:** {result.get('query', 'N/A')}")
            st.write(f"**Confidence:** {result.get('confidence', 0):.2%}")
            st.write(f"**Reasoning Steps:** {result.get('num_reasoning_steps', 0)}")
            st.write(f"**Key Concepts:** {len(result.get('key_concepts', []))}")
            
            # Quality metrics
            if result.get('reasoning_steps'):
                quality = calculate_reasoning_quality(result['reasoning_steps'])
                st.write(f"**Quality Score:** {quality['quality_score']:.2%}")
                st.write(f"**Coherence:** {quality['coherence']:.2%}")
        
        with col2:
            # Confidence breakdown
            if result.get('concept_certainties'):
                st.subheader("🎯 Concept Certainties")
                fig = self.visualizer.visualize_confidence_breakdown(result)
                st.plotly_chart(fig, use_container_width=True)
    
    def render_concepts_tab(self, result: Dict[str, Any]):
        """Render the concepts tab."""
        key_concepts = result.get('key_concepts', [])
        activated_concepts = result.get('activated_concepts', [])
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🔑 Key Concepts")
            if key_concepts:
                for concept in key_concepts:
                    formatted_name = format_concept_name(concept)
                    certainty = result.get('concept_certainties', {}).get(concept, 0)
                    st.markdown(f"""
                    <span class="concept-tag">{formatted_name} ({certainty:.2%})</span>
                    """, unsafe_allow_html=True)
            else:
                st.info("No key concepts identified")
        
        with col2:
            st.subheader("⚡ Activated Concepts")
            if activated_concepts:
                for concept in activated_concepts[:10]:  # Show top 10
                    formatted_name = format_concept_name(concept)
                    st.markdown(f"""
                    <span class="concept-tag">{formatted_name}</span>
                    """, unsafe_allow_html=True)
            else:
                st.info("No concepts activated")
        
        # Concept network visualization
        if hasattr(st.session_state, 'data_loader') and activated_concepts:
            st.subheader("🌐 Concept Network")
            concept_graph = st.session_state.data_loader.concept_graph
            
            if concept_graph and len(concept_graph.nodes()) > 0:
                fig = self.visualizer.visualize_concept_network(
                    concept_graph, activated_concepts
                )
                st.plotly_chart(fig, use_container_width=True)
    
    def render_reasoning_chain_tab(self, result: Dict[str, Any]):
        """Render the reasoning chain tab."""
        reasoning_steps = result.get('reasoning_steps', [])
        
        if not reasoning_steps:
            st.info("No reasoning steps recorded")
            return
        
        # Reasoning chain visualization
        st.subheader("🔗 Reasoning Chain")
        fig = self.visualizer.visualize_reasoning_chain(reasoning_steps)
        st.plotly_chart(fig, use_container_width=True)
        
        # Detailed steps
        st.subheader("📋 Detailed Steps")
        for i, step in enumerate(reasoning_steps):
            with st.expander(f"Step {i+1}: {step.get('reasoning_type', 'Unknown')}"):
                st.write(f"**Type:** {step.get('reasoning_type', 'Unknown')}")
                st.write(f"**Confidence:** {step.get('confidence', 0):.2%}")
                st.write(f"**Explanation:** {step.get('explanation', 'No explanation')}")
                
                # Input concepts
                input_concepts = step.get('input_concepts', [])
                if input_concepts:
                    st.write("**Input Concepts:**")
                    for concept in input_concepts:
                        if isinstance(concept, dict):
                            concept_name = concept.get('concept', 'Unknown')
                            activation = concept.get('activation_strength', 0)
                            st.write(f"  • {format_concept_name(concept_name)} (activation: {activation:.2f})")
                
                # Output concepts
                output_concepts = step.get('output_concepts', [])
                if output_concepts:
                    st.write("**Output Concepts:**")
                    for concept in output_concepts:
                        if isinstance(concept, dict):
                            concept_name = concept.get('concept', 'Unknown')
                            activation = concept.get('activation_strength', 0)
                            st.write(f"  • {format_concept_name(concept_name)} (activation: {activation:.2f})")
    
    def render_metrics_tab(self, result: Dict[str, Any]):
        """Render the metrics tab."""
        # Quality metrics
        if result.get('reasoning_steps'):
            quality_metrics = calculate_reasoning_quality(result['reasoning_steps'])
            
            st.subheader("📊 Quality Metrics")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Quality Score", f"{quality_metrics['quality_score']:.2%}")
                st.metric("Avg Confidence", f"{quality_metrics['avg_confidence']:.2%}")
            
            with col2:
                st.metric("Coherence", f"{quality_metrics['coherence']:.2%}")
                st.metric("Completeness", f"{quality_metrics['completeness']:.2%}")
            
            with col3:
                st.metric("Reasoning Steps", quality_metrics['num_steps'])
                st.metric("Unique Types", quality_metrics['unique_types'])
        
        # Concept summary
        activated_concepts = result.get('activated_concepts', [])
        if activated_concepts and hasattr(st.session_state, 'data_loader'):
            concept_graph = st.session_state.data_loader.concept_graph
            summary = create_concept_summary(activated_concepts, concept_graph)
            
            st.subheader("🧩 Concept Summary")
            
            if summary:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write(f"**Total Concepts:** {summary['total_concepts']}")
                    st.write(f"**Avg Importance:** {summary['avg_importance']:.2f}")
                
                with col2:
                    categories = summary.get('categories', {})
                    if categories:
                        st.write("**Categories:**")
                        for category, count in categories.items():
                            st.write(f"  • {category}: {count}")
    
    def render_recent_queries(self):
        """Render recent queries section."""
        st.subheader("📚 Recent Queries")
        
        try:
            recent_queries = self.db_manager.get_query_history(limit=5, user_session=self.session_id)
            
            if recent_queries:
                for query_data in recent_queries:
                    with st.expander(f"🔍 {query_data['query'][:50]}..."):
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.write(f"**Answer:** {query_data['answer'][:100]}...")
                            st.write(f"**Confidence:** {query_data['confidence']:.2%}")
                        
                        with col2:
                            st.write(f"**Time:** {query_data['timestamp'].strftime('%H:%M:%S')}")
                            st.write(f"**Steps:** {query_data.get('num_reasoning_steps', 0)}")
                        
                        if st.button(f"Reuse Query", key=f"reuse_{query_data['_id']}"):
                            # Clear the text field by incrementing counter
                            if 'query_counter' not in st.session_state:
                                st.session_state.query_counter = 0
                            st.session_state.query_counter += 1
                            # Process the query directly
                            self.process_query_with_animation(query_data['query'])
            else:
                st.info("No recent queries found")
                
        except Exception as e:
            st.error(f"Failed to load recent queries: {e}")
    
    def render_analytics_page(self):
        """Render the analytics page."""
        st.header("📊 Analytics Dashboard")
        
        if not self.db_manager or not st.session_state.get('db_connected', False):
            st.warning("⚠️ Database not connected. Analytics not available.")
            return
        
        try:
            # Get analytics data
            analytics_data = self.db_manager.get_concept_analytics()
            
            if analytics_data:
                # Display metrics dashboard
                self.visualizer.display_metrics_dashboard(analytics_data)
                
                # Session analytics
                st.subheader("👤 Session Analytics")
                session_stats = self.db_manager.get_session_stats(self.session_id)
                
                if session_stats:
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Session Queries", session_stats['query_count'])
                    
                    with col2:
                        st.metric("Avg Confidence", f"{session_stats['avg_confidence']:.2%}")
                    
                    with col3:
                        st.metric("Total Steps", session_stats['total_reasoning_steps'])
                    
                    # Top concepts in session
                    if session_stats.get('top_concepts'):
                        st.write("**Most Used Concepts in Session:**")
                        for concept in session_stats['top_concepts']:
                            st.write(f"• {format_concept_name(concept)}")
            else:
                st.info("No analytics data available yet. Start querying to see insights!")
                
        except Exception as e:
            st.error(f"Failed to load analytics: {e}")
    
    def render_concept_network_page(self):
        """Render the concept network exploration page."""
        st.header("🌐 Concept Network Explorer")
        
        if not hasattr(st.session_state, 'data_loader'):
            st.warning("⚠️ Data not loaded. Please return to the Query Interface.")
            return
        
        concept_graph = st.session_state.data_loader.concept_graph
        
        if not concept_graph or len(concept_graph.nodes()) == 0:
            st.warning("⚠️ No concept network available.")
            return
        
        # Network controls
        col1, col2, col3 = st.columns(3)
        
        with col1:
            show_all = st.checkbox("Show All Concepts", value=False)
        
        with col2:
            min_frequency = st.slider("Min Concept Frequency", 1, 10, 2)
        
        with col3:
            max_concepts = st.slider("Max Concepts to Show", 10, 100, 50)
        
        # Filter concepts
        filtered_concepts = []
        for node in concept_graph.nodes():
            node_data = concept_graph.nodes[node]
            frequency = node_data.get('frequency', 1)
            
            if frequency >= min_frequency:
                filtered_concepts.append(node)
        
        # Limit number of concepts
        if not show_all:
            filtered_concepts = filtered_concepts[:max_concepts]
        
        # Create subgraph
        if filtered_concepts:
            subgraph = concept_graph.subgraph(filtered_concepts)
            
            # Visualize network
            fig = self.visualizer.visualize_concept_network(subgraph)
            st.plotly_chart(fig, use_container_width=True)
            
            # Concept details
            st.subheader("🔍 Concept Details")
            
            selected_concept = st.selectbox(
                "Select a concept to explore:",
                filtered_concepts,
                format_func=format_concept_name
            )
            
            if selected_concept:
                context = st.session_state.data_loader.get_concept_context(selected_concept)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write(f"**Concept:** {format_concept_name(selected_concept)}")
                    st.write(f"**Frequency:** {context.get('frequency', 0)}")
                    st.write(f"**Category:** {context.get('category', 'Unknown')}")
                    st.write(f"**Centrality:** {context.get('centrality', 0):.3f}")
                
                with col2:
                    related_concepts = context.get('related_concepts', [])
                    if related_concepts:
                        st.write("**Related Concepts:**")
                        for related in related_concepts[:5]:
                            st.write(f"• {format_concept_name(related)}")
        else:
            st.info("No concepts match the current filters.")
    
    def render_settings_page(self):
        """Render the settings page."""
        st.header("⚙️ Settings")
        
        # System information
        st.subheader("🖥️ System Information")
        system_stats = get_system_stats()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write(f"**Platform:** {system_stats.get('platform', 'Unknown')}")
            st.write(f"**Python Version:** {system_stats.get('python_version', 'Unknown')}")
            st.write(f"**CPU Cores:** {system_stats.get('cpu_count', 'Unknown')}")
        
        with col2:
            memory_total = system_stats.get('memory_total', 0)
            memory_percent = system_stats.get('memory_percent', 0)
            st.write(f"**Memory Total:** {memory_total / (1024**3):.1f} GB")
            st.write(f"**Memory Usage:** {memory_percent:.1f}%")
        
        st.divider()
        
        # Application settings
        st.subheader("🎛️ Application Settings")
        
        # Data settings
        with st.expander("📊 Data Settings"):
            if st.button("🔄 Reload Dataset"):
                if 'data_loader' in st.session_state:
                    del st.session_state.data_loader
                if 'processed_data' in st.session_state:
                    del st.session_state.processed_data
                st.success("Dataset will be reloaded on next query")
        
        # Database settings
        with st.expander("💾 Database Settings"):
            st.write(f"**Connection Status:** {'✅ Connected' if st.session_state.get('db_connected', False) else '❌ Disconnected'}")
            
            if st.button("🔄 Reconnect Database"):
                if 'db_manager' in st.session_state:
                    del st.session_state.db_manager
                st.success("Database will reconnect on next operation")
        
        # Export/Import
        with st.expander("📤 Export/Import"):
            if self.db_manager and st.session_state.get('db_connected', False):
                if st.button("📤 Export Session Data"):
                    try:
                        session_data = self.db_manager.get_query_history(
                            limit=100, user_session=self.session_id
                        )
                        
                        if session_data:
                            df = pd.DataFrame(session_data)
                            csv = df.to_csv(index=False)
                            
                            st.download_button(
                                label="📥 Download CSV",
                                data=csv,
                                file_name=f"hypercentaur_session_{self.session_id[:8]}.csv",
                                mime="text/csv"
                            )
                        else:
                            st.info("No session data to export")
                    except Exception as e:
                        st.error(f"Export failed: {e}")
            else:
                st.info("Database not connected")
        
        st.divider()
        
        # About
        st.subheader("ℹ️ About Hypercentaur")
        st.markdown("""
        **Hypercentaur** is an ACEP-based psychological reasoning assistant that provides:
        
        - 🧠 **Human-like reasoning** based on psychological principles
        - ⚡ **Instant training** without GPU requirements
        - 🔍 **Explainable AI** with transparent reasoning chains
        - 📊 **Concept visualization** and relationship mapping
        - 💾 **Persistent storage** of queries and insights
        
        Built with:
        - **ACEP (AI Conceptual Exchange Protocol)** for structured reasoning
        - **Psych-101 dataset** for psychological knowledge grounding
        - **NetworkX** for concept relationship modeling
        - **Streamlit** for interactive user interface
        - **MongoDB** for data persistence
        """)
    
    def run(self):
        """Run the main application."""
        # Initialize components
        if not self.initialize_components():
            st.stop()
        
        # Render header
        self.render_header()
        
        # Render sidebar and get current page
        current_page = self.render_sidebar()
        
        # Route to appropriate page
        if current_page == "🔍 Query Interface":
            self.render_query_interface()
        elif current_page == "📊 Analytics":
            self.render_analytics_page()
        elif current_page == "🌐 Concept Network":
            self.render_concept_network_page()
        elif current_page == "⚙️ Settings":
            self.render_settings_page()

def main():
    """Main entry point."""
    try:
        app = HypercentaurApp()
        app.run()
    except Exception as e:
        st.error(f"❌ Application error: {e}")
        logger.error(f"Application error: {traceback.format_exc()}")
        
        # Show error details in expander
        with st.expander("🔍 Error Details"):
            st.code(traceback.format_exc())

if __name__ == "__main__":
    main()