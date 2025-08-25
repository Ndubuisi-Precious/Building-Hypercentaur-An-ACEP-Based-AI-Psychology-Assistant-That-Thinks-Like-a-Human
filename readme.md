# 🧠 Hypercentaur - ACEP Psychological Reasoning Assistant

**Hypercentaur** is an innovative AI reasoning system that uses the **ACEP (AI Conceptual Exchange Protocol)** to provide human-like psychological reasoning without requiring expensive LLM training. Built on the Psych-101 dataset, it offers instant training, explainable reasoning chains, and interactive concept visualization.

## 🌟 Key Features

- **🧠 Human-like Reasoning**: Uses ACEP protocol for structured conceptual reasoning
- **⚡ Instant Training**: No GPU or extensive training time required
- **🔍 Explainable AI**: Transparent reasoning chains with confidence tracking
- **📊 Interactive Visualization**: Concept networks and reasoning flow charts
- **💾 Persistent Storage**: MongoDB integration for query history and analytics
- **🌐 Web Interface**: Beautiful Streamlit UI for easy interaction

## 🏗️ Architecture

### ACEP (AI Conceptual Exchange Protocol)
- **Conceptual Tokens**: Structured representation of psychological concepts
- **Reasoning Types**: Direct lookup, conceptual bridging, analogical, causal, hierarchical, synthesis
- **Certainty Tracking**: Confidence scores for each reasoning step
- **Bounded Reasoning**: Controlled reasoning depth and complexity

### Data Foundation
- **Psych-101 Dataset**: Psychology questions and answers from Hugging Face
- **Concept Graph**: NetworkX-based relationship modeling
- **Semantic Embeddings**: Sentence transformers for concept similarity

## 📁 Project Structure

```
hypercentaur/
│
├── requirements.txt          # Dependencies
├── README.md                # This file
├── app.py                   # Main Streamlit application
│
├── data_loader.py           # Psych-101 dataset processing
├── reasoning_engine.py      # ACEP reasoning implementation
├── db_manager.py           # MongoDB integration
├── visualizer.py           # Reasoning and concept visualizations
└── utils.py                # Helper functions and utilities
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- MongoDB (optional, for persistence)

### Installation

1. **Clone or create the project directory:**
```bash
mkdir hypercentaur && cd hypercentaur
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Download spaCy model (optional):**
```bash
python -m spacy download en_core_web_sm
```

4. **Start MongoDB (optional):**
```bash
# Using Docker
docker run -d -p 27017:27017 mongo

# Or install MongoDB locally
```

### Running the Application

```bash
streamlit run app.py
```

The application will be available at `http://localhost:8501`

## 🎯 Usage Examples

### Basic Query
```
Query: "What is cognitive dissonance?"

Response: 
- Answer: Cognitive dissonance is the mental discomfort experienced when holding contradictory beliefs...
- Confidence: 85%
- Key Concepts: cognitive_dissonance, mental_conflict, belief_inconsistency
- Reasoning Steps: 3 (Direct Lookup → Conceptual Bridge → Synthesis)
```

### Advanced Query
```
Query: "How does confirmation bias affect decision making?"

Reasoning Chain:
1. Direct Lookup: confirmation_bias (confidence: 0.9)
2. Conceptual Bridge: decision_making, cognitive_bias (confidence: 0.8)  
3. Causal Reasoning: bias_influence, judgment_error (confidence: 0.75)
4. Synthesis: comprehensive_answer (confidence: 0.82)
```

## 🧩 Core Components

### Data Loader (`data_loader.py`)
- Loads Psych-101 dataset from Hugging Face
- Extracts psychological concepts using NLP
- Builds concept relationship graph
- Generates semantic embeddings

### Reasoning Engine (`reasoning_engine.py`)
- Implements ACEP protocol
- Parses queries into conceptual tokens
- Performs multi-step reasoning
- Tracks confidence and certainty
- Generates explainable answers

### Database Manager (`db_manager.py`)
- MongoDB integration
- Stores queries, responses, reasoning chains
- Provides analytics and search capabilities
- Manages user sessions

### Visualizer (`visualizer.py`)
- Interactive reasoning chain visualization
- Concept network graphs
- Confidence breakdown charts
- Analytics dashboards

## 🎛️ Configuration

### Reasoning Engine Settings
```python
# Confidence thresholds
concept_activation_threshold = 0.3
certainty_threshold = 0.5
max_reasoning_depth = 10

# Available reasoning types
DIRECT_LOOKUP = "direct_lookup"
CONCEPTUAL_BRIDGE = "conceptual_bridge" 
ANALOGICAL = "analogical"
CAUSAL = "causal"
HIERARCHICAL = "hierarchical"
SYNTHESIS = "synthesis"
```

### Database Configuration
```python
# MongoDB settings
connection_string = "mongodb://localhost:27017/"
database_name = "hypercentaur"

# Collections
collections = {
    'queries': 'user_queries',
    'reasoning_chains': 'reasoning_chains', 
    'concepts': 'concept_knowledge',
    'sessions': 'user_sessions'
}
```

## 📊 Analytics & Insights

### Query Analytics
- Total queries processed
- Average confidence scores
- Most used concepts
- Reasoning type distribution

### Concept Analytics  
- Concept frequency and importance
- Category distribution
- Relationship strength
- Centrality measures

### Session Analytics
- Per-session query count
- User reasoning patterns
- Concept exploration paths

## 🔧 Advanced Usage

### Custom Concept Extraction
```python
# Add domain-specific patterns
psych_patterns = [
    r'\b(?:cognitive|behavioral|social|developmental)\s+\w+',
    r'\b\w+(?:\s+(?:bias|effect|theory|disorder))\b',
    r'\b(?:freud|jung|skinner|pavlov)\w*\b'
]
```

### Custom Reasoning Types
```python
class CustomReasoningType(ReasoningType):
    TEMPORAL = "temporal"
    STATISTICAL = "statistical" 
    COMPARATIVE = "comparative"
```

### Extending the Knowledge Base
```python
# Add custom psychology data
custom_data = [
    {
        "question": "Your custom question",
        "answer": "Your expert answer", 
        "category": "your_category",
        "concepts": ["concept1", "concept2"]
    }
]
```

## 🧪 Testing

### Example Test Queries
1. **"What is cognitive dissonance?"** - Basic concept lookup
2. **"Explain classical conditioning"** - Learning theory
3. **"What causes confirmation bias?"** - Cognitive bias analysis  
4. **"How does operant conditioning work?"** - Mechanism explanation
5. **"What is the fundamental attribution error?"** - Social psychology

### Expected Outputs
- Confident answers (>70%) for core psychology concepts
- Multi-step reasoning chains for complex queries
- Relevant concept activation and relationships
- Explainable reasoning with evidence sources

## 🔮 Future Enhancements

### Planned Features
- **Multi-modal Reasoning**: Images, diagrams, case studies
- **Real-time Learning**: Continuous knowledge updates
- **Expert Integration**: Psychology professional feedback loop
- **Advanced Visualizations**: 3D concept spaces, temporal reasoning
- **API Endpoints**: RESTful API for external integration

### Research Directions
- **ACEP Protocol Evolution**: Enhanced reasoning types and mechanisms
- **Knowledge Graph Expansion**: Broader psychological domains
- **Uncertainty Quantification**: Improved confidence calibration
- **Human-AI Collaboration**: Interactive reasoning refinement

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📜 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **Psych-101 Dataset**: Marcel Binz and team for the psychology Q&A dataset
- **Hugging Face**: For dataset hosting and transformers library
- **NetworkX**: For graph-based concept modeling
- **Streamlit**: For the interactive web interface
- **MongoDB**: For robust data persistence

## 📧 Contact

For questions, suggestions, or collaboration opportunities:

- **GitHub Issues**: [Project Issues](https://github.com/yourname/hypercentaur/issues)
- **Email**: your.email@domain.com
- **Twitter**: @yourhandle

---

**Hypercentaur** - Making psychological reasoning transparent, explainable, and accessible through innovative AI protocols. 🧠✨