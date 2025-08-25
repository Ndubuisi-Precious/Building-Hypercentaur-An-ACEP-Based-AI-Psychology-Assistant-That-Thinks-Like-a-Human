"""
Simple, reliable data loader for psychology concepts.
"""

import pandas as pd
import numpy as np
import networkx as nx
from typing import Dict, List, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimplePsychDataLoader:
    """Simple, working data loader for psychology concepts."""
    
    def __init__(self):
        self.processed_data = None
        self.concept_graph = nx.Graph()
        self.concept_embeddings = {}
        
    def load_dataset(self) -> pd.DataFrame:
        """Load a reliable psychology dataset."""
        logger.info("Loading simple psychology dataset...")
        
        # Create a comprehensive, working dataset
        psychology_data = [
            {
                "question": "What is cognitive dissonance?",
                "answer": "Cognitive dissonance is the mental discomfort experienced when holding contradictory beliefs or when behavior conflicts with beliefs.",
                "category": "cognitive_psychology",
                "concepts": ["cognitive_dissonance", "mental_conflict", "belief_inconsistency"]
            },
            {
                "question": "What is classical conditioning?",
                "answer": "Classical conditioning is a learning process where a neutral stimulus becomes associated with a meaningful stimulus to trigger a response.",
                "category": "learning_theory", 
                "concepts": ["classical_conditioning", "pavlov", "conditioned_response"]
            },
            {
                "question": "What is operant conditioning?",
                "answer": "Operant conditioning is learning through reinforcement and punishment to modify behavior based on consequences.",
                "category": "learning_theory",
                "concepts": ["operant_conditioning", "skinner", "reinforcement", "punishment"]
            },
            {
                "question": "What is confirmation bias?",
                "answer": "Confirmation bias is the tendency to search for and interpret information that confirms pre-existing beliefs.",
                "category": "cognitive_bias",
                "concepts": ["confirmation_bias", "selective_attention", "biased_reasoning"]
            },
            {
                "question": "What is the fundamental attribution error?",
                "answer": "The fundamental attribution error is attributing others' behavior to personality rather than situational factors.",
                "category": "social_psychology",
                "concepts": ["fundamental_attribution_error", "attribution_bias", "social_cognition"]
            },
            {
                "question": "How to heal from heartbreak?",
                "answer": "Healing from heartbreak involves accepting emotions, grieving, maintaining social connections, and focusing on self-care and personal growth.",
                "category": "relationship_psychology",
                "concepts": ["heartbreak", "emotional_healing", "relationship_recovery", "grief"]
            },
            {
                "question": "How to move on after a breakup?",
                "answer": "Moving on after a breakup requires time, self-compassion, removing triggers, rebuilding social support, and engaging in new activities.",
                "category": "relationship_psychology",
                "concepts": ["moving_on", "breakup_recovery", "emotional_resilience", "healing_process"]
            },
            {
                "question": "How to know if someone loves you?",
                "answer": "Signs of love include consistent care, genuine interest in your feelings, making time for you, physical affection, and including you in future plans.",
                "category": "relationship_psychology",
                "concepts": ["love_signs", "relationship_indicators", "emotional_connection", "romantic_attachment"]
            },
            {
                "question": "How does drug addiction affect me?",
                "answer": "Drug addiction affects brain chemistry, decision-making, impulse control, and causes mood swings, anxiety, and relationship problems.",
                "category": "addiction_psychology",
                "concepts": ["drug_addiction", "substance_abuse", "brain_chemistry", "addiction_effects"]
            },
            {
                "question": "How to manage anxiety?",
                "answer": "Anxiety management involves breathing exercises, mindfulness, regular exercise, adequate sleep, and professional therapy when needed.",
                "category": "mental_health",
                "concepts": ["anxiety_management", "coping_strategies", "stress_reduction", "mental_health"]
            },
            {
                "question": "What are signs of depression?",
                "answer": "Depression signs include persistent sadness, loss of interest, fatigue, sleep changes, feelings of worthlessness, and difficulty concentrating.",
                "category": "mental_health",
                "concepts": ["depression_symptoms", "mood_disorder", "mental_health_signs", "clinical_depression"]
            },
            {
                "question": "How to build self-esteem?",
                "answer": "Building self-esteem requires challenging negative self-talk, setting achievable goals, practicing self-compassion, and focusing on strengths.",
                "category": "self_psychology",
                "concepts": ["self_esteem", "self_worth", "self_confidence", "personal_value"]
            },
            {
                "question": "How to manage stress?",
                "answer": "Stress management involves identifying triggers, developing coping mechanisms, exercise, relaxation practices, and maintaining work-life balance.",
                "category": "mental_health",
                "concepts": ["stress_management", "coping_mechanisms", "resilience", "well_being"]
            },
            {
                "question": "How to overcome trust issues?",
                "answer": "Overcoming trust issues involves understanding root causes, practicing gradual vulnerability, communication skills, and often professional therapy.",
                "category": "relationship_psychology",
                "concepts": ["trust_issues", "relationship_trust", "emotional_vulnerability", "betrayal_trauma"]
            },
            {
                "question": "How to deal with loneliness?",
                "answer": "Combating loneliness involves building meaningful connections, joining communities, volunteering, and addressing underlying social anxiety.",
                "category": "social_psychology",
                "concepts": ["loneliness", "social_isolation", "social_connection", "emotional_well_being"]
            }
        ]
        
        return pd.DataFrame(psychology_data)
    
    def extract_concepts(self, text: str) -> List[str]:
        """Extract psychology concepts from text."""
        if not text:
            return []
        
        # Simple keyword extraction
        psychology_terms = [
            "cognitive dissonance", "classical conditioning", "operant conditioning",
            "confirmation bias", "fundamental attribution error", "social facilitation",
            "groupthink", "anchoring bias", "availability heuristic", "working memory",
            "attachment theory", "big five", "self efficacy", "reinforcement", "punishment",
            "pavlov", "skinner", "bandura", "bowlby", "heartbreak", "heal", "moving on",
            "breakup", "love signs", "drug addiction", "substance abuse", "anxiety",
            "depression", "self esteem", "stress", "trust issues", "loneliness",
            "mental health", "emotional healing", "relationship recovery"
        ]
        
        text_lower = text.lower()
        found_concepts = []
        
        for term in psychology_terms:
            if term in text_lower:
                found_concepts.append(term.replace(" ", "_"))
        
        return found_concepts
    
    def build_concept_graph(self, data: pd.DataFrame) -> nx.Graph:
        """Build a simple concept graph."""
        logger.info("Building simple concept graph...")
        
        self.concept_graph = nx.Graph()
        
        for _, row in data.iterrows():
            concepts = row.get('concepts', [])
            category = row.get('category', 'general')
            
            # Add nodes
            for concept in concepts:
                if not self.concept_graph.has_node(concept):
                    self.concept_graph.add_node(concept, frequency=1, category=category)
                else:
                    self.concept_graph.nodes[concept]['frequency'] += 1
            
            # Add edges between concepts in same example
            for i, concept1 in enumerate(concepts):
                for concept2 in concepts[i+1:]:
                    if self.concept_graph.has_edge(concept1, concept2):
                        self.concept_graph[concept1][concept2]['weight'] += 1
                    else:
                        self.concept_graph.add_edge(concept1, concept2, weight=1)
        
        logger.info(f"Built concept graph with {len(self.concept_graph.nodes)} nodes")
        return self.concept_graph
    
    def generate_concept_embeddings(self, concepts: List[str]) -> Dict[str, Any]:
        """Generate simple concept embeddings."""
        logger.info("Generating simple concept embeddings...")
        
        # Simple embedding simulation
        self.concept_embeddings = {
            concept: np.random.rand(384) for concept in concepts  # Mock embeddings
        }
        
        return self.concept_embeddings
    
    def get_concept_context(self, concept: str) -> Dict[str, Any]:
        """Get context for a concept."""
        if not self.concept_graph.has_node(concept):
            return {
                'concept': concept,
                'frequency': 0,
                'category': 'unknown',
                'related_concepts': [],
                'centrality': 0
            }
        
        node_data = self.concept_graph.nodes[concept]
        neighbors = list(self.concept_graph.neighbors(concept))
        
        return {
            'concept': concept,
            'frequency': node_data.get('frequency', 0),
            'category': node_data.get('category', 'unknown'),
            'related_concepts': neighbors,
            'centrality': 0.5  # Simple mock value
        }
    
    def process_dataset(self) -> Dict[str, Any]:
        """Process the dataset reliably."""
        logger.info("Processing simple psychology dataset...")
        
        # Load data
        data = self.load_dataset()
        
        # Build graph
        concept_graph = self.build_concept_graph(data)
        
        # Generate embeddings
        concepts = list(concept_graph.nodes()) if concept_graph.nodes() else []
        concept_embeddings = self.generate_concept_embeddings(concepts)
        
        # Store processed data
        self.processed_data = {
            'raw_data': data,
            'concept_graph': concept_graph,
            'concept_embeddings': concept_embeddings,
            'concepts': concepts,
            'num_examples': len(data)
        }
        
        logger.info("Simple dataset processing complete!")
        return self.processed_data

# Aliases for compatibility
Psych101DataLoader = SimplePsychDataLoader
EnhancedPsych101DataLoader = SimplePsychDataLoader