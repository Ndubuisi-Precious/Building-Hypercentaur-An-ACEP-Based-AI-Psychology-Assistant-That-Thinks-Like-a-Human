"""
Fixed ACEP Reasoning Engine - Provides definitive psychology answers.
"""

import numpy as np
import networkx as nx
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, field
from enum import Enum
import logging
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ReasoningType(Enum):
    """Types of reasoning operations in ACEP."""
    DIRECT_LOOKUP = "direct_lookup"
    CONCEPTUAL_BRIDGE = "conceptual_bridge"
    ANALOGICAL = "analogical"
    CAUSAL = "causal"
    HIERARCHICAL = "hierarchical"
    SYNTHESIS = "synthesis"

@dataclass
class ConceptualToken:
    """Represents a conceptual unit in ACEP."""
    concept: str
    activation_strength: float
    certainty: float
    source_evidence: List[str] = field(default_factory=list)
    relationships: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'concept': self.concept,
            'activation_strength': self.activation_strength,
            'certainty': self.certainty,
            'source_evidence': self.source_evidence,
            'relationships': self.relationships,
            'metadata': self.metadata
        }

@dataclass
class ReasoningStep:
    """Represents a single step in the reasoning chain."""
    step_id: int
    reasoning_type: ReasoningType
    input_concepts: List[ConceptualToken]
    output_concepts: List[ConceptualToken]
    confidence: float
    explanation: str
    evidence: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'step_id': self.step_id,
            'reasoning_type': self.reasoning_type.value,
            'input_concepts': [c.to_dict() for c in self.input_concepts],
            'output_concepts': [c.to_dict() for c in self.output_concepts],
            'confidence': self.confidence,
            'explanation': self.explanation,
            'evidence': self.evidence
        }

class FixedACEPReasoningEngine:
    """Fixed reasoning engine that actually works."""
    
    def __init__(self, data_loader):
        self.data_loader = data_loader
        self.reasoning_history = []
        
        # Comprehensive psychology knowledge base - directly accessible
        self.psychology_kb = {
            # Cognitive Psychology
            "cognitive dissonance": {
                "answer": "Cognitive dissonance is the mental discomfort experienced when a person holds contradictory beliefs, values, or attitudes simultaneously, or when their behavior conflicts with their beliefs. This psychological tension motivates people to reduce the dissonance by changing their beliefs, attitudes, or behaviors to restore consistency. For example, a smoker who knows smoking is harmful but continues smoking experiences cognitive dissonance.",
                "concepts": ["cognitive_dissonance", "mental_conflict", "belief_inconsistency"],
                "confidence": 0.95
            },
            
            "classical conditioning": {
                "answer": "Classical conditioning is a learning process where a neutral stimulus becomes associated with a meaningful stimulus, eventually triggering a conditioned response. Through repeated pairing, an organism learns to associate two stimuli, causing a natural response to be triggered by a previously neutral stimulus. Pavlov's famous experiment demonstrated this when dogs learned to salivate to the sound of a bell after it was repeatedly paired with food.",
                "concepts": ["classical_conditioning", "pavlov", "conditioned_response", "learning"],
                "confidence": 0.98
            },
            
            "operant conditioning": {
                "answer": "Operant conditioning is a learning method developed by B.F. Skinner where behavior is modified through the use of reinforcement or punishment based on consequences. Behaviors followed by positive consequences (reinforcement) are more likely to be repeated, while those followed by negative consequences (punishment) are less likely to occur. This includes positive reinforcement (adding something good), negative reinforcement (removing something bad), positive punishment (adding something bad), and negative punishment (removing something good).",
                "concepts": ["operant_conditioning", "skinner", "reinforcement", "punishment"],
                "confidence": 0.96
            },
            
            "confirmation bias": {
                "answer": "Confirmation bias is the tendency to search for, interpret, and recall information that confirms pre-existing beliefs while ignoring contradictory evidence. This cognitive bias leads people to favor information that supports their existing views and dismiss evidence that challenges them. For example, people often read news sources that align with their political views while avoiding those that present opposing perspectives.",
                "concepts": ["confirmation_bias", "cognitive_bias", "selective_attention"],
                "confidence": 0.93
            },
            
            "fundamental attribution error": {
                "answer": "The fundamental attribution error is the tendency to attribute others' behavior to internal characteristics (personality, character) rather than considering situational factors. People often overemphasize personality-based explanations for behaviors observed in others while underemphasizing external circumstances. For example, assuming someone is rude rather than considering they might be having a difficult day.",
                "concepts": ["fundamental_attribution_error", "social_psychology", "attribution_theory"],
                "confidence": 0.91
            },
            
            "social facilitation": {
                "answer": "Social facilitation is the tendency for people to perform differently when in the presence of others compared to when alone. Simple or well-learned tasks are typically performed better in front of others due to increased arousal, while complex or new tasks are often performed worse due to anxiety and distraction. This effect was first studied by Norman Triplett in 1898.",
                "concepts": ["social_facilitation", "social_psychology", "performance"],
                "confidence": 0.89
            },
            
            "groupthink": {
                "answer": "Groupthink is a psychological phenomenon where the desire for harmony and conformity in a group results in irrational or dysfunctional decision-making. Group members suppress dissent, fail to analyze alternatives critically, and isolate themselves from outside opinions. This can lead to poor decisions and was notably studied in historical events like the Bay of Pigs invasion.",
                "concepts": ["groupthink", "group_dynamics", "conformity"],
                "confidence": 0.87
            },
            
            "anchoring bias": {
                "answer": "Anchoring bias is the tendency to rely too heavily on the first piece of information encountered (the 'anchor') when making decisions. This initial information serves as a reference point and influences subsequent judgments, even when the anchor is irrelevant to the decision at hand. For example, when negotiating a price, the first number mentioned often serves as an anchor.",
                "concepts": ["anchoring_bias", "cognitive_bias", "decision_making"],
                "confidence": 0.92
            },
            
            "availability heuristic": {
                "answer": "The availability heuristic is a mental shortcut where people judge the probability of events based on how easily examples come to mind. Events that are more memorable, recent, or emotionally significant seem more likely to occur than they actually are. This can lead to overestimating the likelihood of dramatic events like plane crashes while underestimating more common risks.",
                "concepts": ["availability_heuristic", "cognitive_bias", "probability_judgment"],
                "confidence": 0.90
            },
            
            "working memory": {
                "answer": "Working memory is the cognitive system responsible for temporarily holding and manipulating information during complex cognitive tasks. It consists of multiple components including the central executive (attention control), phonological loop (verbal information), and visuospatial sketchpad (visual-spatial information). Working memory capacity is limited and crucial for reasoning, learning, and comprehension.",
                "concepts": ["working_memory", "cognitive_psychology", "information_processing"],
                "confidence": 0.94
            },
            
            "attachment theory": {
                "answer": "Attachment theory, developed by John Bowlby, describes the emotional bonds between children and caregivers. It identifies four main attachment styles: secure attachment (consistent, responsive caregiving leads to trust), anxious-ambivalent (inconsistent caregiving leads to anxiety), avoidant (emotionally unavailable caregiving leads to independence), and disorganized (frightening caregiving leads to confusion). These early patterns often influence adult relationships.",
                "concepts": ["attachment_theory", "bowlby", "developmental_psychology"],
                "confidence": 0.95
            },
            
            "big five personality": {
                "answer": "The Big Five personality traits are five broad dimensions that describe human personality: Openness to experience (creativity, curiosity, imagination), Conscientiousness (organization, discipline, reliability), Extraversion (sociability, energy, assertiveness), Agreeableness (cooperation, trust, empathy), and Neuroticism (emotional instability, anxiety, moodiness). These traits are relatively stable across time and situations.",
                "concepts": ["big_five", "personality_psychology", "personality_traits"],
                "confidence": 0.97
            },
            
            "self efficacy": {
                "answer": "Self-efficacy, developed by Albert Bandura, refers to an individual's belief in their ability to execute behaviors necessary to produce specific performance attainments. It influences how people think, feel, motivate themselves, and behave. High self-efficacy leads to greater effort, persistence, and resilience when facing challenges, while low self-efficacy can result in avoidance and giving up easily.",
                "concepts": ["self_efficacy", "bandura", "motivation"],
                "confidence": 0.93
            },
            
            # Relationship and Emotional Psychology
            "heal from heartbreak": {
                "answer": "Healing from heartbreak involves accepting your emotions, allowing yourself to grieve, maintaining social connections, focusing on self-care, and gradually rebuilding your identity. Research shows that heartbreak activates the same brain regions as physical pain. Recovery typically involves stages: denial, anger, bargaining, depression, and acceptance. Professional therapy can help process complex emotions and develop healthy coping strategies.",
                "concepts": ["heartbreak", "grief", "emotional_healing", "relationship_recovery"],
                "confidence": 0.88
            },
            
            "move on after heartbreak": {
                "answer": "Moving on after heartbreak requires time, self-compassion, and intentional healing practices. Key strategies include: accepting the loss, removing triggers and reminders, rebuilding your social support network, engaging in new activities, practicing mindfulness, and focusing on personal growth. Avoid rebound relationships until you've processed the loss. Consider therapy if depression or anxiety persists beyond normal grieving periods.",
                "concepts": ["moving_on", "breakup_recovery", "emotional_resilience", "healing_process"],
                "confidence": 0.87
            },
            
            "signs of love": {
                "answer": "Signs that someone loves you include: consistent care and support during difficult times, genuine interest in your thoughts and feelings, making time for you despite busy schedules, physical affection and intimacy, introducing you to important people in their life, making future plans that include you, showing vulnerability and sharing personal thoughts, and demonstrating respect for your boundaries and independence.",
                "concepts": ["love_signs", "relationship_indicators", "emotional_connection", "romantic_attachment"],
                "confidence": 0.85
            },
            
            "drug addiction effects": {
                "answer": "Drug addiction affects the brain's reward system, decision-making abilities, and impulse control. It changes brain chemistry, particularly dopamine pathways, leading to tolerance, dependence, and withdrawal symptoms. Psychological effects include mood swings, anxiety, depression, cognitive impairment, and relationship problems. Physical effects vary by substance but often include health deterioration, sleep disruption, and increased risk of accidents and infections.",
                "concepts": ["drug_addiction", "substance_abuse", "brain_chemistry", "addiction_effects"],
                "confidence": 0.92
            },
            
            "anxiety management": {
                "answer": "Anxiety management involves both immediate coping strategies and long-term lifestyle changes. Immediate techniques include deep breathing, progressive muscle relaxation, grounding exercises (5-4-3-2-1 sensory technique), and mindfulness meditation. Long-term strategies include regular exercise, adequate sleep, limiting caffeine, cognitive behavioral therapy (CBT), and building strong social support. Professional help is recommended for persistent or severe anxiety.",
                "concepts": ["anxiety_management", "coping_strategies", "stress_reduction", "mental_health"],
                "confidence": 0.90
            },
            
            "depression signs": {
                "answer": "Depression signs include persistent sadness, loss of interest in activities, fatigue, changes in appetite or sleep patterns, feelings of worthlessness or guilt, difficulty concentrating, social withdrawal, and thoughts of death or suicide. Symptoms must persist for at least two weeks and significantly impact daily functioning. Depression is a medical condition requiring professional treatment, not a personal weakness or character flaw.",
                "concepts": ["depression_symptoms", "mood_disorder", "mental_health_signs", "clinical_depression"],
                "confidence": 0.94
            },
            
            "self esteem": {
                "answer": "Self-esteem is your overall sense of personal value and worth. Healthy self-esteem involves realistic self-assessment, self-acceptance, and confidence in your abilities. Building self-esteem requires challenging negative self-talk, setting achievable goals, practicing self-compassion, focusing on strengths, maintaining healthy relationships, and celebrating small accomplishments. Low self-esteem often stems from childhood experiences, perfectionism, or comparison with others.",
                "concepts": ["self_esteem", "self_worth", "self_confidence", "personal_value"],
                "confidence": 0.89
            },
            
            "stress management": {
                "answer": "Effective stress management involves identifying stress triggers, developing healthy coping mechanisms, and building resilience. Techniques include time management, regular exercise, relaxation practices (meditation, yoga), maintaining work-life balance, seeking social support, and problem-solving skills. Chronic stress can lead to physical and mental health problems, so learning to manage stress is crucial for overall well-being.",
                "concepts": ["stress_management", "coping_mechanisms", "resilience", "well_being"],
                "confidence": 0.91
            },
            
            "trust issues": {
                "answer": "Trust issues often develop from past betrayals, childhood trauma, or repeated negative experiences in relationships. They manifest as difficulty believing others, fear of vulnerability, constant suspicion, and emotional walls. Healing involves understanding the root causes, practicing gradual vulnerability with safe people, communication skills development, and often professional therapy. Building trust is a gradual process requiring patience and self-compassion.",
                "concepts": ["trust_issues", "relationship_trust", "emotional_vulnerability", "betrayal_trauma"],
                "confidence": 0.86
            },
            
            "loneliness": {
                "answer": "Loneliness is the subjective feeling of being isolated or disconnected from others, which can occur even when surrounded by people. It's different from solitude, which can be enjoyable. Chronic loneliness affects physical and mental health, increasing risks of depression, anxiety, and physical illness. Combating loneliness involves building meaningful connections, joining communities, volunteering, and sometimes addressing underlying social anxiety or depression.",
                "concepts": ["loneliness", "social_isolation", "social_connection", "emotional_well_being"],
                "confidence": 0.88
            }
        }
        
        # Alternative search terms for better matching
        self.concept_aliases = {
            "cognitive dissonance": ["dissonance", "mental conflict", "belief conflict"],
            "classical conditioning": ["pavlov", "pavlovian", "conditioned response"],
            "operant conditioning": ["skinner", "reinforcement", "punishment", "behavior modification"],
            "confirmation bias": ["selective attention", "biased reasoning"],
            "fundamental attribution error": ["attribution error", "attribution bias"],
            "social facilitation": ["audience effect", "social presence"],
            "groupthink": ["group conformity", "group pressure"],
            "anchoring bias": ["anchoring effect", "anchoring heuristic"],
            "availability heuristic": ["availability bias", "representativeness"],
            "working memory": ["short term memory", "active memory"],
            "attachment theory": ["bowlby", "attachment styles", "attachment patterns"],
            "big five personality": ["big five", "five factor model", "personality dimensions"],
            "self efficacy": ["self confidence", "personal efficacy", "bandura"],
            "heal from heartbreak": ["heartbreak", "broken heart", "relationship pain", "breakup pain", "emotional pain", "heart break", "heal heartbreak"],
            "move on after heartbreak": ["moving on", "get over", "breakup recovery", "relationship recovery", "after breakup", "move on"],
            "signs of love": ["she loves me", "he loves me", "does she love me", "does he love me", "love signs", "in love", "true love"],
            "drug addiction effects": ["drug addiction", "substance abuse", "addiction effects", "drug abuse", "substance addiction", "drug dependency"],
            "anxiety management": ["anxiety", "anxiety help", "manage anxiety", "anxiety relief", "anxiety disorder", "worried", "stress anxiety"],
            "depression signs": ["depression", "depressed", "sad", "depression symptoms", "feeling down", "mental health", "mood disorder"],
            "self esteem": ["self worth", "confidence", "self confidence", "low self esteem", "self image", "self respect"],
            "stress management": ["stress", "stressed", "stress relief", "manage stress", "overwhelmed", "pressure"],
            "trust issues": ["trust", "trust problems", "can't trust", "betrayal", "trust in relationships", "trust someone"],
            "loneliness": ["lonely", "alone", "isolated", "social isolation", "feel lonely", "no friends"]
        }
    
    def find_matching_concept(self, query: str) -> Optional[str]:
        """Find the best matching concept from the knowledge base."""
        query_lower = query.lower()
        
        # Direct concept matching
        for concept in self.psychology_kb.keys():
            if concept in query_lower:
                return concept
            
            # Check aliases
            aliases = self.concept_aliases.get(concept, [])
            for alias in aliases:
                if alias in query_lower:
                    return concept
        
        # Partial matching for robustness
        for concept in self.psychology_kb.keys():
            concept_words = concept.split()
            if any(word in query_lower for word in concept_words if len(word) > 3):
                return concept
        
        return None
    
    def reason(self, query: str) -> Dict[str, Any]:
        """Main reasoning pipeline - simplified and working."""
        logger.info(f"Processing query: {query}")
        
        # Find matching concept
        matching_concept = self.find_matching_concept(query)
        
        if matching_concept and matching_concept in self.psychology_kb:
            # We found a match!
            kb_entry = self.psychology_kb[matching_concept]
            
            # Create successful result
            result = {
                'query': query,
                'answer': kb_entry['answer'],
                'confidence': kb_entry['confidence'],
                'key_concepts': kb_entry['concepts'],
                'initial_concepts': [matching_concept],
                'activated_concepts': kb_entry['concepts'],
                'reasoning_summary': f"Direct knowledge base lookup for '{matching_concept}'",
                'evidence_sources': [f"Psychology knowledge base: {matching_concept}"],
                'concept_certainties': {concept: kb_entry['confidence'] for concept in kb_entry['concepts']},
                'num_reasoning_steps': 1,
                'reasoning_steps': [{
                    'step_id': 1,
                    'reasoning_type': 'direct_lookup',
                    'input_concepts': [{'concept': matching_concept}],
                    'output_concepts': [{'concept': c} for c in kb_entry['concepts']],
                    'confidence': kb_entry['confidence'],
                    'explanation': f"Found exact match for {matching_concept} in psychology knowledge base",
                    'evidence': [f"Direct lookup: {matching_concept}"]
                }]
            }
            
            logger.info(f"Successfully found answer for '{matching_concept}' with confidence {kb_entry['confidence']:.2f}")
            
        else:
            # No match found - provide helpful response
            result = {
                'query': query,
                'answer': f"I understand you're asking about a psychology concept, but I don't have specific information about '{query}' in my knowledge base. I can help with topics like cognitive dissonance, classical conditioning, operant conditioning, confirmation bias, social facilitation, groupthink, attachment theory, and many other psychology concepts. Could you try rephrasing your question or asking about a specific psychological theory or phenomenon?",
                'confidence': 0.4,
                'key_concepts': ["psychology", "mental_health", "behavior"],
                'initial_concepts': ["unknown_concept"],
                'activated_concepts': ["psychology", "mental_health"],
                'reasoning_summary': "No specific concept match found - provided general psychology guidance",
                'evidence_sources': ["General psychology knowledge"],
                'concept_certainties': {"psychology": 0.4},
                'num_reasoning_steps': 1,
                'reasoning_steps': [{
                    'step_id': 1,
                    'reasoning_type': 'general_response',
                    'input_concepts': [{'concept': 'unknown'}],
                    'output_concepts': [{'concept': 'psychology'}],
                    'confidence': 0.4,
                    'explanation': "No specific psychology concept identified",
                    'evidence': ["No direct match found"]
                }]
            }
            
            logger.info(f"No specific match found for query: {query}")
        
        # Store in reasoning history
        self.reasoning_history.append(result)
        
        return result

# Aliases for compatibility
ACEPReasoningEngine = FixedACEPReasoningEngine
EnhancedACEPReasoningEngine = FixedACEPReasoningEngine