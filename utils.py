"""
Utility functions for Hypercentaur.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple
import hashlib
import re
import json
import logging
from datetime import datetime
import uuid

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def calculate_similarity_matrix(embeddings: Dict[str, np.ndarray]) -> np.ndarray:
    """Calculate cosine similarity matrix for concept embeddings."""
    concepts = list(embeddings.keys())
    n = len(concepts)
    
    if n == 0:
        return np.array([])
    
    # Stack embeddings
    embedding_matrix = np.stack([embeddings[concept] for concept in concepts])
    
    # Calculate cosine similarity
    norms = np.linalg.norm(embedding_matrix, axis=1, keepdims=True)
    normalized_embeddings = embedding_matrix / norms
    similarity_matrix = np.dot(normalized_embeddings, normalized_embeddings.T)
    
    return similarity_matrix

def find_concept_clusters(similarity_matrix: np.ndarray, concepts: List[str], 
                         threshold: float = 0.7) -> List[List[str]]:
    """Find clusters of similar concepts."""
    if len(similarity_matrix) == 0:
        return []
    
    clusters = []
    visited = set()
    
    for i, concept in enumerate(concepts):
        if concept in visited:
            continue
        
        # Find all concepts similar to current concept
        similar_indices = np.where(similarity_matrix[i] > threshold)[0]
        cluster = [concepts[j] for j in similar_indices if concepts[j] not in visited]
        
        if len(cluster) > 1:
            clusters.append(cluster)
            visited.update(cluster)
        else:
            visited.add(concept)
    
    return clusters

def normalize_text(text: str) -> str:
    """Normalize text for consistent processing."""
    if not text:
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove special characters but keep underscores for concept names
    text = re.sub(r'[^\w\s_-]', '', text)
    
    return text.strip()

def extract_keywords(text: str, min_length: int = 3, max_keywords: int = 10) -> List[str]:
    """Extract keywords from text using simple heuristics."""
    if not text:
        return []
    
    # Normalize text
    normalized = normalize_text(text)
    
    # Split into words
    words = normalized.split()
    
    # Filter by length and common stop words
    stop_words = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those', 'a', 'an'}
    
    keywords = []
    for word in words:
        if (len(word) >= min_length and 
            word not in stop_words and 
            not word.isdigit()):
            keywords.append(word)
    
    # Remove duplicates while preserving order
    seen = set()
    unique_keywords = []
    for keyword in keywords:
        if keyword not in seen:
            seen.add(keyword)
            unique_keywords.append(keyword)
    
    return unique_keywords[:max_keywords]

def calculate_concept_importance(concept: str, concept_graph, 
                               frequency_weight: float = 0.4,
                               centrality_weight: float = 0.6) -> float:
    """Calculate importance score for a concept."""
    if not concept_graph.has_node(concept):
        return 0.0
    
    # Get node attributes
    node_data = concept_graph.nodes[concept]
    frequency = node_data.get('frequency', 1)
    
    # Calculate centrality
    try:
        import networkx as nx
        centrality = nx.degree_centrality(concept_graph)[concept]
    except:
        centrality = 0.0
    
    # Normalize frequency (assuming max frequency of 100)
    normalized_frequency = min(frequency / 100.0, 1.0)
    
    # Weighted combination
    importance = (frequency_weight * normalized_frequency + 
                 centrality_weight * centrality)
    
    return importance

def generate_session_id() -> str:
    """Generate a unique session ID."""
    return str(uuid.uuid4())

def hash_query(query: str) -> str:
    """Generate a hash for a query for deduplication."""
    normalized_query = normalize_text(query)
    return hashlib.md5(normalized_query.encode()).hexdigest()

def format_concept_name(concept: str) -> str:
    """Format concept name for display."""
    return concept.replace('_', ' ').title()

def validate_confidence_score(score: float) -> float:
    """Validate and normalize confidence score to [0, 1] range."""
    if score < 0:
        return 0.0
    elif score > 1:
        return 1.0
    else:
        return float(score)

def calculate_reasoning_quality(reasoning_steps: List[Dict[str, Any]]) -> Dict[str, float]:
    """Calculate quality metrics for a reasoning chain."""
    if not reasoning_steps:
        return {'quality_score': 0.0, 'coherence': 0.0, 'completeness': 0.0}
    
    # Extract confidence scores
    confidences = [step.get('confidence', 0) for step in reasoning_steps]
    
    # Quality metrics
    avg_confidence = np.mean(confidences)
    min_confidence = np.min(confidences)
    confidence_variance = np.var(confidences)
    
    # Coherence: low variance in confidence indicates coherent reasoning
    coherence = max(0, 1 - confidence_variance)
    
    # Completeness: based on number of reasoning steps and types
    reasoning_types = set(step.get('reasoning_type', '') for step in reasoning_steps)
    completeness = min(len(reasoning_types) / 4.0, 1.0)  # Assume 4 types is complete
    
    # Overall quality score
    quality_score = (0.5 * avg_confidence + 
                    0.2 * min_confidence + 
                    0.15 * coherence + 
                    0.15 * completeness)
    
    return {
        'quality_score': quality_score,
        'avg_confidence': avg_confidence,
        'min_confidence': min_confidence,
        'coherence': coherence,
        'completeness': completeness,
        'num_steps': len(reasoning_steps),
        'unique_types': len(reasoning_types)
    }

def create_concept_summary(concepts: List[str], concept_graph) -> Dict[str, Any]:
    """Create a summary of concepts and their relationships."""
    if not concepts or not concept_graph:
        return {}
    
    summary = {
        'total_concepts': len(concepts),
        'concepts': [],
        'categories': {},
        'avg_importance': 0.0,
        'top_concepts': []
    }
    
    importance_scores = []
    
    for concept in concepts:
        if concept_graph.has_node(concept):
            node_data = concept_graph.nodes[concept]
            frequency = node_data.get('frequency', 1)
            category = node_data.get('category', 'unknown')
            
            importance = calculate_concept_importance(concept, concept_graph)
            importance_scores.append(importance)
            
            concept_info = {
                'name': concept,
                'formatted_name': format_concept_name(concept),
                'frequency': frequency,
                'category': category,
                'importance': importance
            }
            
            summary['concepts'].append(concept_info)
            
            # Category statistics
            if category not in summary['categories']:
                summary['categories'][category] = 0
            summary['categories'][category] += 1
    
    # Calculate averages
    if importance_scores:
        summary['avg_importance'] = np.mean(importance_scores)
    
    # Sort concepts by importance and get top 5
    summary['concepts'].sort(key=lambda x: x['importance'], reverse=True)
    summary['top_concepts'] = summary['concepts'][:5]
    
    return summary

def validate_reasoning_step(step: Dict[str, Any]) -> bool:
    """Validate a reasoning step structure."""
    required_fields = ['step_id', 'reasoning_type', 'confidence', 'explanation']
    
    for field in required_fields:
        if field not in step:
            return False
    
    # Validate confidence score
    confidence = step.get('confidence')
    if not isinstance(confidence, (int, float)) or confidence < 0 or confidence > 1:
        return False
    
    return True

def merge_concept_tokens(tokens: List[Dict[str, Any]], 
                        merge_threshold: float = 0.8) -> List[Dict[str, Any]]:
    """Merge similar concept tokens to reduce redundancy."""
    if len(tokens) <= 1:
        return tokens
    
    merged_tokens = []
    processed_indices = set()
    
    for i, token1 in enumerate(tokens):
        if i in processed_indices:
            continue
        
        # Find similar tokens
        similar_indices = [i]
        concept1 = token1['concept']
        
        for j, token2 in enumerate(tokens[i+1:], i+1):
            if j in processed_indices:
                continue
            
            concept2 = token2['concept']
            
            # Simple similarity check based on concept names
            similarity = calculate_string_similarity(concept1, concept2)
            
            if similarity > merge_threshold:
                similar_indices.append(j)
        
        # Merge similar tokens
        if len(similar_indices) > 1:
            merged_token = merge_tokens([tokens[idx] for idx in similar_indices])
            merged_tokens.append(merged_token)
        else:
            merged_tokens.append(token1)
        
        processed_indices.update(similar_indices)
    
    return merged_tokens

def merge_tokens(tokens: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Merge multiple concept tokens into one."""
    if not tokens:
        return {}
    
    if len(tokens) == 1:
        return tokens[0]
    
    # Use the token with highest activation as base
    base_token = max(tokens, key=lambda t: t.get('activation_strength', 0))
    
    merged = base_token.copy()
    
    # Combine activation strengths and certainties
    activations = [t.get('activation_strength', 0) for t in tokens]
    certainties = [t.get('certainty', 0) for t in tokens]
    
    merged['activation_strength'] = np.mean(activations)
    merged['certainty'] = np.mean(certainties)
    
    # Combine evidence sources
    all_evidence = []
    for token in tokens:
        all_evidence.extend(token.get('source_evidence', []))
    merged['source_evidence'] = list(set(all_evidence))
    
    # Add merge metadata
    merged['metadata'] = merged.get('metadata', {})
    merged['metadata']['merged_from'] = [t['concept'] for t in tokens]
    merged['metadata']['merge_count'] = len(tokens)
    
    return merged

def calculate_string_similarity(str1: str, str2: str) -> float:
    """Calculate similarity between two strings using Jaccard similarity."""
    if not str1 or not str2:
        return 0.0
    
    # Convert to sets of characters/words
    set1 = set(str1.lower().split('_'))
    set2 = set(str2.lower().split('_'))
    
    if not set1 and not set2:
        return 1.0
    
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    
    if union == 0:
        return 0.0
    
    return intersection / union

def export_reasoning_to_json(reasoning_result: Dict[str, Any], 
                           filename: str = None) -> str:
    """Export reasoning result to JSON format."""
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"reasoning_export_{timestamp}.json"
    
    # Prepare export data
    export_data = {
        'timestamp': datetime.now().isoformat(),
        'query': reasoning_result.get('query', ''),
        'answer': reasoning_result.get('answer', ''),
        'confidence': reasoning_result.get('confidence', 0),
        'reasoning_steps': reasoning_result.get('reasoning_steps', []),
        'key_concepts': reasoning_result.get('key_concepts', []),
        'metadata': {
            'num_reasoning_steps': reasoning_result.get('num_reasoning_steps', 0),
            'reasoning_summary': reasoning_result.get('reasoning_summary', ''),
            'evidence_sources': reasoning_result.get('evidence_sources', [])
        }
    }
    
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Reasoning result exported to {filename}")
        return filename
        
    except Exception as e:
        logger.error(f"Failed to export reasoning result: {e}")
        return ""

def load_reasoning_from_json(filename: str) -> Dict[str, Any]:
    """Load reasoning result from JSON file."""
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        logger.info(f"Reasoning result loaded from {filename}")
        return data
        
    except Exception as e:
        logger.error(f"Failed to load reasoning result: {e}")
        return {}

def get_system_stats() -> Dict[str, Any]:
    """Get basic system statistics."""
    import psutil
    import platform
    
    try:
        stats = {
            'platform': platform.system(),
            'python_version': platform.python_version(),
            'cpu_count': psutil.cpu_count(),
            'memory_total': psutil.virtual_memory().total,
            'memory_available': psutil.virtual_memory().available,
            'memory_percent': psutil.virtual_memory().percent,
            'timestamp': datetime.now().isoformat()
        }
        
        return stats
        
    except Exception as e:
        logger.error(f"Failed to get system stats: {e}")
        return {'error': str(e)}