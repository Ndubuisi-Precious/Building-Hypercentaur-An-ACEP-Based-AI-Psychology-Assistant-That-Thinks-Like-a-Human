"""
MongoDB database manager for Hypercentaur.
Handles storage and retrieval of queries, reasoning chains, and results.
"""

import pymongo
from pymongo import MongoClient
from typing import Dict, List, Any, Optional
import datetime
import logging
import json
from bson import ObjectId

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HypercentaurDBManager:
    """Manages MongoDB operations for Hypercentaur."""
    
    def __init__(self, connection_string: str = "mongodb://localhost:27017/", db_name: str = "hypercentaur"):
        self.connection_string = connection_string
        self.db_name = db_name
        self.client = None
        self.db = None
        self.collections = {
            'queries': 'user_queries',
            'reasoning_chains': 'reasoning_chains',
            'concepts': 'concept_knowledge',
            'sessions': 'user_sessions'
        }
        
    def connect(self) -> bool:
        """Establish connection to MongoDB."""
        try:
            self.client = MongoClient(self.connection_string)
            self.db = self.client[self.db_name]
            
            # Test connection
            self.client.admin.command('ismaster')
            logger.info(f"Connected to MongoDB: {self.db_name}")
            
            # Create indexes for better performance
            self._create_indexes()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to MongoDB: {e}")
            return False
    
    def _create_indexes(self):
        """Create database indexes for optimal performance."""
        try:
            # Index for queries collection
            queries_collection = self.db[self.collections['queries']]
            queries_collection.create_index([("timestamp", -1)])
            queries_collection.create_index([("query_hash", 1)])
            queries_collection.create_index([("concepts", 1)])
            
            # Index for reasoning chains
            reasoning_collection = self.db[self.collections['reasoning_chains']]
            reasoning_collection.create_index([("query_id", 1)])
            reasoning_collection.create_index([("confidence", -1)])
            
            # Index for concepts
            concepts_collection = self.db[self.collections['concepts']]
            concepts_collection.create_index([("concept", 1)], unique=True)
            concepts_collection.create_index([("category", 1)])
            concepts_collection.create_index([("frequency", -1)])
            
            logger.info("Database indexes created successfully")
            
        except Exception as e:
            logger.warning(f"Failed to create indexes: {e}")
    
    def store_query_result(self, query_data: Dict[str, Any]) -> Optional[str]:
        """Store a complete query and its reasoning result."""
        try:
            # Prepare document for storage
            document = {
                'query': query_data['query'],
                'answer': query_data['answer'],
                'confidence': query_data['confidence'],
                'initial_concepts': query_data.get('initial_concepts', []),
                'activated_concepts': query_data.get('activated_concepts', []),
                'key_concepts': query_data.get('key_concepts', []),
                'reasoning_summary': query_data.get('reasoning_summary', ''),
                'evidence_sources': query_data.get('evidence_sources', []),
                'concept_certainties': query_data.get('concept_certainties', {}),
                'num_reasoning_steps': query_data.get('num_reasoning_steps', 0),
                'timestamp': datetime.datetime.utcnow(),
                'query_hash': hash(query_data['query']),
                'metadata': {
                    'processing_time': query_data.get('processing_time'),
                    'user_session': query_data.get('user_session'),
                    'version': '1.0'
                }
            }
            
            # Store in queries collection
            queries_collection = self.db[self.collections['queries']]
            result = queries_collection.insert_one(document)
            query_id = str(result.inserted_id)
            
            # Store detailed reasoning chain separately
            if 'reasoning_steps' in query_data:
                self._store_reasoning_chain(query_id, query_data['reasoning_steps'])
            
            # Update concept statistics
            self._update_concept_stats(query_data.get('activated_concepts', []))
            
            logger.info(f"Stored query result with ID: {query_id}")
            return query_id
            
        except Exception as e:
            logger.error(f"Failed to store query result: {e}")
            return None
    
    def _store_reasoning_chain(self, query_id: str, reasoning_steps: List[Dict[str, Any]]):
        """Store detailed reasoning chain."""
        try:
            reasoning_document = {
                'query_id': query_id,
                'reasoning_steps': reasoning_steps,
                'num_steps': len(reasoning_steps),
                'timestamp': datetime.datetime.utcnow()
            }
            
            reasoning_collection = self.db[self.collections['reasoning_chains']]
            reasoning_collection.insert_one(reasoning_document)
            
        except Exception as e:
            logger.error(f"Failed to store reasoning chain: {e}")
    
    def _update_concept_stats(self, concepts: List[str]):
        """Update concept usage statistics."""
        try:
            concepts_collection = self.db[self.collections['concepts']]
            
            for concept in concepts:
                concepts_collection.update_one(
                    {'concept': concept},
                    {
                        '$inc': {'usage_count': 1},
                        '$set': {'last_used': datetime.datetime.utcnow()},
                        '$setOnInsert': {
                            'concept': concept,
                            'first_used': datetime.datetime.utcnow(),
                            'category': 'unknown'
                        }
                    },
                    upsert=True
                )
                
        except Exception as e:
            logger.error(f"Failed to update concept stats: {e}")
    
    def get_query_history(self, limit: int = 50, user_session: Optional[str] = None) -> List[Dict[str, Any]]:
        """Retrieve recent query history."""
        try:
            queries_collection = self.db[self.collections['queries']]
            
            # Build query filter
            query_filter = {}
            if user_session:
                query_filter['metadata.user_session'] = user_session
            
            # Retrieve queries sorted by timestamp
            cursor = queries_collection.find(query_filter).sort('timestamp', -1).limit(limit)
            
            queries = []
            for doc in cursor:
                doc['_id'] = str(doc['_id'])  # Convert ObjectId to string
                queries.append(doc)
            
            logger.info(f"Retrieved {len(queries)} queries from history")
            return queries
            
        except Exception as e:
            logger.error(f"Failed to retrieve query history: {e}")
            return []
    
    def get_similar_queries(self, query: str, concepts: List[str], limit: int = 5) -> List[Dict[str, Any]]:
        """Find similar queries based on concepts."""
        try:
            queries_collection = self.db[self.collections['queries']]
            
            # Find queries with overlapping concepts
            query_filter = {
                '$or': [
                    {'initial_concepts': {'$in': concepts}},
                    {'activated_concepts': {'$in': concepts}},
                    {'key_concepts': {'$in': concepts}}
                ]
            }
            
            cursor = queries_collection.find(query_filter).sort('confidence', -1).limit(limit)
            
            similar_queries = []
            for doc in cursor:
                doc['_id'] = str(doc['_id'])
                similar_queries.append(doc)
            
            return similar_queries
            
        except Exception as e:
            logger.error(f"Failed to find similar queries: {e}")
            return []
    
    def get_concept_analytics(self) -> Dict[str, Any]:
        """Get analytics about concept usage."""
        try:
            concepts_collection = self.db[self.collections['concepts']]
            queries_collection = self.db[self.collections['queries']]
            
            # Most used concepts
            most_used = list(concepts_collection.find().sort('usage_count', -1).limit(10))
            
            # Total queries
            total_queries = queries_collection.count_documents({})
            
            # Average confidence
            pipeline = [
                {'$group': {
                    '_id': None,
                    'avg_confidence': {'$avg': '$confidence'},
                    'max_confidence': {'$max': '$confidence'},
                    'min_confidence': {'$min': '$confidence'}
                }}
            ]
            confidence_stats = list(queries_collection.aggregate(pipeline))
            
            # Concept categories distribution
            category_pipeline = [
                {'$group': {
                    '_id': '$category',
                    'count': {'$sum': 1}
                }},
                {'$sort': {'count': -1}}
            ]
            category_dist = list(concepts_collection.aggregate(category_pipeline))
            
            return {
                'most_used_concepts': [
                    {
                        'concept': doc['concept'],
                        'usage_count': doc.get('usage_count', 0),
                        'category': doc.get('category', 'unknown')
                    }
                    for doc in most_used
                ],
                'total_queries': total_queries,
                'confidence_stats': confidence_stats[0] if confidence_stats else {},
                'category_distribution': category_dist,
                'total_concepts': concepts_collection.count_documents({})
            }
            
        except Exception as e:
            logger.error(f"Failed to get concept analytics: {e}")
            return {}
    
    def get_reasoning_chain(self, query_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve detailed reasoning chain for a specific query."""
        try:
            reasoning_collection = self.db[self.collections['reasoning_chains']]
            
            chain = reasoning_collection.find_one({'query_id': query_id})
            if chain:
                chain['_id'] = str(chain['_id'])
                return chain
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to retrieve reasoning chain: {e}")
            return None
    
    def search_queries(self, search_term: str, limit: int = 20) -> List[Dict[str, Any]]:
        """Search queries by text."""
        try:
            queries_collection = self.db[self.collections['queries']]
            
            # Create text search query
            query_filter = {
                '$or': [
                    {'query': {'$regex': search_term, '$options': 'i'}},
                    {'answer': {'$regex': search_term, '$options': 'i'}},
                    {'initial_concepts': {'$regex': search_term, '$options': 'i'}},
                    {'key_concepts': {'$regex': search_term, '$options': 'i'}}
                ]
            }
            
            cursor = queries_collection.find(query_filter).sort('timestamp', -1).limit(limit)
            
            results = []
            for doc in cursor:
                doc['_id'] = str(doc['_id'])
                results.append(doc)
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to search queries: {e}")
            return []
    
    def export_data(self, collection_name: str, query_filter: Optional[Dict] = None) -> List[Dict[str, Any]]:
        """Export data from a specific collection."""
        try:
            if collection_name not in self.collections.values():
                raise ValueError(f"Unknown collection: {collection_name}")
            
            collection = self.db[collection_name]
            filter_query = query_filter or {}
            
            cursor = collection.find(filter_query)
            
            data = []
            for doc in cursor:
                doc['_id'] = str(doc['_id'])  # Convert ObjectId to string
                data.append(doc)
            
            return data
            
        except Exception as e:
            logger.error(f"Failed to export data: {e}")
            return []
    
    def get_session_stats(self, user_session: str) -> Dict[str, Any]:
        """Get statistics for a specific user session."""
        try:
            queries_collection = self.db[self.collections['queries']]
            
            session_filter = {'metadata.user_session': user_session}
            
            # Count queries
            query_count = queries_collection.count_documents(session_filter)
            
            # Get average confidence
            pipeline = [
                {'$match': session_filter},
                {'$group': {
                    '_id': None,
                    'avg_confidence': {'$avg': '$confidence'},
                    'total_reasoning_steps': {'$sum': '$num_reasoning_steps'}
                }}
            ]
            
            stats = list(queries_collection.aggregate(pipeline))
            
            # Get most used concepts in session
            concept_pipeline = [
                {'$match': session_filter},
                {'$unwind': '$activated_concepts'},
                {'$group': {
                    '_id': '$activated_concepts',
                    'count': {'$sum': 1}
                }},
                {'$sort': {'count': -1}},
                {'$limit': 5}
            ]
            
            top_concepts = list(queries_collection.aggregate(concept_pipeline))
            
            return {
                'session_id': user_session,
                'query_count': query_count,
                'avg_confidence': stats[0]['avg_confidence'] if stats else 0,
                'total_reasoning_steps': stats[0]['total_reasoning_steps'] if stats else 0,
                'top_concepts': [doc['_id'] for doc in top_concepts]
            }
            
        except Exception as e:
            logger.error(f"Failed to get session stats: {e}")
            return {}
    
    def close_connection(self):
        """Close MongoDB connection."""
        if self.client:
            self.client.close()
            logger.info("MongoDB connection closed")