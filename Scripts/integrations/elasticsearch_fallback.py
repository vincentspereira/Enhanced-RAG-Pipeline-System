from typing import List, Dict, Any, Optional, Union
from elasticsearch import AsyncElasticsearch
from elasticsearch.helpers import async_bulk
import logging
from dataclasses import dataclass
import asyncio
from datetime import datetime
import json
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import hashlib

logger = logging.getLogger(__name__)

@dataclass
class ElasticsearchConfig:
    hosts: List[str]
    index_name: str
    username: Optional[str] = None
    password: Optional[str] = None
    api_key: Optional[str] = None
    cloud_id: Optional[str] = None
    index_settings: Optional[Dict[str, Any]] = None
    index_mappings: Optional[Dict[str, Any]] = None
    batch_size: int = 1000
    timeout: int = 30
    max_retries: int = 3
    retry_interval: int = 1
    enable_fallback: bool = True
    fallback_threshold: float = 0.8
    enable_hybrid_search: bool = True
    semantic_weight: float = 0.7
    keyword_weight: float = 0.3

class ElasticsearchFallback:
    def __init__(self, config: ElasticsearchConfig):
        self.config = config
        self.es = AsyncElasticsearch(
            hosts=config.hosts,
            username=config.username,
            password=config.password,
            api_key=config.api_key,
            cloud_id=config.cloud_id,
            timeout=config.timeout,
            max_retries=config.max_retries,
            retry_on_timeout=True
        )
        self._setup_default_mappings()

    def _setup_default_mappings(self):
        """Setup default index mappings if not provided"""
        if not self.config.index_mappings:
            self.config.index_mappings = {
                "mappings": {
                    "properties": {
                        "content": {
                            "type": "text",
                            "analyzer": "standard",
                            "fields": {
                                "keyword": {"type": "keyword"},
                                "vector": {"type": "dense_vector", "dims": 768}
                            }
                        },
                        "metadata": {"type": "object"},
                        "embedding": {"type": "dense_vector", "dims": 768},
                        "timestamp": {"type": "date"},
                        "doc_id": {"type": "keyword"}
                    }
                }
            }

        if not self.config.index_settings:
            self.config.index_settings = {
                "settings": {
                    "number_of_shards": 1,
                    "number_of_replicas": 1,
                    "index": {
                        "similarity": {
                            "hybrid_sim": {
                                "type": "scripted_similarity",
                                "script": {
                                    "source": "double semantic = doc['embedding'].size() > 0 ? cosineSimilarity(params.query_vector, 'embedding') : 0; double keyword = doc['content.keyword'].size() > 0 ? 1.0 : 0; return params.semantic_weight * semantic + params.keyword_weight * keyword;"
                                }
                            }
                        }
                    }
                }
            }

    async def initialize(self):
        """Initialize Elasticsearch index"""
        try:
            if not await self.es.indices.exists(index=self.config.index_name):
                await self.es.indices.create(
                    index=self.config.index_name,
                    body={
                        **self.config.index_settings,
                        **self.config.index_mappings
                    }
                )
                logger.info(f"Created index: {self.config.index_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize Elasticsearch: {e}")
            return False

    async def index_documents(
        self,
        documents: List[Dict[str, Any]],
        embeddings: Optional[List[List[float]]] = None
    ) -> bool:
        """Index documents with their embeddings"""
        try:
            actions = []
            for i, doc in enumerate(documents):
                action = {
                    "_index": self.config.index_name,
                    "_id": doc.get("id") or hashlib.md5(
                        doc["content"].encode()
                    ).hexdigest(),
                    "_source": {
                        "content": doc["content"],
                        "metadata": doc.get("metadata", {}),
                        "timestamp": doc.get("timestamp", datetime.now().isoformat()),
                        "doc_id": doc.get("doc_id", str(i))
                    }
                }
                
                if embeddings and i < len(embeddings):
                    action["_source"]["embedding"] = embeddings[i]
                
                actions.append(action)

            # Index in batches
            success = True
            for i in range(0, len(actions), self.config.batch_size):
                batch = actions[i:i + self.config.batch_size]
                try:
                    await async_bulk(self.es, batch)
                except Exception as e:
                    logger.error(f"Failed to index batch: {e}")
                    success = False

            return success
        except Exception as e:
            logger.error(f"Indexing failed: {e}")
            return False

    async def search(
        self,
        query: str,
        query_vector: Optional[List[float]] = None,
        size: int = 10,
        min_score: float = 0.0
    ) -> List[Dict[str, Any]]:
        """
        Search documents using hybrid approach (semantic + keyword)
        Falls back to keyword search if semantic search fails
        """
        try:
            # Prepare search body
            search_body = {
                "size": size,
                "min_score": min_score,
                "_source": ["content", "metadata", "doc_id"],
                "query": {
                    "bool": {
                        "should": []
                    }
                }
            }

            # Add keyword match
            search_body["query"]["bool"]["should"].append({
                "match": {
                    "content": {
                        "query": query,
                        "boost": self.config.keyword_weight
                    }
                }
            })

            # Add semantic search if vector provided
            if query_vector and self.config.enable_hybrid_search:
                search_body["query"]["bool"]["should"].append({
                    "script_score": {
                        "query": {"match_all": {}},
                        "script": {
                            "source": "cosineSimilarity(params.vector, 'embedding') + 1.0",
                            "params": {"vector": query_vector}
                        },
                        "boost": self.config.semantic_weight
                    }
                })

            # Execute search
            response = await self.es.search(
                index=self.config.index_name,
                body=search_body
            )

            # Process results
            hits = response["hits"]["hits"]
            results = []
            
            for hit in hits:
                if hit["_score"] >= min_score:
                    results.append({
                        "content": hit["_source"]["content"],
                        "metadata": hit["_source"].get("metadata", {}),
                        "doc_id": hit["_source"]["doc_id"],
                        "score": hit["_score"]
                    })

            return results

        except Exception as e:
            logger.error(f"Search failed: {e}")
            if self.config.enable_fallback:
                # Fallback to simple keyword search
                try:
                    response = await self.es.search(
                        index=self.config.index_name,
                        body={
                            "size": size,
                            "query": {
                                "match": {
                                    "content": query
                                }
                            }
                        }
                    )
                    
                    return [{
                        "content": hit["_source"]["content"],
                        "metadata": hit["_source"].get("metadata", {}),
                        "doc_id": hit["_source"]["doc_id"],
                        "score": hit["_score"]
                    } for hit in response["hits"]["hits"]]
                except Exception as fallback_error:
                    logger.error(f"Fallback search failed: {fallback_error}")
                    
            return []

    async def delete_document(self, doc_id: str) -> bool:
        """Delete a document by ID"""
        try:
            await self.es.delete(
                index=self.config.index_name,
                id=doc_id
            )
            return True
        except Exception as e:
            logger.error(f"Delete failed: {e}")
            return False

    async def update_document(
        self,
        doc_id: str,
        content: Optional[str] = None,
        metadata: Optional[Dict] = None,
        embedding: Optional[List[float]] = None
    ) -> bool:
        """Update a document's content, metadata, or embedding"""
        try:
            update_body = {"doc": {}}
            
            if content is not None:
                update_body["doc"]["content"] = content
            if metadata is not None:
                update_body["doc"]["metadata"] = metadata
            if embedding is not None:
                update_body["doc"]["embedding"] = embedding
                
            update_body["doc"]["timestamp"] = datetime.now().isoformat()

            await self.es.update(
                index=self.config.index_name,
                id=doc_id,
                body=update_body
            )
            return True
        except Exception as e:
            logger.error(f"Update failed: {e}")
            return False

    async def get_document(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Get a document by ID"""
        try:
            response = await self.es.get(
                index=self.config.index_name,
                id=doc_id
            )
            return response["_source"]
        except Exception as e:
            logger.error(f"Get document failed: {e}")
            return None

    async def count_documents(self) -> int:
        """Get total number of documents in the index"""
        try:
            response = await self.es.count(index=self.config.index_name)
            return response["count"]
        except Exception as e:
            logger.error(f"Count failed: {e}")
            return 0

    async def clear_index(self) -> bool:
        """Clear all documents from the index"""
        try:
            await self.es.delete_by_query(
                index=self.config.index_name,
                body={"query": {"match_all": {}}}
            )
            return True
        except Exception as e:
            logger.error(f"Clear index failed: {e}")
            return False

    async def close(self):
        """Close Elasticsearch connection"""
        await self.es.close()

    async def health_check(self) -> Dict[str, Any]:
        """Check Elasticsearch and index health"""
        try:
            cluster_health = await self.es.cluster.health()
            index_health = await self.es.indices.stats(index=self.config.index_name)
            
            return {
                "cluster_status": cluster_health["status"],
                "number_of_nodes": cluster_health["number_of_nodes"],
                "active_shards": cluster_health["active_shards"],
                "index_docs_count": index_health["indices"][self.config.index_name]["total"]["docs"]["count"],
                "index_size_bytes": index_health["indices"][self.config.index_name]["total"]["store"]["size_in_bytes"]
            }
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {
                "status": "error",
                "error": str(e)
            }
