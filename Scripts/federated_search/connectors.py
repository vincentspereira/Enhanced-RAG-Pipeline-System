from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import logging
import asyncio # Added for mock web api latency

# Assuming SearchResult is defined in Scripts.search_system
# Adjust import path if necessary, or redefine/import from a common types module
# For now, let's try a relative import path assuming a certain structure or rely on PYTHONPATH
try:
    from ..search_system import SearchResult
except ImportError:
    # Fallback for cases where relative import fails (e.g. running script directly)
    # This might require search_system.py to be in PYTHONPATH
    # Or define a minimal SearchResult here for type hinting if it's simple enough
    # from dataclasses import dataclass
    # @dataclass
    # class SearchResult:
    #     doc_id: str
    #     score: float
    #     content: Optional[str] = None
    #     metadata: Optional[Dict[str, Any]] = None
    #     source_type: str = "unknown_federated"
    # For now, assume the try-except for relative import works during actual execution by AdvancedSearchSystem
    pass


logger = logging.getLogger(__name__)

class FederatedDataSource(ABC):
    """Abstract Base Class for federated data source connectors."""

    def __init__(self, source_name: str, config: Optional[Dict[str, Any]] = None):
        self.source_name = source_name
        self.config = config or {}
        logger.info(f"FederatedDataSource '{self.source_name}' initialized.")

    @abstractmethod
    async def search(self, query: str, top_k: int = 5) -> List[SearchResult]:
        """
        Performs a search against the federated data source.
        Must be implemented by concrete connector classes.
        """
        pass

    def is_available(self) -> bool:
        """
        Checks if the federated data source is available (e.g., credentials configured, service reachable).
        Default implementation returns True. Override for specific checks.
        """
        return True


class MockDatabaseConnector(FederatedDataSource):
    """A mock database connector for federated search."""

    def __init__(self, source_name: str = "mock_db", config: Optional[Dict[str, Any]] = None):
        super().__init__(source_name, config)
        self.mock_data = {
            "ai": [
                SearchResult(doc_id=f"{self.source_name}_doc_ai1", score=0.88, content="Mock DB content about AI and models.", metadata={"db_table": "articles"}, source_type=self.source_name),
                SearchResult(doc_id=f"{self.source_name}_doc_ai2", score=0.78, content="Another AI result from mock DB.", metadata={"db_table": "papers"}, source_type=self.source_name),
            ],
            "nature": [
                SearchResult(doc_id=f"{self.source_name}_doc_nature1", score=0.90, content="Mock DB content about nature and forests.", metadata={"db_table": "images_meta"}, source_type=self.source_name),
            ]
        }

    async def search(self, query: str, top_k: int = 5) -> List[SearchResult]:
        logger.info(f"MockDatabaseConnector '{self.source_name}' searching for: '{query}'")
        results = []
        for keyword, docs in self.mock_data.items():
            if keyword in query.lower():
                results.extend(docs)

        # Sort by score and return top_k
        results.sort(key=lambda x: x.score, reverse=True)
        return results[:top_k]

class MockWebApiConnector(FederatedDataSource):
    """A mock Web API connector for federated search."""

    def __init__(self, source_name: str = "mock_web_api", config: Optional[Dict[str, Any]] = None):
        super().__init__(source_name, config)
        self.api_endpoint = self.config.get("api_endpoint", "http://mock-api.example.com/search")

    async def search(self, query: str, top_k: int = 5) -> List[SearchResult]:
        logger.info(f"MockWebApiConnector '{self.source_name}' searching for: '{query}' at {self.api_endpoint}")
        # Simulate an API call
        # In a real scenario, use httpx.AsyncClient here
        await asyncio.sleep(0.1) # Simulate network latency

        mock_api_responses = {
            "AI": [
                {"id": "web_ai_001", "title": "AI Breakthroughs 2024", "snippet": "Web content discussing AI...", "url": "http://example.com/ai1", "score": 0.82},
            ],
            "federated search": [
                 {"id": "web_fed_001", "title": "Understanding Federated Search", "snippet": "Web content on federated systems...", "url": "http://example.com/fed1", "score": 0.91},
            ]
        }

        results = []
        for term, api_hits in mock_api_responses.items():
            if term.lower() in query.lower():
                for hit in api_hits:
                    results.append(SearchResult(
                        doc_id=f"{self.source_name}_{hit['id']}",
                        score=hit['score'],
                        content=hit['snippet'],
                        metadata={"title": hit['title'], "url": hit['url']},
                        source_type=self.source_name
                    ))

        results.sort(key=lambda x: x.score, reverse=True)
        return results[:top_k]

# Example of how these might be instantiated
# if __name__ == "__main__":
#     async def main():
#         db_connector = MockDatabaseConnector()
#         web_connector = MockWebApiConnector()

#         db_results = await db_connector.search("AI query", top_k=1)
#         print("DB Results:", db_results)

#         web_results = await web_connector.search("AI query", top_k=1)
#         print("Web Results:", web_results)

#     asyncio.run(main())
```
