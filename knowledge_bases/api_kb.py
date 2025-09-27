"""
API-based knowledge base implementation for KERAG.
"""

import asyncio
import aiohttp
import json
from typing import List, Dict, Any, Optional
from urllib.parse import urljoin

from core.interfaces import KnowledgeBaseInterface
from core.types import Entity, Triple, RetrievalResult
from core.exceptions import APIError, EntityNotFoundError
from config.settings import KnowledgeBaseConfig


class APIKnowledgeBase(KnowledgeBaseInterface):
    """API-based knowledge base implementation."""

    def __init__(
        self,
        base_url: str,
        api_key: Optional[str] = None,
        timeout: int = 30,
        headers: Optional[Dict[str, str]] = None,
    ):
        """
        Initialize API knowledge base.

        Args:
            base_url: Base URL for the API
            api_key: API key for authentication (optional)
            timeout: Request timeout in seconds
            headers: Additional headers (optional)
        """
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout

        # Set up headers
        self.headers = KnowledgeBaseConfig.API_CONFIG["headers"].copy()
        if headers:
            self.headers.update(headers)

        if api_key:
            self.headers["Authorization"] = f"Bearer {api_key}"

        # Create session for connection pooling
        self.session = None

    async def __aenter__(self):
        """Async context manager entry."""
        connector = aiohttp.TCPConnector(limit=100, limit_per_host=30)
        timeout = aiohttp.ClientTimeout(total=self.timeout)
        self.session = aiohttp.ClientSession(
            connector=connector, timeout=timeout, headers=self.headers
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.close()

    async def _get_session(self):
        """Get or create HTTP session."""
        if not self.session:
            connector = aiohttp.TCPConnector(limit=100, limit_per_host=30)
            timeout = aiohttp.ClientTimeout(total=self.timeout)
            self.session = aiohttp.ClientSession(
                connector=connector, timeout=timeout, headers=self.headers
            )
        return self.session

    async def _make_request(self, method: str, url: str, **kwargs) -> Dict[str, Any]:
        """Make HTTP request and return JSON response."""
        try:
            session = await self._get_session()
            async with session.request(method, url, **kwargs) as response:
                if response.status == 404:
                    raise EntityNotFoundError(f"Resource not found: {url}")
                elif response.status >= 400:
                    raise APIError(f"Request failed with status {response.status}")

                return await response.json()
        except aiohttp.ClientError as e:
            raise APIError(f"Network error: {str(e)}")
        except json.JSONDecodeError as e:
            raise APIError(f"Invalid JSON response: {str(e)}")
        except Exception as e:
            raise APIError(f"Request failed: {str(e)}")

    async def get_entity_info(self, entity_id: str) -> Entity:
        """Get detailed information about an entity."""
        try:
            session = await self._get_session()
            url = urljoin(self.base_url, f"/entities/{entity_id}")

            async with session.get(url) as response:
                if response.status == 404:
                    raise EntityNotFoundError(f"Entity {entity_id} not found")
                elif response.status != 200:
                    raise APIError(f"API request failed with status {response.status}")

                data = await response.json()
                return self._parse_entity_response(data)

        except aiohttp.ClientError as e:
            raise APIError(f"Network error: {str(e)}")
        except json.JSONDecodeError as e:
            raise APIError(f"Invalid JSON response: {str(e)}")
        except Exception as e:
            raise APIError(f"Failed to get entity info for {entity_id}: {str(e)}")

    async def get_entity_neighbors(
        self, entity_id: str, relations: List[str] = None, max_hops: int = 1
    ) -> RetrievalResult:
        """Get neighbors of an entity up to max_hops."""
        try:
            session = await self._get_session()

            # Build query parameters
            params = {"max_hops": max_hops, "include_attributes": True}

            if relations:
                params["relations"] = ",".join(relations)

            url = urljoin(self.base_url, f"/entities/{entity_id}/neighbors")

            async with session.get(url, params=params) as response:
                if response.status == 404:
                    raise EntityNotFoundError(f"Entity {entity_id} not found")
                elif response.status != 200:
                    raise APIError(f"API request failed with status {response.status}")

                data = await response.json()
                return self._parse_neighbors_response(
                    data, entity_id, max_hops, relations
                )

        except aiohttp.ClientError as e:
            raise APIError(f"Network error: {str(e)}")
        except json.JSONDecodeError as e:
            raise APIError(f"Invalid JSON response: {str(e)}")
        except Exception as e:
            raise APIError(f"Failed to get neighbors for {entity_id}: {str(e)}")

    async def search_entities(
        self, query: str, entity_type: str = None, limit: int = 10
    ) -> List[Entity]:
        """Search for entities matching a query."""
        try:
            session = await self._get_session()

            # Build query parameters
            params = {"q": query, "limit": limit}

            if entity_type:
                params["type"] = entity_type

            url = urljoin(self.base_url, "/search/entities")

            async with session.get(url, params=params) as response:
                if response.status != 200:
                    raise APIError(
                        f"Search request failed with status {response.status}"
                    )

                data = await response.json()
                return self._parse_search_response(data)

        except aiohttp.ClientError as e:
            raise APIError(f"Network error: {str(e)}")
        except json.JSONDecodeError as e:
            raise APIError(f"Invalid JSON response: {str(e)}")
        except Exception as e:
            raise APIError(f"Failed to search entities: {str(e)}")

    async def get_relations(self, entity_id: str) -> List[str]:
        """Get all relations for an entity."""
        try:
            session = await self._get_session()
            url = urljoin(self.base_url, f"/entities/{entity_id}/relations")

            async with session.get(url) as response:
                if response.status == 404:
                    raise EntityNotFoundError(f"Entity {entity_id} not found")
                elif response.status != 200:
                    raise APIError(f"API request failed with status {response.status}")

                data = await response.json()
                return data.get("relations", [])

        except aiohttp.ClientError as e:
            raise APIError(f"Network error: {str(e)}")
        except json.JSONDecodeError as e:
            raise APIError(f"Invalid JSON response: {str(e)}")
        except Exception as e:
            raise APIError(f"Failed to get relations for {entity_id}: {str(e)}")

    async def execute_query(self, query: str) -> RetrievalResult:
        """Execute a custom query against the knowledge base."""
        try:
            session = await self._get_session()

            # This would depend on the specific API - here's a generic approach
            payload = {"query": query, "format": "json"}

            url = urljoin(self.base_url, "/query")

            async with session.post(url, json=payload) as response:
                if response.status != 200:
                    raise APIError(
                        f"Query request failed with status {response.status}"
                    )

                data = await response.json()
                return self._parse_query_response(data)

        except aiohttp.ClientError as e:
            raise APIError(f"Network error: {str(e)}")
        except json.JSONDecodeError as e:
            raise APIError(f"Invalid JSON response: {str(e)}")
        except Exception as e:
            raise APIError(f"Failed to execute query: {str(e)}")

    def _parse_entity_response(self, data: Dict[str, Any]) -> Entity:
        """Parse entity information from API response."""
        return Entity(
            id=data.get("id", ""),
            name=data.get("name", ""),
            type=data.get("type", "Unknown"),
            attributes=data.get("attributes", {}),
            neighbors=[],  # Will be populated separately if needed
        )

    def _parse_neighbors_response(
        self,
        data: Dict[str, Any],
        source_entity: str,
        max_hops: int,
        relations: Optional[List[str]],
    ) -> RetrievalResult:
        """Parse neighbors response from API."""
        entities = []
        triples = []

        # Parse entities
        for entity_data in data.get("entities", []):
            entity = self._parse_entity_response(entity_data)
            entities.append(entity)

        # Parse triples
        for triple_data in data.get("triples", []):
            triple = Triple(
                subject=triple_data.get("subject", ""),
                predicate=triple_data.get("predicate", ""),
                object=triple_data.get("object", ""),
                confidence=triple_data.get("confidence"),
            )
            triples.append(triple)

        return RetrievalResult(
            entities=entities,
            triples=triples,
            metadata={
                "source_entity": source_entity,
                "max_hops": max_hops,
                "relations_filter": relations,
                "total_entities": len(entities),
                "total_triples": len(triples),
                "api_response": data,
            },
        )

    def _parse_search_response(self, data: Dict[str, Any]) -> List[Entity]:
        """Parse search results from API response."""
        entities = []

        for result in data.get("results", []):
            entity = self._parse_entity_response(result)
            entities.append(entity)

        return entities

    def _parse_query_response(self, data: Dict[str, Any]) -> RetrievalResult:
        """Parse custom query response from API."""
        entities = []
        triples = []

        # This would depend on the specific API response format
        # Here's a generic implementation

        for item in data.get("results", []):
            if "type" in item and item["type"] == "entity":
                entity = self._parse_entity_response(item)
                entities.append(entity)
            elif "subject" in item and "predicate" in item and "object" in item:
                triple = Triple(
                    subject=item["subject"],
                    predicate=item["predicate"],
                    object=item["object"],
                    confidence=item.get("confidence"),
                )
                triples.append(triple)

        return RetrievalResult(
            entities=entities,
            triples=triples,
            metadata={
                "query": data.get("query", ""),
                "result_count": len(data.get("results", [])),
                "api_response": data,
            },
        )

    async def batch_get_entities(self, entity_ids: List[str]) -> List[Entity]:
        """Batch retrieve multiple entities."""
        try:
            session = await self._get_session()

            payload = {"entity_ids": entity_ids}
            url = urljoin(self.base_url, "/entities/batch")

            async with session.post(url, json=payload) as response:
                if response.status != 200:
                    raise APIError(
                        f"Batch request failed with status {response.status}"
                    )

                data = await response.json()
                entities = []

                for entity_data in data.get("entities", []):
                    entity = self._parse_entity_response(entity_data)
                    entities.append(entity)

                return entities

        except aiohttp.ClientError as e:
            raise APIError(f"Network error: {str(e)}")
        except json.JSONDecodeError as e:
            raise APIError(f"Invalid JSON response: {str(e)}")
        except Exception as e:
            raise APIError(f"Failed to batch get entities: {str(e)}")

    async def get_entity_statistics(self, entity_id: str) -> Dict[str, Any]:
        """Get statistics about an entity."""
        try:
            session = await self._get_session()
            url = urljoin(self.base_url, f"/entities/{entity_id}/stats")

            async with session.get(url) as response:
                if response.status == 404:
                    raise EntityNotFoundError(f"Entity {entity_id} not found")
                elif response.status != 200:
                    raise APIError(f"API request failed with status {response.status}")

                data = await response.json()
                return data

        except aiohttp.ClientError as e:
            raise APIError(f"Network error: {str(e)}")
        except json.JSONDecodeError as e:
            raise APIError(f"Invalid JSON response: {str(e)}")
        except Exception as e:
            raise APIError(f"Failed to get entity statistics: {str(e)}")
