"""
Extended test suite for knowledge bases to improve coverage.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch

from core.types import Entity, Triple, RetrievalResult
from core.exceptions import SPARQLError, APIError, EntityNotFoundError
from knowledge_bases.sparql_kb import SPARQLKnowledgeBase
from knowledge_bases.api_kb import APIKnowledgeBase


class TestSPARQLKnowledgeBaseExtended:
    """Extended test suite for SPARQL knowledge base."""

    @pytest.fixture
    def sparql_kb(self):
        """Create SPARQL knowledge base instance for testing."""
        return SPARQLKnowledgeBase(
            endpoint_url="http://test-endpoint.com/sparql", timeout=10
        )

    @pytest.mark.asyncio
    async def test_get_entity_type_success(self, sparql_kb):
        """Test successful entity type retrieval."""
        mock_response = {
            "results": {
                "bindings": [
                    {
                        "type": {"value": "http://example.org/Person"},
                    }
                ]
            }
        }

        with patch.object(sparql_kb, "_execute_query", return_value=mock_response):
            entity_type = await sparql_kb._get_entity_type(
                "http://example.org/test_entity"
            )
            assert entity_type == "http://example.org/Person"

    @pytest.mark.asyncio
    async def test_get_entity_type_not_found(self, sparql_kb):
        """Test entity type retrieval when not found."""
        empty_response = {"results": {"bindings": []}}

        with patch.object(sparql_kb, "_execute_query", return_value=empty_response):
            entity_type = await sparql_kb._get_entity_type(
                "http://example.org/nonexistent"
            )
            assert entity_type == "http://www.w3.org/2002/07/owl#Thing"  # Default type

    @pytest.mark.asyncio
    async def test_get_entity_neighbors_with_filters(self, sparql_kb):
        """Test getting entity neighbors with filters."""
        mock_response = {
            "results": {
                "bindings": [
                    {
                        "neighbor": {"value": "http://example.org/neighbor1"},
                        "relation": {"value": "http://example.org/relatedTo"},
                        "hop": {"value": "1"},
                    }
                ]
            }
        }

        mock_entity = Entity(
            id="http://example.org/neighbor1",
            name="Neighbor 1",
            type="Person",
            attributes={},
            neighbors=[],
        )

        with patch.object(sparql_kb, "_execute_query", return_value=mock_response):
            with patch.object(sparql_kb, "get_entity_info", return_value=mock_entity):
                result = await sparql_kb.get_entity_neighbors(
                    "http://example.org/test_entity",
                    max_hops=1,
                    relations=["http://example.org/relatedTo"],
                    filters={"type": "Person"},
                )

                assert len(result.entities) == 1
                assert len(result.triples) == 1

    @pytest.mark.asyncio
    async def test_search_entities_with_filters(self, sparql_kb):
        """Test entity search with filters."""
        mock_response = {
            "results": {
                "bindings": [
                    {
                        "entity": {"value": "http://example.org/entity1"},
                        "label": {"value": "Test Entity 1"},
                    }
                ]
            }
        }

        mock_entity = Entity(
            id="http://example.org/entity1",
            name="Test Entity 1",
            type="Person",
            attributes={},
            neighbors=[],
        )

        with patch.object(sparql_kb, "_execute_query", return_value=mock_response):
            with patch.object(sparql_kb, "get_entity_info", return_value=mock_entity):
                entities = await sparql_kb.search_entities(
                    "test",
                    limit=5,
                    entity_type="Person",
                    filters={"birth_year": ">1990"},
                )

                assert len(entities) == 1
                assert entities[0].name == "Test Entity 1"

    @pytest.mark.asyncio
    async def test_execute_query_success(self, sparql_kb):
        """Test successful query execution."""
        mock_response = {"results": {"bindings": []}}

        with patch.object(sparql_kb.sparql, "query") as mock_query:
            mock_query.return_value = mock_response

            result = await sparql_kb.execute_query("SELECT * WHERE { ?s ?p ?o }")
            assert result == mock_response

    @pytest.mark.asyncio
    async def test_execute_query_error(self, sparql_kb):
        """Test query execution error handling."""
        with patch.object(sparql_kb.sparql, "query") as mock_query:
            mock_query.side_effect = Exception("SPARQL endpoint error")

            with pytest.raises(SPARQLError, match="Failed to execute query"):
                await sparql_kb.execute_query("INVALID SPARQL QUERY")

    @pytest.mark.asyncio
    async def test_get_relations_success(self, sparql_kb):
        """Test getting available relations."""
        mock_response = {
            "results": {
                "bindings": [
                    {"relation": {"value": "http://example.org/relatedTo"}},
                    {"relation": {"value": "http://example.org/knows"}},
                ]
            }
        }

        with patch.object(sparql_kb, "_execute_query", return_value=mock_response):
            relations = await sparql_kb.get_relations()
            assert len(relations) == 2
            assert "http://example.org/relatedTo" in relations

    @pytest.mark.asyncio
    async def test_get_relations_error(self, sparql_kb):
        """Test getting relations with error."""
        with patch.object(sparql_kb, "_execute_query") as mock_execute:
            mock_execute.side_effect = SPARQLError("Query failed")

            with pytest.raises(SPARQLError):
                await sparql_kb.get_relations()

    def test_extract_name_from_uri_edge_cases(self, sparql_kb):
        """Test URI name extraction with edge cases."""
        # Test with fragment
        assert sparql_kb._extract_name_from_uri("http://example.org#Person") == "Person"

        # Test with path
        assert (
            sparql_kb._extract_name_from_uri("http://example.org/entity/name") == "name"
        )

        # Test with query parameters
        assert (
            sparql_kb._extract_name_from_uri("http://example.org/entity?param=value")
            == "entity"
        )

        # Test with simple name
        assert sparql_kb._extract_name_from_uri("simple_name") == "simple_name"

        # Test with empty string
        assert sparql_kb._extract_name_from_uri("") == ""

    @pytest.mark.asyncio
    async def test_sparql_kb_initialization(self):
        """Test SPARQL knowledge base initialization."""
        # Test with default graph
        kb = SPARQLKnowledgeBase(
            endpoint_url="http://test.com/sparql",
            default_graph="http://test.com/graph",
            timeout=30,
        )

        assert kb.endpoint_url == "http://test.com/sparql"
        assert kb.default_graph == "http://test.com/graph"
        assert kb.timeout == 30

    @pytest.mark.asyncio
    async def test_sparql_kb_context_manager(self, sparql_kb):
        """Test SPARQL knowledge base context manager."""
        async with sparql_kb:
            assert sparql_kb.sparql is not None


class TestAPIKnowledgeBaseExtended:
    """Extended test suite for API knowledge base."""

    @pytest.fixture
    def api_kb(self):
        """Create API knowledge base instance for testing."""
        return APIKnowledgeBase(
            base_url="https://api.test.com", api_key="test_key", timeout=10
        )

    @pytest.mark.asyncio
    async def test_get_entity_info_with_attributes(self, api_kb):
        """Test entity info retrieval with complex attributes."""
        mock_response_data = {
            "id": "test_entity_123",
            "name": "Test Entity",
            "type": "Person",
            "attributes": {
                "age": 30,
                "location": "Test City",
                "skills": ["programming", "writing"],
                "metadata": {"verified": True, "score": 0.95},
            },
        }

        with patch.object(api_kb, "_make_request") as mock_request:
            mock_request.return_value = mock_response_data

            entity = await api_kb.get_entity_info("test_entity_123")

            assert entity.id == "test_entity_123"
            assert entity.name == "Test Entity"
            assert entity.type == "Person"
            assert entity.attributes["age"] == 30
            assert entity.attributes["skills"] == ["programming", "writing"]
            assert entity.attributes["metadata"]["verified"] is True

    @pytest.mark.asyncio
    async def test_get_entity_neighbors_with_complex_data(self, api_kb):
        """Test getting entity neighbors with complex data."""
        mock_response_data = {
            "entities": [
                {
                    "id": "neighbor1",
                    "name": "Neighbor 1",
                    "type": "Person",
                    "attributes": {"age": 25, "role": "friend"},
                },
                {
                    "id": "neighbor2",
                    "name": "Neighbor 2",
                    "type": "Organization",
                    "attributes": {"type": "company", "size": "large"},
                },
            ],
            "triples": [
                {
                    "subject": "test_entity",
                    "predicate": "relatedTo",
                    "object": "neighbor1",
                    "confidence": 0.9,
                    "metadata": {"source": "user_input", "timestamp": "2024-01-01"},
                },
                {
                    "subject": "test_entity",
                    "predicate": "worksFor",
                    "object": "neighbor2",
                    "confidence": 0.8,
                },
            ],
        }

        with patch.object(api_kb, "_make_request") as mock_request:
            mock_request.return_value = mock_response_data

            result = await api_kb.get_entity_neighbors("test_entity", max_hops=2)

            assert len(result.entities) == 2
            assert len(result.triples) == 2
            assert result.triples[0].confidence == 0.9
            assert result.triples[1].confidence == 0.8

    @pytest.mark.asyncio
    async def test_search_entities_with_pagination(self, api_kb):
        """Test entity search with pagination."""
        mock_response_data = {
            "results": [
                {
                    "id": "entity1",
                    "name": "Test Entity 1",
                    "type": "Person",
                    "attributes": {},
                },
                {
                    "id": "entity2",
                    "name": "Test Entity 2",
                    "type": "Organization",
                    "attributes": {},
                },
            ],
            "pagination": {"page": 1, "per_page": 2, "total": 10, "has_next": True},
        }

        with patch.object(api_kb, "_make_request") as mock_request:
            mock_request.return_value = mock_response_data

            entities = await api_kb.search_entities("test", limit=2, page=1)

            assert len(entities) == 2
            assert entities[0].name == "Test Entity 1"
            assert entities[1].name == "Test Entity 2"

    @pytest.mark.asyncio
    async def test_batch_get_entities_large_batch(self, api_kb):
        """Test batch entity retrieval with large batch."""
        # Create a large batch of entities
        large_batch = [f"entity{i}" for i in range(100)]

        mock_response_data = {
            "entities": [
                {
                    "id": f"entity{i}",
                    "name": f"Entity {i}",
                    "type": "Person",
                    "attributes": {},
                }
                for i in range(100)
            ]
        }

        with patch.object(api_kb, "_make_request") as mock_request:
            mock_request.return_value = mock_response_data

            entities = await api_kb.batch_get_entities(large_batch)

            assert len(entities) == 100
            assert entities[0].name == "Entity 0"
            assert entities[99].name == "Entity 99"

    @pytest.mark.asyncio
    async def test_api_kb_initialization_with_headers(self):
        """Test API knowledge base initialization with custom headers."""
        custom_headers = {
            "X-Custom-Header": "custom_value",
            "User-Agent": "KERAG-Test/1.0",
        }

        kb = APIKnowledgeBase(
            base_url="https://api.test.com",
            api_key="test_key",
            headers=custom_headers,
            timeout=30,
        )

        assert "X-Custom-Header" in kb.headers
        assert kb.headers["X-Custom-Header"] == "custom_value"
        assert kb.headers["User-Agent"] == "KERAG-Test/1.0"

    @pytest.mark.asyncio
    async def test_api_kb_context_manager(self, api_kb):
        """Test API knowledge base context manager."""
        async with api_kb:
            assert api_kb.session is not None

    @pytest.mark.asyncio
    async def test_api_kb_error_handling_various_status_codes(self, api_kb):
        """Test API knowledge base error handling with various status codes."""
        # Test 400 Bad Request
        with patch.object(api_kb, "_make_request") as mock_request:
            mock_request.side_effect = APIError("Bad Request", status_code=400)

            with pytest.raises(APIError, match="Bad Request"):
                await api_kb.get_entity_info("invalid_entity")

        # Test 403 Forbidden
        with patch.object(api_kb, "_make_request") as mock_request:
            mock_request.side_effect = APIError("Forbidden", status_code=403)

            with pytest.raises(APIError, match="Forbidden"):
                await api_kb.get_entity_info("restricted_entity")

        # Test 429 Too Many Requests
        with patch.object(api_kb, "_make_request") as mock_request:
            mock_request.side_effect = APIError("Too Many Requests", status_code=429)

            with pytest.raises(APIError, match="Too Many Requests"):
                await api_kb.get_entity_info("rate_limited_entity")

    @pytest.mark.asyncio
    async def test_api_kb_network_error_handling(self, api_kb):
        """Test API knowledge base network error handling."""
        with patch.object(api_kb, "_make_request") as mock_request:
            mock_request.side_effect = Exception("Network connection failed")

            with pytest.raises(APIError, match="Network connection failed"):
                await api_kb.get_entity_info("test_entity")

    @pytest.mark.asyncio
    async def test_api_kb_json_decode_error(self, api_kb):
        """Test API knowledge base JSON decode error handling."""
        with patch.object(api_kb, "_make_request") as mock_request:
            mock_request.side_effect = Exception("Invalid JSON response")

            with pytest.raises(APIError, match="Invalid JSON response"):
                await api_kb.get_entity_info("test_entity")

    @pytest.mark.asyncio
    async def test_api_kb_timeout_handling(self, api_kb):
        """Test API knowledge base timeout handling."""
        with patch.object(api_kb, "_make_request") as mock_request:
            mock_request.side_effect = Exception("Request timeout")

            with pytest.raises(APIError, match="Request timeout"):
                await api_kb.get_entity_info("test_entity")

    @pytest.mark.asyncio
    async def test_api_kb_parse_entity_response_edge_cases(self, api_kb):
        """Test API knowledge base entity response parsing edge cases."""
        # Test with missing fields
        incomplete_response = {
            "id": "test_entity",
            # Missing name and type
        }

        with patch.object(api_kb, "_make_request") as mock_request:
            mock_request.return_value = incomplete_response

            entity = await api_kb.get_entity_info("test_entity")

            assert entity.id == "test_entity"
            assert entity.name == "test_entity"  # Default to ID
            assert entity.type == "Thing"  # Default type

        # Test with null values
        null_response = {
            "id": "test_entity",
            "name": None,
            "type": None,
            "attributes": None,
        }

        with patch.object(api_kb, "_make_request") as mock_request:
            mock_request.return_value = null_response

            entity = await api_kb.get_entity_info("test_entity")

            assert entity.id == "test_entity"
            assert entity.name == "test_entity"
            assert entity.type == "Thing"
            assert entity.attributes == {}

    @pytest.mark.asyncio
    async def test_api_kb_parse_retrieval_result_edge_cases(self, api_kb):
        """Test API knowledge base retrieval result parsing edge cases."""
        # Test with empty results
        empty_response = {"entities": [], "triples": []}

        with patch.object(api_kb, "_make_request") as mock_request:
            mock_request.return_value = empty_response

            result = await api_kb.get_entity_neighbors("test_entity")

            assert len(result.entities) == 0
            assert len(result.triples) == 0

        # Test with malformed triples
        malformed_response = {
            "entities": [
                {
                    "id": "entity1",
                    "name": "Entity 1",
                    "type": "Person",
                    "attributes": {},
                }
            ],
            "triples": [
                {
                    "subject": "entity1",
                    # Missing predicate and object
                }
            ],
        }

        with patch.object(api_kb, "_make_request") as mock_request:
            mock_request.return_value = malformed_response

            result = await api_kb.get_entity_neighbors("test_entity")

            assert len(result.entities) == 1
            assert len(result.triples) == 0  # Malformed triple should be skipped


class TestKnowledgeBaseIntegrationExtended:
    """Extended integration tests for knowledge bases."""

    @pytest.mark.asyncio
    async def test_sparql_kb_complex_query(self):
        """Test SPARQL knowledge base with complex query."""
        sparql_kb = SPARQLKnowledgeBase("http://test-endpoint.com/sparql")

        complex_query = """
        PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        
        SELECT ?entity ?name ?type WHERE {
            ?entity rdf:type ?type .
            ?entity rdfs:label ?name .
            FILTER(CONTAINS(LCASE(?name), "test"))
        }
        ORDER BY ?name
        LIMIT 10
        """

        mock_response = {
            "results": {
                "bindings": [
                    {
                        "entity": {"value": "http://example.org/entity1"},
                        "name": {"value": "Test Entity 1"},
                        "type": {"value": "http://example.org/Person"},
                    }
                ]
            }
        }

        with patch.object(sparql_kb, "_execute_query", return_value=mock_response):
            result = await sparql_kb.execute_query(complex_query)
            assert result == mock_response

    @pytest.mark.asyncio
    async def test_api_kb_complex_search(self):
        """Test API knowledge base with complex search."""
        api_kb = APIKnowledgeBase("https://api.test.com", "test_key")

        mock_response_data = {
            "results": [
                {
                    "id": "entity1",
                    "name": "Test Entity 1",
                    "type": "Person",
                    "attributes": {"age": 30, "location": "Test City"},
                }
            ],
            "facets": {
                "types": {"Person": 1, "Organization": 0},
                "locations": {"Test City": 1},
            },
            "pagination": {"page": 1, "per_page": 10, "total": 1, "has_next": False},
        }

        with patch.object(api_kb, "_make_request") as mock_request:
            mock_request.return_value = mock_response_data

            entities = await api_kb.search_entities(
                "test",
                limit=10,
                filters={"type": "Person", "location": "Test City"},
                facets=True,
            )

            assert len(entities) == 1
            assert entities[0].name == "Test Entity 1"

    @pytest.mark.asyncio
    async def test_knowledge_base_performance_metrics(self):
        """Test knowledge base performance metrics tracking."""
        sparql_kb = SPARQLKnowledgeBase("http://test-endpoint.com/sparql")

        mock_response = {"results": {"bindings": []}}

        with patch.object(sparql_kb, "_execute_query", return_value=mock_response):
            start_time = asyncio.get_event_loop().time()
            await sparql_kb.execute_query("SELECT * WHERE { ?s ?p ?o }")
            end_time = asyncio.get_event_loop().time()

            # Should complete within reasonable time
            assert (end_time - start_time) < 1.0  # Less than 1 second


# Test utilities for knowledge bases
def create_test_sparql_entity(
    entity_id: str, name: str, entity_type: str, attributes: dict = None
) -> Entity:
    """Create a test entity for SPARQL tests."""
    return Entity(
        id=entity_id,
        name=name,
        type=entity_type,
        attributes=attributes or {},
        neighbors=[],
    )


def create_test_api_entity(
    entity_id: str, name: str, entity_type: str, attributes: dict = None
) -> Entity:
    """Create a test entity for API tests."""
    return Entity(
        id=entity_id,
        name=name,
        type=entity_type,
        attributes=attributes or {},
        neighbors=[],
    )


def create_test_knowledge_base_result(
    entities: list, triples: list, metadata: dict = None
) -> RetrievalResult:
    """Create a test retrieval result for knowledge base tests."""
    return RetrievalResult(entities=entities, triples=triples, metadata=metadata or {})


# Pytest configuration
@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()
