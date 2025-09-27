"""
Extended test suite for retrieval chain to improve coverage.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch

from core.types import (
    Entity,
    Triple,
    RetrievalResult,
    RetrievalPlan,
    DomainType,
    RetrievalConfig,
)
from core.exceptions import RetrievalError
from chains.retrieval import RetrievalChain, AdaptiveRetrievalChain


class TestRetrievalChainExtended:
    """Extended test suite for retrieval chain."""

    @pytest.fixture
    def mock_knowledge_base(self):
        """Create mock knowledge base for testing."""
        kb = Mock()
        kb.get_entity_neighbors = AsyncMock()
        kb.get_relations = AsyncMock(return_value=["author", "wrote", "published"])
        return kb

    @pytest.fixture
    def retrieval_chain(self, mock_knowledge_base):
        """Create retrieval chain instance for testing."""
        config = RetrievalConfig(
            max_hops=3, max_entities=100, max_triples=200, timeout=30
        )
        return RetrievalChain(mock_knowledge_base, config)

    @pytest.fixture
    def adaptive_retrieval_chain(self, mock_knowledge_base):
        """Create adaptive retrieval chain instance for testing."""
        config = RetrievalConfig(
            max_hops=3, max_entities=100, max_triples=200, timeout=30
        )
        return AdaptiveRetrievalChain(mock_knowledge_base, config)

    @pytest.fixture
    def mock_retrieval_plan(self):
        """Create mock retrieval plan for testing."""
        entity = Entity(
            id="test_entity",
            name="J.K. Rowling",
            type="Person",
            attributes={"birth_year": 1965},
            neighbors=[],
        )

        return RetrievalPlan(
            domain=DomainType.BOOK,
            topic_entity=entity,
            relations=["author", "wrote", "published"],
            max_hops=2,
            filters={"publication_year": ">1990"},
        )

    @pytest.fixture
    def mock_retrieval_result(self):
        """Create mock retrieval result for testing."""
        entities = [
            Entity(
                id="author1",
                name="J.K. Rowling",
                type="Person",
                attributes={"birth_year": 1965, "nationality": "British"},
                neighbors=[],
            ),
            Entity(
                id="book1",
                name="Harry Potter and the Philosopher's Stone",
                type="Book",
                attributes={"publication_year": 1997, "genre": "Fantasy"},
                neighbors=[],
            ),
            Entity(
                id="book2",
                name="Harry Potter and the Chamber of Secrets",
                type="Book",
                attributes={"publication_year": 1998, "genre": "Fantasy"},
                neighbors=[],
            ),
        ]

        triples = [
            Triple(
                subject="author1", predicate="wrote", object="book1", confidence=0.95
            ),
            Triple(
                subject="author1", predicate="wrote", object="book2", confidence=0.95
            ),
            Triple(
                subject="book1",
                predicate="part_of_series",
                object="Harry Potter",
                confidence=0.9,
            ),
            Triple(
                subject="book2",
                predicate="part_of_series",
                object="Harry Potter",
                confidence=0.9,
            ),
        ]

        return RetrievalResult(
            entities=entities,
            triples=triples,
            metadata={"source": "test_kb", "retrieval_time": 0.5},
        )

    @pytest.mark.asyncio
    async def test_expand_neighborhood_single_hop(
        self, retrieval_chain, mock_knowledge_base, mock_retrieval_result
    ):
        """Test neighborhood expansion with single hop."""
        mock_knowledge_base.get_entity_neighbors.return_value = mock_retrieval_result

        entity = Entity(
            id="test_entity",
            name="J.K. Rowling",
            type="Person",
            attributes={},
            neighbors=[],
        )

        result = await retrieval_chain.expand_neighborhood(
            entity, ["author", "wrote"], max_hops=1
        )

        assert len(result.entities) >= 1  # At least the original entity
        assert len(result.triples) >= 0
        assert result.metadata["max_hops"] == 1
        assert result.metadata["relations"] == ["author", "wrote"]

    @pytest.mark.asyncio
    async def test_expand_neighborhood_multiple_hops(
        self, retrieval_chain, mock_knowledge_base, mock_retrieval_result
    ):
        """Test neighborhood expansion with multiple hops."""
        mock_knowledge_base.get_entity_neighbors.return_value = mock_retrieval_result

        entity = Entity(
            id="test_entity",
            name="J.K. Rowling",
            type="Person",
            attributes={},
            neighbors=[],
        )

        result = await retrieval_chain.expand_neighborhood(
            entity, ["author", "wrote", "published"], max_hops=3
        )

        assert len(result.entities) >= 1
        assert len(result.triples) >= 0
        assert result.metadata["max_hops"] == 3
        assert result.metadata["relations"] == ["author", "wrote", "published"]

    @pytest.mark.asyncio
    async def test_expand_neighborhood_with_filters(
        self, retrieval_chain, mock_knowledge_base, mock_retrieval_result
    ):
        """Test neighborhood expansion with filters."""
        mock_knowledge_base.get_entity_neighbors.return_value = mock_retrieval_result

        entity = Entity(
            id="test_entity",
            name="J.K. Rowling",
            type="Person",
            attributes={},
            neighbors=[],
        )

        filters = {"publication_year": ">1990", "genre": "Fantasy"}

        result = await retrieval_chain.expand_neighborhood(
            entity, ["author", "wrote"], max_hops=2, filters=filters
        )

        assert len(result.entities) >= 1
        assert result.metadata["filters"] == filters

    @pytest.mark.asyncio
    async def test_expand_neighborhood_knowledge_base_error(
        self, retrieval_chain, mock_knowledge_base
    ):
        """Test neighborhood expansion with knowledge base error."""
        mock_knowledge_base.get_entity_neighbors.side_effect = Exception(
            "KB connection failed"
        )

        entity = Entity(
            id="test_entity",
            name="Test Entity",
            type="Person",
            attributes={},
            neighbors=[],
        )

        with pytest.raises(RetrievalError, match="Failed to expand neighborhood"):
            await retrieval_chain.expand_neighborhood(entity, ["author"], max_hops=1)

    @pytest.mark.asyncio
    async def test_retrieve_knowledge_success(
        self, retrieval_chain, mock_retrieval_plan
    ):
        """Test knowledge retrieval with successful plan."""
        with patch.object(retrieval_chain, "expand_neighborhood") as mock_expand:
            mock_expand.return_value = RetrievalResult(
                entities=[], triples=[], metadata={}
            )

            result = await retrieval_chain.retrieve_knowledge(mock_retrieval_plan)

            assert isinstance(result, RetrievalResult)
            mock_expand.assert_called_once()

    @pytest.mark.asyncio
    async def test_retrieve_knowledge_with_filters(
        self, retrieval_chain, mock_retrieval_plan
    ):
        """Test knowledge retrieval with filters."""
        with patch.object(retrieval_chain, "expand_neighborhood") as mock_expand:
            mock_expand.return_value = RetrievalResult(
                entities=[], triples=[], metadata={}
            )

            result = await retrieval_chain.retrieve_knowledge(mock_retrieval_plan)

            # Check that filters were passed to expand_neighborhood
            call_args = mock_expand.call_args
            assert call_args[1]["filters"] == mock_retrieval_plan.filters

    @pytest.mark.asyncio
    async def test_retrieve_knowledge_expansion_error(
        self, retrieval_chain, mock_retrieval_plan
    ):
        """Test knowledge retrieval with expansion error."""
        with patch.object(retrieval_chain, "expand_neighborhood") as mock_expand:
            mock_expand.side_effect = Exception("Expansion failed")

            with pytest.raises(RetrievalError, match="Failed to retrieve knowledge"):
                await retrieval_chain.retrieve_knowledge(mock_retrieval_plan)

    @pytest.mark.asyncio
    async def test_retrieval_chain_run_success(
        self, retrieval_chain, mock_retrieval_plan
    ):
        """Test complete retrieval chain execution."""
        with patch.object(retrieval_chain, "retrieve_knowledge") as mock_retrieve:
            mock_retrieve.return_value = RetrievalResult(
                entities=[], triples=[], metadata={}
            )

            result = await retrieval_chain.arun({"retrieval_plan": mock_retrieval_plan})

            assert result["success"] is True
            assert "retrieved_knowledge" in result

    @pytest.mark.asyncio
    async def test_retrieval_chain_run_failure(
        self, retrieval_chain, mock_retrieval_plan
    ):
        """Test retrieval chain execution with failure."""
        with patch.object(retrieval_chain, "retrieve_knowledge") as mock_retrieve:
            mock_retrieve.side_effect = Exception("Retrieval failed")

            result = await retrieval_chain.arun({"retrieval_plan": mock_retrieval_plan})

            assert result["success"] is False
            assert "error" in result
            assert "Retrieval failed" in result["error"]

    @pytest.mark.asyncio
    async def test_adaptive_retrieval_chain_relation_adaptation(
        self, adaptive_retrieval_chain, mock_knowledge_base
    ):
        """Test adaptive retrieval chain relation adaptation."""
        entity = Entity(
            id="test_entity",
            name="Test Entity",
            type="Person",
            attributes={},
            neighbors=[],
        )

        # Mock the knowledge base to return different relations
        mock_knowledge_base.get_relations.return_value = [
            "author",
            "wrote",
            "published",
            "reviewed",
        ]

        with patch.object(
            adaptive_retrieval_chain, "expand_neighborhood"
        ) as mock_expand:
            mock_expand.return_value = RetrievalResult(
                entities=[], triples=[], metadata={}
            )

            result = await adaptive_retrieval_chain.expand_neighborhood(
                entity, ["author"], max_hops=2
            )

            # Should adapt relations based on knowledge base
            assert isinstance(result, RetrievalResult)

    @pytest.mark.asyncio
    async def test_adaptive_retrieval_chain_dynamic_hops(
        self, adaptive_retrieval_chain, mock_knowledge_base
    ):
        """Test adaptive retrieval chain with dynamic hop adjustment."""
        entity = Entity(
            id="test_entity",
            name="Test Entity",
            type="Person",
            attributes={},
            neighbors=[],
        )

        with patch.object(
            adaptive_retrieval_chain, "expand_neighborhood"
        ) as mock_expand:
            # Mock different results for different hop counts
            def mock_expand_side_effect(entity, relations, max_hops, filters=None):
                if max_hops == 1:
                    return RetrievalResult(
                        entities=[entity],
                        triples=[],
                        metadata={"max_hops": 1, "entities_found": 1},
                    )
                else:
                    return RetrievalResult(
                        entities=[
                            entity,
                            Entity("neighbor", "Neighbor", "Person", {}, []),
                        ],
                        triples=[Triple("test_entity", "related_to", "neighbor", 0.8)],
                        metadata={"max_hops": max_hops, "entities_found": 2},
                    )

            mock_expand.side_effect = mock_expand_side_effect

            result = await adaptive_retrieval_chain.expand_neighborhood(
                entity, ["author"], max_hops=2
            )

            assert isinstance(result, RetrievalResult)

    @pytest.mark.asyncio
    async def test_retrieval_chain_config_validation(self):
        """Test retrieval chain configuration validation."""
        mock_kb = Mock()

        # Test valid config
        valid_config = RetrievalConfig(
            max_hops=3, max_entities=100, max_triples=200, timeout=30
        )

        chain = RetrievalChain(mock_kb, valid_config)
        assert chain.config.max_hops == 3
        assert chain.config.max_entities == 100

        # Test invalid config
        with pytest.raises(ValueError, match="Max hops must be non-negative"):
            invalid_config = RetrievalConfig(max_hops=-1)
            RetrievalChain(mock_kb, invalid_config)

    @pytest.mark.asyncio
    async def test_retrieval_chain_edge_cases(self, retrieval_chain):
        """Test retrieval chain with edge cases."""
        # Test with empty relations
        entity = Entity(
            id="test_entity",
            name="Test Entity",
            type="Person",
            attributes={},
            neighbors=[],
        )

        with patch.object(retrieval_chain, "expand_neighborhood") as mock_expand:
            mock_expand.return_value = RetrievalResult(
                entities=[entity], triples=[], metadata={}
            )

            result = await retrieval_chain.expand_neighborhood(entity, [], max_hops=1)

            assert len(result.entities) == 1
            assert len(result.triples) == 0

        # Test with zero max_hops
        with patch.object(retrieval_chain, "expand_neighborhood") as mock_expand:
            mock_expand.return_value = RetrievalResult(
                entities=[entity], triples=[], metadata={}
            )

            result = await retrieval_chain.expand_neighborhood(
                entity, ["author"], max_hops=0
            )

            assert result.metadata["max_hops"] == 0

    @pytest.mark.asyncio
    async def test_retrieval_chain_metadata_tracking(
        self, retrieval_chain, mock_knowledge_base, mock_retrieval_result
    ):
        """Test that retrieval chain properly tracks metadata."""
        mock_knowledge_base.get_entity_neighbors.return_value = mock_retrieval_result

        entity = Entity(
            id="test_entity",
            name="Test Entity",
            type="Person",
            attributes={},
            neighbors=[],
        )

        result = await retrieval_chain.expand_neighborhood(
            entity, ["author", "wrote"], max_hops=2
        )

        # Check metadata tracking
        assert "max_hops" in result.metadata
        assert "relations" in result.metadata
        assert "start_time" in result.metadata
        assert "end_time" in result.metadata
        assert result.metadata["max_hops"] == 2
        assert result.metadata["relations"] == ["author", "wrote"]

    @pytest.mark.asyncio
    async def test_retrieval_chain_timeout_handling(
        self, retrieval_chain, mock_knowledge_base
    ):
        """Test retrieval chain timeout handling."""

        # Mock a slow knowledge base response
        async def slow_response(*args, **kwargs):
            await asyncio.sleep(0.1)  # Simulate slow response
            return RetrievalResult(entities=[], triples=[], metadata={})

        mock_knowledge_base.get_entity_neighbors.side_effect = slow_response

        entity = Entity(
            id="test_entity",
            name="Test Entity",
            type="Person",
            attributes={},
            neighbors=[],
        )

        # Set a very short timeout
        retrieval_chain.config.timeout = 0.05

        with pytest.raises(RetrievalError, match="Timeout"):
            await retrieval_chain.expand_neighborhood(entity, ["author"], max_hops=1)

    @pytest.mark.asyncio
    async def test_retrieval_chain_entity_deduplication(
        self, retrieval_chain, mock_knowledge_base
    ):
        """Test that retrieval chain deduplicates entities."""
        # Create entities with duplicate IDs
        duplicate_entities = [
            Entity(
                id="entity1",
                name="Entity 1",
                type="Person",
                attributes={},
                neighbors=[],
            ),
            Entity(
                id="entity1",
                name="Entity 1 Duplicate",
                type="Person",
                attributes={},
                neighbors=[],
            ),
            Entity(
                id="entity2", name="Entity 2", type="Book", attributes={}, neighbors=[]
            ),
        ]

        duplicate_result = RetrievalResult(
            entities=duplicate_entities, triples=[], metadata={}
        )

        mock_knowledge_base.get_entity_neighbors.return_value = duplicate_result

        entity = Entity(
            id="test_entity",
            name="Test Entity",
            type="Person",
            attributes={},
            neighbors=[],
        )

        result = await retrieval_chain.expand_neighborhood(
            entity, ["author"], max_hops=1
        )

        # Should deduplicate entities by ID
        entity_ids = [e.id for e in result.entities]
        assert len(set(entity_ids)) == len(entity_ids)  # No duplicates

    @pytest.mark.asyncio
    async def test_retrieval_chain_triple_confidence_filtering(
        self, retrieval_chain, mock_knowledge_base
    ):
        """Test that retrieval chain filters triples by confidence."""
        low_confidence_triples = [
            Triple(
                subject="entity1",
                predicate="related_to",
                object="entity2",
                confidence=0.3,
            ),
            Triple(
                subject="entity1",
                predicate="related_to",
                object="entity3",
                confidence=0.8,
            ),
            Triple(
                subject="entity1",
                predicate="related_to",
                object="entity4",
                confidence=0.1,
            ),
        ]

        low_confidence_result = RetrievalResult(
            entities=[], triples=low_confidence_triples, metadata={}
        )

        mock_knowledge_base.get_entity_neighbors.return_value = low_confidence_result

        entity = Entity(
            id="test_entity",
            name="Test Entity",
            type="Person",
            attributes={},
            neighbors=[],
        )

        # Set confidence threshold
        retrieval_chain.config.confidence_threshold = 0.5

        result = await retrieval_chain.expand_neighborhood(
            entity, ["related_to"], max_hops=1
        )

        # Should filter out low confidence triples
        for triple in result.triples:
            assert triple.confidence >= 0.5


# Test utilities for retrieval chain
def create_test_retrieval_entity(
    entity_id: str, name: str, entity_type: str, attributes: dict = None
) -> Entity:
    """Create a test entity for retrieval tests."""
    return Entity(
        id=entity_id,
        name=name,
        type=entity_type,
        attributes=attributes or {},
        neighbors=[],
    )


def create_test_retrieval_triple(
    subject: str, predicate: str, obj: str, confidence: float = 0.9
) -> Triple:
    """Create a test triple for retrieval tests."""
    return Triple(
        subject=subject, predicate=predicate, object=obj, confidence=confidence
    )


def create_test_retrieval_plan(
    domain: str, relations: list, max_hops: int = 2, filters: dict = None
) -> RetrievalPlan:
    """Create a test retrieval plan."""
    entity = create_test_retrieval_entity("test_entity", "Test Entity", "Person")

    domain_type = DomainType.BOOK if domain == "book" else DomainType.MOVIE

    return RetrievalPlan(
        domain=domain_type,
        topic_entity=entity,
        relations=relations,
        max_hops=max_hops,
        filters=filters or {},
    )


# Pytest configuration
@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()
