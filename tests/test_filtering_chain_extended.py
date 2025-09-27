"""
Extended test suite for filtering chain to improve coverage.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch

from core.types import Entity, Triple, RetrievalResult, FilteringConfig
from core.exceptions import FilteringError
from chains.filtering import FilteringChain, AdvancedFilteringChain


class TestFilteringChainExtended:
    """Extended test suite for filtering chain."""

    @pytest.fixture
    def filtering_chain(self):
        """Create filtering chain instance for testing."""
        config = FilteringConfig(
            relevance_threshold=0.5,
            max_triples=100,
            max_entities=50,
            use_semantic_filtering=True,
        )
        with patch("langchain_openai.ChatOpenAI") as mock_openai:
            mock_openai.return_value = Mock()
            return FilteringChain(llm_model="gpt-4", api_key="test_key", config=config)

    @pytest.fixture
    def advanced_filtering_chain(self):
        """Create advanced filtering chain instance for testing."""
        config = FilteringConfig(
            relevance_threshold=0.5,
            max_triples=100,
            use_semantic_filtering=True,
            use_graph_based_filtering=True,
        )
        with patch("langchain_openai.ChatOpenAI") as mock_openai:
            mock_openai.return_value = Mock()
            return AdvancedFilteringChain(
                llm_model="gpt-4", api_key="test_key", config=config
            )

    @pytest.fixture
    def mock_retrieval_result(self):
        """Create mock retrieval result for filtering."""
        entities = [
            Entity(
                id="entity1",
                name="J.K. Rowling",
                type="Person",
                attributes={"birth_year": 1965, "nationality": "British"},
                neighbors=[],
            ),
            Entity(
                id="entity2",
                name="Harry Potter and the Philosopher's Stone",
                type="Book",
                attributes={"publication_year": 1997, "genre": "Fantasy"},
                neighbors=[],
            ),
            Entity(
                id="entity3",
                name="Fantasy Literature",
                type="Genre",
                attributes={"description": "A genre of fiction"},
                neighbors=[],
            ),
            Entity(
                id="entity4",
                name="Unrelated Entity",
                type="Organization",
                attributes={"type": "company"},
                neighbors=[],
            ),
        ]

        triples = [
            Triple(
                subject="entity1",
                predicate="wrote",
                object="entity2",
                confidence=0.95,
            ),
            Triple(
                subject="entity2",
                predicate="belongs_to_genre",
                object="entity3",
                confidence=0.9,
            ),
            Triple(
                subject="entity1",
                predicate="born_in",
                object="United Kingdom",
                confidence=0.8,
            ),
            Triple(
                subject="entity4",
                predicate="unrelated_to",
                object="entity1",
                confidence=0.2,
            ),
        ]

        return RetrievalResult(
            entities=entities,
            triples=triples,
            metadata={"source": "test_kb", "retrieval_time": 0.5},
        )

    @pytest.mark.asyncio
    async def test_compute_relevance_scores_comprehensive(
        self, filtering_chain, mock_retrieval_result
    ):
        """Test comprehensive relevance score computation."""
        entities = mock_retrieval_result.entities
        triples = mock_retrieval_result.triples

        mock_llm_response = {
            "entity_scores": {
                "entity1": 0.95,  # J.K. Rowling - highly relevant
                "entity2": 0.9,  # Harry Potter book - highly relevant
                "entity3": 0.7,  # Fantasy genre - moderately relevant
                "entity4": 0.1,  # Unrelated entity - low relevance
            },
            "triple_scores": {
                "0": 0.95,  # wrote relationship - highly relevant
                "1": 0.8,  # genre relationship - moderately relevant
                "2": 0.6,  # born_in relationship - somewhat relevant
                "3": 0.1,  # unrelated relationship - low relevance
            },
            "reasoning": "Entities and relationships related to J.K. Rowling and her books are highly relevant to the question about her works.",
        }

        with patch.object(filtering_chain, "relevance_chain") as mock_chain:
            mock_chain.ainvoke = AsyncMock(return_value=mock_llm_response)

            scores = await filtering_chain.compute_relevance_scores(
                "What books did J.K. Rowling write?", entities, triples
            )

            assert scores["entity_entity1"] == 0.95
            assert scores["entity_entity2"] == 0.9
            assert scores["entity_entity3"] == 0.7
            assert scores["entity_entity4"] == 0.1
            assert scores["triple_0"] == 0.95
            assert scores["triple_1"] == 0.8
            assert scores["triple_2"] == 0.6
            assert scores["triple_3"] == 0.1

    @pytest.mark.asyncio
    async def test_filter_knowledge_with_threshold(
        self, filtering_chain, mock_retrieval_result
    ):
        """Test knowledge filtering with relevance threshold."""
        with patch.object(filtering_chain, "compute_relevance_scores") as mock_scores:
            mock_scores.return_value = {
                "entity_entity1": 0.95,
                "entity_entity2": 0.9,
                "entity_entity3": 0.3,  # Below threshold
                "entity_entity4": 0.1,  # Below threshold
                "triple_0": 0.95,
                "triple_1": 0.8,
                "triple_2": 0.3,  # Below threshold
                "triple_3": 0.1,  # Below threshold
            }

            result = await filtering_chain.filter_knowledge(
                "What books did J.K. Rowling write?", mock_retrieval_result
            )

            # Should filter out low-relevance items
            assert len(result.entities) == 2  # Only entity1 and entity2 should remain
            assert len(result.triples) == 2  # Only triple 0 and 1 should remain

            # Check that the correct entities remain
            entity_names = [entity.name for entity in result.entities]
            assert "J.K. Rowling" in entity_names
            assert "Harry Potter and the Philosopher's Stone" in entity_names
            assert "Unrelated Entity" not in entity_names

    @pytest.mark.asyncio
    async def test_filter_knowledge_with_max_limits(
        self, filtering_chain, mock_retrieval_result
    ):
        """Test knowledge filtering with maximum limits."""
        # Create a large retrieval result
        large_entities = []
        large_triples = []

        for i in range(100):
            large_entities.append(
                Entity(
                    id=f"entity{i}",
                    name=f"Entity {i}",
                    type="Person",
                    attributes={},
                    neighbors=[],
                )
            )
            large_triples.append(
                Triple(
                    subject=f"entity{i}",
                    predicate="related_to",
                    object=f"entity{i+1}",
                    confidence=0.8,
                )
            )

        large_result = RetrievalResult(
            entities=large_entities, triples=large_triples, metadata={}
        )

        with patch.object(filtering_chain, "compute_relevance_scores") as mock_scores:
            # All items have high relevance
            scores = {}
            for i in range(100):
                scores[f"entity_entity{i}"] = 0.9
                scores[f"triple_{i}"] = 0.9
            mock_scores.return_value = scores

            result = await filtering_chain.filter_knowledge(
                "Test question", large_result
            )

            # Should respect max limits
            assert len(result.entities) <= filtering_chain.config.max_entities
            assert len(result.triples) <= filtering_chain.config.max_triples

    @pytest.mark.asyncio
    async def test_filter_knowledge_empty_result(self, filtering_chain):
        """Test knowledge filtering with empty retrieval result."""
        empty_result = RetrievalResult(entities=[], triples=[], metadata={})

        with patch.object(filtering_chain, "compute_relevance_scores") as mock_scores:
            mock_scores.return_value = {}

            result = await filtering_chain.filter_knowledge(
                "Test question", empty_result
            )

            assert len(result.entities) == 0
            assert len(result.triples) == 0

    @pytest.mark.asyncio
    async def test_filtering_chain_run_success(
        self, filtering_chain, mock_retrieval_result
    ):
        """Test complete filtering chain execution."""
        with patch.object(filtering_chain, "compute_relevance_scores") as mock_scores:
            mock_scores.return_value = {
                "entity_entity1": 0.9,
                "entity_entity2": 0.8,
                "triple_0": 0.9,
                "triple_1": 0.8,
            }

            result = await filtering_chain.arun(
                {
                    "question": "What books did J.K. Rowling write?",
                    "retrieved_knowledge": mock_retrieval_result,
                }
            )

            assert result["success"] is True
            assert "filtered_knowledge" in result
            assert len(result["filtered_knowledge"].entities) > 0

    @pytest.mark.asyncio
    async def test_filtering_chain_run_failure(
        self, filtering_chain, mock_retrieval_result
    ):
        """Test filtering chain execution with failure."""
        with patch.object(filtering_chain, "compute_relevance_scores") as mock_scores:
            mock_scores.side_effect = Exception("LLM connection failed")

            result = await filtering_chain.arun(
                {
                    "question": "What books did J.K. Rowling write?",
                    "retrieved_knowledge": mock_retrieval_result,
                }
            )

            assert result["success"] is False
            assert "error" in result
            assert "LLM connection failed" in result["error"]

    @pytest.mark.asyncio
    async def test_advanced_filtering_chain_semantic_filtering(
        self, advanced_filtering_chain, mock_retrieval_result
    ):
        """Test advanced filtering chain with semantic filtering."""
        with patch.object(
            advanced_filtering_chain, "_semantic_filtering"
        ) as mock_semantic:
            mock_semantic.return_value = mock_retrieval_result

            result = await advanced_filtering_chain.filter_knowledge(
                "What books did J.K. Rowling write?", mock_retrieval_result
            )

            assert isinstance(result, RetrievalResult)
            mock_semantic.assert_called_once()

    @pytest.mark.asyncio
    async def test_advanced_filtering_chain_graph_based_filtering(
        self, advanced_filtering_chain, mock_retrieval_result
    ):
        """Test advanced filtering chain with graph-based filtering."""
        with patch.object(
            advanced_filtering_chain, "_graph_based_filtering"
        ) as mock_graph:
            mock_graph.return_value = mock_retrieval_result

            result = await advanced_filtering_chain.filter_knowledge(
                "What books did J.K. Rowling write?", mock_retrieval_result
            )

            assert isinstance(result, RetrievalResult)
            mock_graph.assert_called_once()

    @pytest.mark.asyncio
    async def test_advanced_filtering_chain_hybrid_filtering(
        self, advanced_filtering_chain, mock_retrieval_result
    ):
        """Test advanced filtering chain with hybrid filtering strategy."""
        with patch.object(
            advanced_filtering_chain, "_semantic_filtering"
        ) as mock_semantic, patch.object(
            advanced_filtering_chain, "_graph_based_filtering"
        ) as mock_graph, patch.object(
            advanced_filtering_chain, "_hybrid_filtering"
        ) as mock_hybrid:

            mock_semantic.return_value = mock_retrieval_result
            mock_graph.return_value = mock_retrieval_result
            mock_hybrid.return_value = mock_retrieval_result

            # Set config to use hybrid strategy
            advanced_filtering_chain.config.strategy = "hybrid"

            result = await advanced_filtering_chain.filter_knowledge(
                "What books did J.K. Rowling write?", mock_retrieval_result
            )

            assert isinstance(result, RetrievalResult)
            mock_hybrid.assert_called_once()

    @pytest.mark.asyncio
    async def test_filtering_chain_initialization_different_llms(self):
        """Test filtering chain initialization with different LLM models."""
        config = FilteringConfig(relevance_threshold=0.5, max_triples=100)

        # Test Claude
        with patch("langchain_anthropic.ChatAnthropic") as mock_claude:
            mock_claude.return_value = Mock()
            chain = FilteringChain(
                llm_model="claude-3-haiku", api_key="test_key", config=config
            )
            assert chain.llm_model == "claude-3-haiku"

        # Test Gemini
        with patch("langchain_google_genai.ChatGoogleGenerativeAI") as mock_gemini:
            mock_gemini.return_value = Mock()
            chain = FilteringChain(
                llm_model="gemini-1.5-pro", api_key="test_key", config=config
            )
            assert chain.llm_model == "gemini-1.5-pro"

        # Test invalid model
        with pytest.raises(ValueError, match="Unsupported LLM model"):
            FilteringChain(llm_model="invalid-model", api_key="test_key", config=config)

    @pytest.mark.asyncio
    async def test_filtering_chain_edge_cases(self, filtering_chain):
        """Test filtering chain with edge cases."""
        # Test with empty question
        empty_result = RetrievalResult(entities=[], triples=[], metadata={})

        with patch.object(filtering_chain, "compute_relevance_scores") as mock_scores:
            mock_scores.return_value = {}

            result = await filtering_chain.filter_knowledge("", empty_result)
            assert len(result.entities) == 0
            assert len(result.triples) == 0

        # Test with very long question
        long_question = "What books did " + "J.K. Rowling " * 100 + "write?"

        with patch.object(filtering_chain, "compute_relevance_scores") as mock_scores:
            mock_scores.return_value = {}

            result = await filtering_chain.filter_knowledge(long_question, empty_result)
            assert isinstance(result, RetrievalResult)

    @pytest.mark.asyncio
    async def test_filtering_chain_config_validation(self):
        """Test filtering chain configuration validation."""
        # Test valid config
        valid_config = FilteringConfig(
            relevance_threshold=0.5,
            max_triples=100,
            max_entities=50,
            use_semantic_filtering=True,
        )

        with patch("langchain_openai.ChatOpenAI") as mock_openai:
            mock_openai.return_value = Mock()
            chain = FilteringChain(
                llm_model="gpt-4", api_key="test_key", config=valid_config
            )
            assert chain.config.relevance_threshold == 0.5
            assert chain.config.max_triples == 100

        # Test invalid config
        with pytest.raises(ValueError, match="Relevance threshold must be between"):
            invalid_config = FilteringConfig(relevance_threshold=1.5)  # > 1.0
            FilteringChain(llm_model="gpt-4", api_key="test_key", config=invalid_config)

    @pytest.mark.asyncio
    async def test_filtering_chain_chain_creation(self, filtering_chain):
        """Test that filtering chains are properly created."""
        # Test chain creation
        assert hasattr(filtering_chain, "relevance_chain")

        # Test that chains are callable
        assert callable(filtering_chain.relevance_chain.ainvoke)

    @pytest.mark.asyncio
    async def test_compute_relevance_scores_malformed_response(
        self, filtering_chain, mock_retrieval_result
    ):
        """Test relevance score computation with malformed LLM response."""
        entities = mock_retrieval_result.entities
        triples = mock_retrieval_result.triples

        mock_llm_response = {
            "invalid_field": "value",
            # Missing required fields
        }

        with patch.object(filtering_chain, "relevance_chain") as mock_chain:
            mock_chain.ainvoke = AsyncMock(return_value=mock_llm_response)

            with pytest.raises(FilteringError, match="Invalid LLM response"):
                await filtering_chain.compute_relevance_scores(
                    "Test question", entities, triples
                )

    @pytest.mark.asyncio
    async def test_filter_knowledge_preserves_metadata(
        self, filtering_chain, mock_retrieval_result
    ):
        """Test that filtering preserves metadata."""
        with patch.object(filtering_chain, "compute_relevance_scores") as mock_scores:
            mock_scores.return_value = {
                "entity_entity1": 0.9,
                "entity_entity2": 0.8,
                "triple_0": 0.9,
                "triple_1": 0.8,
            }

            result = await filtering_chain.filter_knowledge(
                "Test question", mock_retrieval_result
            )

            # Check that metadata is preserved
            assert "source" in result.metadata
            assert "retrieval_time" in result.metadata
            assert result.metadata["source"] == "test_kb"


# Test utilities for filtering chain
def create_test_filtering_entity(
    entity_id: str, name: str, entity_type: str, attributes: dict = None
) -> Entity:
    """Create a test entity for filtering tests."""
    return Entity(
        id=entity_id,
        name=name,
        type=entity_type,
        attributes=attributes or {},
        neighbors=[],
    )


def create_test_filtering_triple(
    subject: str, predicate: str, obj: str, confidence: float = 0.9
) -> Triple:
    """Create a test triple for filtering tests."""
    return Triple(
        subject=subject, predicate=predicate, object=obj, confidence=confidence
    )


def create_test_filtering_result(
    entities: list, triples: list, metadata: dict = None
) -> RetrievalResult:
    """Create a test retrieval result for filtering tests."""
    return RetrievalResult(entities=entities, triples=triples, metadata=metadata or {})


# Pytest configuration
@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()
