"""
Extended test suite for planning chain to improve coverage.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch

from core.types import Entity, DomainType, RetrievalPlan
from core.exceptions import PlanningError
from chains.planning import PlanningChain, AdvancedPlanningChain


class TestPlanningChainExtended:
    """Extended test suite for planning chain."""

    @pytest.fixture
    def planning_chain(self):
        """Create planning chain instance for testing."""
        with patch("langchain_openai.ChatOpenAI") as mock_openai:
            mock_openai.return_value = Mock()
            return PlanningChain(llm_model="gpt-4", api_key="test_key", temperature=0.1)

    @pytest.fixture
    def advanced_planning_chain(self):
        """Create advanced planning chain instance for testing."""
        mock_kb = Mock()
        mock_kb.search_entities = AsyncMock(return_value=[])

        with patch("langchain_openai.ChatOpenAI") as mock_openai:
            mock_openai.return_value = Mock()
            return AdvancedPlanningChain(
                llm_model="gpt-4", api_key="test_key", knowledge_base=mock_kb
            )

    @pytest.mark.asyncio
    async def test_identify_topic_entity_with_confidence_threshold(
        self, planning_chain
    ):
        """Test topic entity identification with low confidence."""
        mock_llm_response = {
            "entity_name": "Uncertain Entity",
            "entity_type": "Person",
            "confidence": 0.3,  # Low confidence
        }

        with patch.object(planning_chain, "entity_chain") as mock_chain:
            mock_chain.ainvoke = AsyncMock(return_value=mock_llm_response)

            with pytest.raises(PlanningError, match="Low confidence"):
                await planning_chain.identify_topic_entity("Uncertain question")

    @pytest.mark.asyncio
    async def test_identify_topic_entity_malformed_response(self, planning_chain):
        """Test topic entity identification with malformed LLM response."""
        mock_llm_response = {
            "invalid_field": "value",
            # Missing required fields
        }

        with patch.object(planning_chain, "entity_chain") as mock_chain:
            mock_chain.ainvoke = AsyncMock(return_value=mock_llm_response)

            with pytest.raises(PlanningError):
                await planning_chain.identify_topic_entity("Test question")

    @pytest.mark.asyncio
    async def test_detect_domain_with_multiple_domains(self, planning_chain):
        """Test domain detection with multiple possible domains."""
        mock_entity = Entity(
            id="test_entity",
            name="Test Entity",
            type="Person",
            attributes={"confidence": 0.9},
            neighbors=[],
        )

        mock_domain_response = {
            "domain": "mixed",
            "confidence": 0.7,
            "reasoning": "Question spans multiple domains",
        }

        with patch.object(planning_chain, "domain_chain") as mock_chain:
            mock_chain.ainvoke = AsyncMock(return_value=mock_domain_response)

            domain = await planning_chain.detect_domain(
                "Complex question about books and movies", mock_entity
            )

            assert domain == "mixed"

    @pytest.mark.asyncio
    async def test_detect_domain_low_confidence(self, planning_chain):
        """Test domain detection with low confidence."""
        mock_entity = Entity(
            id="test_entity",
            name="Test Entity",
            type="Person",
            attributes={"confidence": 0.9},
            neighbors=[],
        )

        mock_domain_response = {
            "domain": "unknown",
            "confidence": 0.2,  # Low confidence
            "reasoning": "Cannot determine domain",
        }

        with patch.object(planning_chain, "domain_chain") as mock_chain:
            mock_chain.ainvoke = AsyncMock(return_value=mock_domain_response)

            with pytest.raises(PlanningError, match="Low confidence"):
                await planning_chain.detect_domain("Unclear question", mock_entity)

    @pytest.mark.asyncio
    async def test_generate_retrieval_plan_complex_relations(self, planning_chain):
        """Test retrieval plan generation with complex relations."""
        mock_entity = Entity(
            id="test_entity",
            name="Test Entity",
            type="Person",
            attributes={"confidence": 0.9},
            neighbors=[],
        )

        mock_plan_response = {
            "relations": ["author", "wrote", "published", "reviewed", "influenced"],
            "max_hops": 3,
            "filters": {"type": "specific", "date_range": "recent"},
            "reasoning": "Need to explore multiple relationship types",
        }

        with patch.object(planning_chain, "retrieval_planning_chain") as mock_chain:
            mock_chain.ainvoke = AsyncMock(return_value=mock_plan_response)

            plan = await planning_chain.generate_retrieval_plan(
                "Complex question about author relationships", mock_entity, "book"
            )

            assert plan.domain == DomainType.BOOK
            assert len(plan.relations) == 5
            assert plan.max_hops == 3
            assert "date_range" in plan.filters

    @pytest.mark.asyncio
    async def test_generate_retrieval_plan_invalid_domain(self, planning_chain):
        """Test retrieval plan generation with invalid domain."""
        mock_entity = Entity(
            id="test_entity",
            name="Test Entity",
            type="Person",
            attributes={"confidence": 0.9},
            neighbors=[],
        )

        with pytest.raises(PlanningError, match="Invalid domain"):
            await planning_chain.generate_retrieval_plan(
                "Test question", mock_entity, "invalid_domain"
            )

    @pytest.mark.asyncio
    async def test_planning_chain_run_with_error_handling(self, planning_chain):
        """Test planning chain execution with error handling."""
        with patch.object(planning_chain, "entity_chain") as mock_entity_chain:
            mock_entity_chain.ainvoke = AsyncMock(
                side_effect=Exception("LLM connection failed")
            )

            result = await planning_chain.arun({"question": "Test question"})

            assert result["success"] is False
            assert "error" in result
            assert "LLM connection failed" in result["error"]

    @pytest.mark.asyncio
    async def test_advanced_planning_chain_entity_resolution(
        self, advanced_planning_chain
    ):
        """Test advanced planning chain entity resolution."""
        mock_kb_entities = [
            Entity(
                id="kb_entity_1",
                name="KB Entity 1",
                type="Person",
                attributes={"confidence": 0.9},
                neighbors=[],
            ),
            Entity(
                id="kb_entity_2",
                name="KB Entity 2",
                type="Person",
                attributes={"confidence": 0.7},
                neighbors=[],
            ),
        ]

        advanced_planning_chain.knowledge_base.search_entities.return_value = (
            mock_kb_entities
        )

        with patch.object(
            advanced_planning_chain, "identify_topic_entity"
        ) as mock_identify:
            mock_identify.return_value = Entity(
                id="llm_entity",
                name="LLM Entity",
                type="Person",
                attributes={"confidence": 0.8},
                neighbors=[],
            )

            entity = (
                await advanced_planning_chain.identify_topic_entity_with_resolution(
                    "Test question"
                )
            )

            # Should return the highest confidence KB entity
            assert entity.name == "KB Entity 1"
            assert entity.attributes["confidence"] == 0.9

    @pytest.mark.asyncio
    async def test_advanced_planning_chain_no_kb_entities(
        self, advanced_planning_chain
    ):
        """Test advanced planning chain when no KB entities found."""
        advanced_planning_chain.knowledge_base.search_entities.return_value = []

        with patch.object(
            advanced_planning_chain, "identify_topic_entity"
        ) as mock_identify:
            mock_llm_entity = Entity(
                id="llm_entity",
                name="LLM Entity",
                type="Person",
                attributes={"confidence": 0.8},
                neighbors=[],
            )
            mock_identify.return_value = mock_llm_entity

            entity = (
                await advanced_planning_chain.identify_topic_entity_with_resolution(
                    "Test question"
                )
            )

            # Should return the LLM entity when no KB entities found
            assert entity.name == "LLM Entity"

    @pytest.mark.asyncio
    async def test_planning_chain_initialization_different_llms(self):
        """Test planning chain initialization with different LLM models."""
        # Test Claude
        with patch("langchain_anthropic.ChatAnthropic") as mock_claude:
            mock_claude.return_value = Mock()
            chain = PlanningChain(llm_model="claude-3-haiku", api_key="test_key")
            assert chain.llm_model == "claude-3-haiku"

        # Test Gemini
        with patch("langchain_google_genai.ChatGoogleGenerativeAI") as mock_gemini:
            mock_gemini.return_value = Mock()
            chain = PlanningChain(llm_model="gemini-1.5-pro", api_key="test_key")
            assert chain.llm_model == "gemini-1.5-pro"

        # Test invalid model
        with pytest.raises(ValueError, match="Unsupported LLM model"):
            PlanningChain(llm_model="invalid-model", api_key="test_key")

    @pytest.mark.asyncio
    async def test_planning_chain_edge_cases(self, planning_chain):
        """Test planning chain with edge cases."""
        # Test empty question
        with pytest.raises(PlanningError, match="Empty question"):
            await planning_chain.identify_topic_entity("")

        # Test very long question
        long_question = "What is " + "a very long question " * 100
        with patch.object(planning_chain, "entity_chain") as mock_chain:
            mock_chain.ainvoke = AsyncMock(
                return_value={
                    "entity_name": "Test Entity",
                    "entity_type": "Person",
                    "confidence": 0.9,
                }
            )

            entity = await planning_chain.identify_topic_entity(long_question)
            assert entity.name == "Test Entity"

    @pytest.mark.asyncio
    async def test_planning_chain_chain_creation(self, planning_chain):
        """Test that planning chains are properly created."""
        # Test entity chain creation
        assert hasattr(planning_chain, "entity_chain")
        assert hasattr(planning_chain, "domain_chain")
        assert hasattr(planning_chain, "retrieval_planning_chain")

        # Test that chains are callable
        assert callable(planning_chain.entity_chain.ainvoke)
        assert callable(planning_chain.domain_chain.ainvoke)
        assert callable(planning_chain.retrieval_planning_chain.ainvoke)


# Test utilities for planning chain
def create_test_planning_entity(
    entity_id: str, name: str, entity_type: str, confidence: float = 0.9
) -> Entity:
    """Create a test entity for planning tests."""
    return Entity(
        id=entity_id,
        name=name,
        type=entity_type,
        attributes={"confidence": confidence},
        neighbors=[],
    )


def create_test_retrieval_plan(
    domain: str, relations: list, max_hops: int = 2
) -> RetrievalPlan:
    """Create a test retrieval plan."""
    entity = create_test_planning_entity("test_entity", "Test Entity", "Person")

    domain_type = DomainType.BOOK if domain == "book" else DomainType.MOVIE

    return RetrievalPlan(
        domain=domain_type,
        topic_entity=entity,
        relations=relations,
        max_hops=max_hops,
        filters={},
    )


# Pytest configuration
@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()
