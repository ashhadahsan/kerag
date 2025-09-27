"""
Test suite for KERAG chains.
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
    FilteredKnowledge,
    PlanningResult,
)
from core.exceptions import (
    PlanningError,
    RetrievalError,
    FilteringError,
    SummarizationError,
)
from chains.planning import PlanningChain, AdvancedPlanningChain
from chains.retrieval import RetrievalChain, AdaptiveRetrievalChain
from chains.filtering import FilteringChain, AdvancedFilteringChain
from chains.summarization import SummarizationChain, AdvancedSummarizationChain


class TestPlanningChain:
    """Test suite for planning chain."""

    @pytest.fixture
    def planning_chain(self):
        """Create planning chain instance for testing."""
        with patch("langchain_openai.ChatOpenAI") as mock_openai:
            mock_openai.return_value = Mock()
            return PlanningChain(llm_model="gpt-4", api_key="test_key", temperature=0.1)

    @pytest.fixture
    def mock_entity(self):
        """Create mock entity for testing."""
        return Entity(
            id="test_entity",
            name="Test Entity",
            type="Person",
            attributes={"confidence": 0.9},
            neighbors=[],
        )

    @pytest.fixture
    def mock_llm_response(self):
        """Mock LLM response for planning."""
        return {
            "entity_name": "J.K. Rowling",
            "entity_type": "Person",
            "confidence": 0.95,
        }

    @pytest.mark.asyncio
    async def test_identify_topic_entity_success(
        self, planning_chain, mock_llm_response
    ):
        """Test successful topic entity identification."""
        with patch.object(planning_chain, "entity_chain") as mock_chain:
            mock_chain.ainvoke = AsyncMock(return_value=mock_llm_response)

            entity = await planning_chain.identify_topic_entity(
                "Which books did J.K. Rowling write?"
            )

            assert entity.name == "J.K. Rowling"
            assert entity.type == "Person"
            assert entity.attributes["confidence"] == 0.95

    @pytest.mark.asyncio
    async def test_identify_topic_entity_failure(self, planning_chain):
        """Test topic entity identification failure."""
        with patch.object(planning_chain, "entity_chain") as mock_chain:
            mock_chain.ainvoke = AsyncMock(return_value={"error": "Failed"})

            with pytest.raises(PlanningError):
                await planning_chain.identify_topic_entity("Test question")

    @pytest.mark.asyncio
    async def test_detect_domain(self, planning_chain, mock_entity):
        """Test domain detection."""
        mock_domain_response = {
            "domain": "book",
            "confidence": 0.9,
            "reasoning": "Question is about books and authors",
        }

        with patch.object(planning_chain, "domain_chain") as mock_chain:
            mock_chain.ainvoke = AsyncMock(return_value=mock_domain_response)

            domain = await planning_chain.detect_domain(
                "Which books did J.K. Rowling write?", mock_entity
            )

            assert domain == "book"

    @pytest.mark.asyncio
    async def test_generate_retrieval_plan(self, planning_chain, mock_entity):
        """Test retrieval plan generation."""
        mock_plan_response = {
            "relations": ["author", "wrote", "published"],
            "max_hops": 2,
            "filters": {"type": "specific"},
            "reasoning": "Need to explore author relationships",
        }

        with patch.object(planning_chain, "retrieval_planning_chain") as mock_chain:
            mock_chain.ainvoke = AsyncMock(return_value=mock_plan_response)

            plan = await planning_chain.generate_retrieval_plan(
                "Which books did J.K. Rowling write?", mock_entity, "book"
            )

            assert plan.domain == DomainType.BOOK
            assert "author" in plan.relations
            assert plan.max_hops == 2

    @pytest.mark.asyncio
    async def test_planning_chain_run(self, planning_chain):
        """Test complete planning chain execution."""
        # Mock all sub-chains
        with patch.object(
            planning_chain, "entity_chain"
        ) as mock_entity_chain, patch.object(
            planning_chain, "domain_chain"
        ) as mock_domain_chain, patch.object(
            planning_chain, "retrieval_planning_chain"
        ) as mock_plan_chain:

            mock_entity_chain.ainvoke = AsyncMock(
                return_value={
                    "entity_name": "J.K. Rowling",
                    "entity_type": "Person",
                    "confidence": 0.95,
                }
            )

            mock_domain_chain.ainvoke = AsyncMock(
                return_value={"domain": "book", "confidence": 0.9}
            )

            mock_plan_chain.ainvoke = AsyncMock(
                return_value={
                    "relations": ["author", "wrote"],
                    "max_hops": 2,
                    "filters": {},
                }
            )

            result = await planning_chain.arun(
                {"question": "Which books did J.K. Rowling write?"}
            )

            assert result["success"] is True
            assert result["domain"] == "book"
            assert result["topic_entity"].name == "J.K. Rowling"


class TestRetrievalChain:
    """Test suite for retrieval chain."""

    @pytest.fixture
    def mock_knowledge_base(self):
        """Create mock knowledge base for testing."""
        kb = Mock()
        kb.get_entity_neighbors = AsyncMock()
        return kb

    @pytest.fixture
    def retrieval_chain(self, mock_knowledge_base):
        """Create retrieval chain instance for testing."""
        from core.types import RetrievalConfig

        config = RetrievalConfig(max_hops=2, max_entities=100)
        return RetrievalChain(mock_knowledge_base, config)

    @pytest.fixture
    def mock_retrieval_plan(self):
        """Create mock retrieval plan for testing."""
        entity = Entity(
            id="test_entity",
            name="Test Entity",
            type="Person",
            attributes={},
            neighbors=[],
        )

        return RetrievalPlan(
            domain=DomainType.BOOK,
            topic_entity=entity,
            relations=["author", "wrote"],
            max_hops=2,
            filters={},
        )

    @pytest.fixture
    def mock_retrieval_result(self):
        """Create mock retrieval result for testing."""
        entities = [
            Entity(id="book1", name="Book 1", type="Book", attributes={}, neighbors=[]),
            Entity(id="book2", name="Book 2", type="Book", attributes={}, neighbors=[]),
        ]

        triples = [
            Triple(subject="test_entity", predicate="author", object="book1"),
            Triple(subject="test_entity", predicate="author", object="book2"),
        ]

        return RetrievalResult(
            entities=entities, triples=triples, metadata={"test": "metadata"}
        )

    @pytest.mark.asyncio
    async def test_expand_neighborhood(
        self, retrieval_chain, mock_knowledge_base, mock_retrieval_result
    ):
        """Test neighborhood expansion."""
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

        assert len(result.entities) == 3  # Original entity + 2 neighbors
        assert len(result.triples) == 6  # 2 relations × 3 hops = 6 triples
        assert result.metadata["max_hops"] == 2

    @pytest.mark.asyncio
    async def test_retrieve_knowledge(self, retrieval_chain, mock_retrieval_plan):
        """Test knowledge retrieval."""
        with patch.object(retrieval_chain, "expand_neighborhood") as mock_expand:
            mock_expand.return_value = RetrievalResult(
                entities=[], triples=[], metadata={}
            )

            result = await retrieval_chain.retrieve_knowledge(mock_retrieval_plan)

            assert isinstance(result, RetrievalResult)

    @pytest.mark.asyncio
    async def test_retrieval_chain_run(self, retrieval_chain, mock_retrieval_plan):
        """Test complete retrieval chain execution."""
        with patch.object(retrieval_chain, "retrieve_knowledge") as mock_retrieve:
            mock_retrieve.return_value = RetrievalResult(
                entities=[], triples=[], metadata={}
            )

            result = await retrieval_chain.arun({"retrieval_plan": mock_retrieval_plan})

            assert result["success"] is True
            assert "retrieved_knowledge" in result


class TestFilteringChain:
    """Test suite for filtering chain."""

    @pytest.fixture
    def filtering_chain(self):
        """Create filtering chain instance for testing."""
        from core.types import FilteringConfig

        config = FilteringConfig(relevance_threshold=0.5, max_triples=100)
        return FilteringChain(llm_model="gpt-4", api_key="test_key", config=config)

    @pytest.fixture
    def mock_retrieval_result(self):
        """Create mock retrieval result for filtering."""
        entities = [
            Entity(
                id="entity1",
                name="Relevant Entity",
                type="Person",
                attributes={},
                neighbors=[],
            ),
            Entity(
                id="entity2",
                name="Irrelevant Entity",
                type="Organization",
                attributes={},
                neighbors=[],
            ),
        ]

        triples = [
            Triple(
                subject="entity1",
                predicate="relatedTo",
                object="entity2",
                confidence=0.8,
            ),
            Triple(
                subject="entity2",
                predicate="unrelatedTo",
                object="entity3",
                confidence=0.2,
            ),
        ]

        return RetrievalResult(entities=entities, triples=triples, metadata={})

    @pytest.mark.asyncio
    async def test_compute_relevance_scores(self, filtering_chain):
        """Test relevance score computation."""
        entities = [
            Entity(
                id="entity1",
                name="Relevant Entity",
                type="Person",
                attributes={},
                neighbors=[],
            )
        ]

        triples = [Triple(subject="entity1", predicate="relatedTo", object="entity2")]

        mock_llm_response = {
            "entity_scores": {"entity1": 0.9},
            "triple_scores": {"0": 0.8},
            "reasoning": "Entity is highly relevant",
        }

        with patch.object(filtering_chain, "relevance_chain") as mock_chain:
            mock_chain.ainvoke = AsyncMock(return_value=mock_llm_response)

            scores = await filtering_chain.compute_relevance_scores(
                "Test question", entities, triples
            )

            assert scores["entity_entity1"] == 0.9
            assert scores["triple_0"] == 0.8

    @pytest.mark.asyncio
    async def test_filter_knowledge(self, filtering_chain, mock_retrieval_result):
        """Test knowledge filtering."""
        with patch.object(filtering_chain, "compute_relevance_scores") as mock_scores:
            mock_scores.return_value = {
                "entity_entity1": 0.9,
                "entity_entity2": 0.3,
                "triple_0": 0.8,
                "triple_1": 0.2,
            }

            result = await filtering_chain.filter_knowledge(
                "Test question", mock_retrieval_result
            )

            # Should filter out low-relevance items
            assert len(result.entities) == 1  # Only entity1 should remain
            assert len(result.triples) == 1  # Only triple 0 should remain

    @pytest.mark.asyncio
    async def test_filtering_chain_run(self, filtering_chain, mock_retrieval_result):
        """Test complete filtering chain execution."""
        with patch.object(filtering_chain, "compute_relevance_scores") as mock_scores:
            mock_scores.return_value = {"entity_entity1": 0.9, "triple_0": 0.8}

            result = await filtering_chain.arun(
                {
                    "question": "Test question",
                    "retrieved_knowledge": mock_retrieval_result,
                }
            )

            assert result["success"] is True
            assert "filtered_knowledge" in result


class TestSummarizationChain:
    """Test suite for summarization chain."""

    @pytest.fixture
    def summarization_chain(self):
        """Create summarization chain instance for testing."""
        from core.types import SummarizationConfig

        config = SummarizationConfig(use_cot=True, max_tokens=1000)
        return SummarizationChain(llm_model="gpt-4", api_key="test_key", config=config)

    @pytest.fixture
    def mock_filtered_knowledge(self):
        """Create mock filtered knowledge for summarization."""
        entities = [
            Entity(
                id="author1",
                name="J.K. Rowling",
                type="Person",
                attributes={},
                neighbors=[],
            )
        ]

        triples = [Triple(subject="author1", predicate="wrote", object="Harry Potter")]

        return RetrievalResult(entities=entities, triples=triples, metadata={})

    @pytest.mark.asyncio
    async def test_generate_answer_cot(
        self, summarization_chain, mock_filtered_knowledge
    ):
        """Test answer generation with Chain-of-Thought."""
        mock_cot_response = """
        Reasoning: The question asks about books written by J.K. Rowling. 
        From the knowledge base, I can see that J.K. Rowling is an author 
        who wrote Harry Potter.
        
        Answer: J.K. Rowling wrote Harry Potter.
        """

        with patch.object(summarization_chain, "cot_chain") as mock_chain:
            mock_chain.ainvoke = AsyncMock(return_value=mock_cot_response)

            answer = await summarization_chain.generate_answer(
                "What books did J.K. Rowling write?", mock_filtered_knowledge
            )

            assert "Harry Potter" in answer

    @pytest.mark.asyncio
    async def test_compute_confidence(
        self, summarization_chain, mock_filtered_knowledge
    ):
        """Test confidence computation."""
        mock_confidence_response = """
        The answer is well-supported by the knowledge base. 
        J.K. Rowling is clearly identified as an author and 
        Harry Potter is clearly linked as a work she wrote.
        Confidence: 0.9
        """

        with patch.object(summarization_chain, "confidence_chain") as mock_chain:
            mock_chain.ainvoke = AsyncMock(return_value=mock_confidence_response)

            confidence = await summarization_chain.compute_confidence(
                "What books did J.K. Rowling write?",
                "J.K. Rowling wrote Harry Potter.",
                mock_filtered_knowledge,
            )

            assert 0.8 <= confidence <= 1.0

    @pytest.mark.asyncio
    async def test_extract_answer_from_cot(self, summarization_chain):
        """Test answer extraction from CoT response."""
        cot_response = """
        Reasoning: Let me think about this step by step.
        The question asks about books by J.K. Rowling.
        From the data, I can see she wrote Harry Potter.
        
        Answer: J.K. Rowling wrote Harry Potter and other books in the series.
        """

        answer = summarization_chain._extract_answer_from_cot(cot_response)
        assert "Harry Potter" in answer

    @pytest.mark.asyncio
    async def test_summarization_chain_run(
        self, summarization_chain, mock_filtered_knowledge
    ):
        """Test complete summarization chain execution."""
        with patch.object(summarization_chain, "cot_chain") as mock_cot, patch.object(
            summarization_chain, "confidence_chain"
        ) as mock_conf:

            mock_cot.ainvoke = AsyncMock(return_value="Answer: Test answer")
            mock_conf.ainvoke = AsyncMock(return_value="Confidence: 0.9")

            result = await summarization_chain.arun(
                {
                    "question": "Test question",
                    "filtered_knowledge": mock_filtered_knowledge,
                }
            )

            assert result["success"] is True
            assert "answer" in result
            assert "confidence" in result


class TestAdvancedChains:
    """Test suite for advanced chain implementations."""

    @pytest.mark.asyncio
    async def test_advanced_planning_chain(self):
        """Test advanced planning chain with knowledge base integration."""
        mock_kb = Mock()
        mock_kb.search_entities = AsyncMock(
            return_value=[
                Entity(
                    id="kb_entity",
                    name="KB Entity",
                    type="Person",
                    attributes={},
                    neighbors=[],
                )
            ]
        )

        chain = AdvancedPlanningChain(
            llm_model="gpt-4", api_key="test_key", knowledge_base=mock_kb
        )

        with patch.object(chain, "identify_topic_entity") as mock_identify:
            mock_identify.return_value = Entity(
                id="test_entity",
                name="Test Entity",
                type="Person",
                attributes={},
                neighbors=[],
            )

            entity = await chain.identify_topic_entity_with_resolution("Test question")
            # The method should return the KB entity, not the mocked one
            assert entity.name == "KB Entity"

    @pytest.mark.asyncio
    async def test_adaptive_retrieval_chain(self):
        """Test adaptive retrieval chain."""
        mock_kb = Mock()
        mock_kb.get_relations = AsyncMock(return_value=["relation1", "relation2"])
        mock_kb.get_entity_neighbors = AsyncMock(
            return_value=RetrievalResult(entities=[], triples=[], metadata={})
        )

        from core.types import RetrievalConfig

        config = RetrievalConfig(max_hops=2, max_entities=100)

        chain = AdaptiveRetrievalChain(mock_kb, config)

        entity = Entity(
            id="test", name="Test", type="Person", attributes={}, neighbors=[]
        )

        result = await chain.expand_neighborhood(entity, ["rel1"], 2)
        assert isinstance(result, RetrievalResult)

    @pytest.mark.asyncio
    async def test_advanced_filtering_chain(self):
        """Test advanced filtering chain with multiple strategies."""
        from core.types import FilteringConfig

        config = FilteringConfig(relevance_threshold=0.5)

        chain = AdvancedFilteringChain(
            llm_model="gpt-4", api_key="test_key", config=config
        )

        mock_result = RetrievalResult(entities=[], triples=[], metadata={})

        with patch.object(chain, "_semantic_filtering") as mock_semantic, patch.object(
            chain, "_graph_based_filtering"
        ) as mock_graph:

            mock_semantic.return_value = mock_result
            mock_graph.return_value = mock_result

            result = await chain.filter_knowledge("Test question", mock_result)
            assert isinstance(result, RetrievalResult)

    @pytest.mark.asyncio
    async def test_advanced_summarization_chain(self):
        """Test advanced summarization chain with multiple strategies."""
        from core.types import SummarizationConfig

        config = SummarizationConfig(use_cot=True)

        chain = AdvancedSummarizationChain(
            llm_model="gpt-4", api_key="test_key", config=config
        )

        mock_knowledge = RetrievalResult(entities=[], triples=[], metadata={})

        with patch.object(
            chain, "_extraction_based_answer"
        ) as mock_extract, patch.object(
            chain, "_template_based_answer"
        ) as mock_template, patch.object(
            chain, "cot_chain"
        ) as mock_cot:

            mock_extract.return_value = "Extraction answer"
            mock_template.return_value = "Template answer"
            mock_cot.ainvoke = AsyncMock(return_value="Answer: Test answer")

            answer = await chain.generate_answer("Test question", mock_knowledge)
            assert isinstance(answer, str)


# Test utilities
def create_test_entity(entity_id: str, name: str, entity_type: str) -> Entity:
    """Create a test entity."""
    return Entity(
        id=entity_id,
        name=name,
        type=entity_type,
        attributes={"test": "value"},
        neighbors=[],
    )


def create_test_triple(subject: str, predicate: str, obj: str) -> Triple:
    """Create a test triple."""
    return Triple(subject=subject, predicate=predicate, object=obj, confidence=0.8)


# Pytest configuration
@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()
