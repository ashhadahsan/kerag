"""
Extended test suite for summarization chain to improve coverage.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch

from core.types import Entity, Triple, RetrievalResult, SummarizationConfig
from core.exceptions import SummarizationError
from chains.summarization import SummarizationChain, AdvancedSummarizationChain


class TestSummarizationChainExtended:
    """Extended test suite for summarization chain."""

    @pytest.fixture
    def summarization_chain(self):
        """Create summarization chain instance for testing."""
        config = SummarizationConfig(use_cot=True, max_tokens=1000, temperature=0.1)
        with patch("langchain_openai.ChatOpenAI") as mock_openai:
            mock_openai.return_value = Mock()
            return SummarizationChain(
                llm_model="gpt-4", api_key="test_key", config=config
            )

    @pytest.fixture
    def advanced_summarization_chain(self):
        """Create advanced summarization chain instance for testing."""
        config = SummarizationConfig(use_cot=True, max_tokens=1000)
        with patch("langchain_openai.ChatOpenAI") as mock_openai:
            mock_openai.return_value = Mock()
            return AdvancedSummarizationChain(
                llm_model="gpt-4", api_key="test_key", config=config
            )

    @pytest.fixture
    def mock_knowledge_base(self):
        """Create mock knowledge base for testing."""
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
    async def test_generate_answer_with_chain_of_thought(
        self, summarization_chain, mock_knowledge_base
    ):
        """Test answer generation with Chain-of-Thought reasoning."""
        mock_cot_response = """
        Reasoning: The question asks about books written by J.K. Rowling. 
        From the knowledge base, I can see that J.K. Rowling is an author 
        who wrote multiple books in the Harry Potter series.
        
        Step 1: Identify the author - J.K. Rowling
        Step 2: Find books by this author - Harry Potter and the Philosopher's Stone, Harry Potter and the Chamber of Secrets
        Step 3: Provide comprehensive answer
        
        Answer: J.K. Rowling wrote several books, including "Harry Potter and the Philosopher's Stone" (1997) and "Harry Potter and the Chamber of Secrets" (1998), both part of the Harry Potter fantasy series.
        """

        with patch.object(summarization_chain, "cot_chain") as mock_chain:
            mock_chain.ainvoke = AsyncMock(return_value=mock_cot_response)

            answer = await summarization_chain.generate_answer(
                "What books did J.K. Rowling write?", mock_knowledge_base
            )

            assert "Harry Potter" in answer
            assert "J.K. Rowling" in answer
            assert "1997" in answer
            assert "1998" in answer

    @pytest.mark.asyncio
    async def test_generate_answer_without_cot(
        self, summarization_chain, mock_knowledge_base
    ):
        """Test answer generation without Chain-of-Thought."""
        summarization_chain.config.use_cot = False

        mock_direct_response = "J.K. Rowling wrote the Harry Potter series, including 'Harry Potter and the Philosopher's Stone' and 'Harry Potter and the Chamber of Secrets'."

        with patch.object(summarization_chain, "direct_chain") as mock_chain:
            mock_chain.ainvoke = AsyncMock(return_value=mock_direct_response)

            answer = await summarization_chain.generate_answer(
                "What books did J.K. Rowling write?", mock_knowledge_base
            )

            assert answer == mock_direct_response

    @pytest.mark.asyncio
    async def test_compute_confidence_high_confidence(
        self, summarization_chain, mock_knowledge_base
    ):
        """Test confidence computation with high confidence."""
        mock_confidence_response = """
        The answer is well-supported by the knowledge base. 
        J.K. Rowling is clearly identified as an author and 
        multiple books are clearly linked as works she wrote.
        The relationships are direct and unambiguous.
        Confidence: 0.95
        """

        with patch.object(summarization_chain, "confidence_chain") as mock_chain:
            mock_chain.ainvoke = AsyncMock(return_value=mock_confidence_response)

            confidence = await summarization_chain.compute_confidence(
                "What books did J.K. Rowling write?",
                "J.K. Rowling wrote the Harry Potter series.",
                mock_knowledge_base,
            )

            assert 0.9 <= confidence <= 1.0

    @pytest.mark.asyncio
    async def test_compute_confidence_low_confidence(
        self, summarization_chain, mock_knowledge_base
    ):
        """Test confidence computation with low confidence."""
        mock_confidence_response = """
        The answer has limited support in the knowledge base.
        While J.K. Rowling is mentioned, the specific books
        mentioned in the answer are not clearly linked.
        Confidence: 0.3
        """

        with patch.object(summarization_chain, "confidence_chain") as mock_chain:
            mock_chain.ainvoke = AsyncMock(return_value=mock_confidence_response)

            confidence = await summarization_chain.compute_confidence(
                "What books did J.K. Rowling write?",
                "J.K. Rowling wrote some fantasy books.",
                mock_knowledge_base,
            )

            assert 0.0 <= confidence <= 0.5

    @pytest.mark.asyncio
    async def test_extract_answer_from_cot_various_formats(self, summarization_chain):
        """Test answer extraction from various CoT response formats."""
        # Test with "Answer:" prefix
        cot_response_1 = """
        Reasoning: Let me think about this step by step.
        The question asks about books by J.K. Rowling.
        From the data, I can see she wrote Harry Potter books.
        
        Answer: J.K. Rowling wrote the Harry Potter series.
        """

        answer_1 = summarization_chain._extract_answer_from_cot(cot_response_1)
        assert "Harry Potter" in answer_1

        # Test with "Final Answer:" prefix
        cot_response_2 = """
        Step 1: Identify the author
        Step 2: Find their works
        Step 3: Compile the list
        
        Final Answer: J.K. Rowling wrote multiple books including Harry Potter.
        """

        answer_2 = summarization_chain._extract_answer_from_cot(cot_response_2)
        assert "Harry Potter" in answer_2

        # Test with no clear answer marker
        cot_response_3 = """
        Based on the information provided, J.K. Rowling is known for writing
        the Harry Potter series of fantasy novels.
        """

        answer_3 = summarization_chain._extract_answer_from_cot(cot_response_3)
        assert "Harry Potter" in answer_3

    @pytest.mark.asyncio
    async def test_summarization_chain_run_success(
        self, summarization_chain, mock_knowledge_base
    ):
        """Test complete summarization chain execution."""
        with patch.object(summarization_chain, "cot_chain") as mock_cot, patch.object(
            summarization_chain, "confidence_chain"
        ) as mock_conf:

            mock_cot.ainvoke = AsyncMock(
                return_value="Answer: J.K. Rowling wrote the Harry Potter series."
            )
            mock_conf.ainvoke = AsyncMock(return_value="Confidence: 0.9")

            result = await summarization_chain.arun(
                {
                    "question": "What books did J.K. Rowling write?",
                    "filtered_knowledge": mock_knowledge_base,
                }
            )

            assert result["success"] is True
            assert "answer" in result
            assert "confidence" in result
            assert result["confidence"] == 0.9

    @pytest.mark.asyncio
    async def test_summarization_chain_run_failure(
        self, summarization_chain, mock_knowledge_base
    ):
        """Test summarization chain execution with failure."""
        with patch.object(summarization_chain, "cot_chain") as mock_cot:
            mock_cot.ainvoke = AsyncMock(side_effect=Exception("LLM connection failed"))

            result = await summarization_chain.arun(
                {
                    "question": "What books did J.K. Rowling write?",
                    "filtered_knowledge": mock_knowledge_base,
                }
            )

            assert result["success"] is False
            assert "error" in result
            assert "LLM connection failed" in result["error"]

    @pytest.mark.asyncio
    async def test_advanced_summarization_chain_extraction_based(
        self, advanced_summarization_chain, mock_knowledge_base
    ):
        """Test advanced summarization chain with extraction-based strategy."""
        with patch.object(
            advanced_summarization_chain, "_extraction_based_answer"
        ) as mock_extract:
            mock_extract.return_value = (
                "Extraction-based answer: J.K. Rowling wrote Harry Potter books."
            )

            answer = await advanced_summarization_chain.generate_answer(
                "What books did J.K. Rowling write?", mock_knowledge_base
            )

            assert "Extraction-based answer" in answer
            mock_extract.assert_called_once()

    @pytest.mark.asyncio
    async def test_advanced_summarization_chain_template_based(
        self, advanced_summarization_chain, mock_knowledge_base
    ):
        """Test advanced summarization chain with template-based strategy."""
        with patch.object(
            advanced_summarization_chain, "_template_based_answer"
        ) as mock_template:
            mock_template.return_value = "Template-based answer: Based on the data, J.K. Rowling wrote the Harry Potter series."

            answer = await advanced_summarization_chain.generate_answer(
                "What books did J.K. Rowling write?", mock_knowledge_base
            )

            assert "Template-based answer" in answer
            mock_template.assert_called_once()

    @pytest.mark.asyncio
    async def test_advanced_summarization_chain_hybrid_strategy(
        self, advanced_summarization_chain, mock_knowledge_base
    ):
        """Test advanced summarization chain with hybrid strategy."""
        with patch.object(
            advanced_summarization_chain, "_extraction_based_answer"
        ) as mock_extract, patch.object(
            advanced_summarization_chain, "_template_based_answer"
        ) as mock_template, patch.object(
            advanced_summarization_chain, "_hybrid_answer"
        ) as mock_hybrid:

            mock_extract.return_value = "Extraction answer"
            mock_template.return_value = "Template answer"
            mock_hybrid.return_value = "Hybrid answer combining both approaches"

            # Set config to use hybrid strategy
            advanced_summarization_chain.config.strategy = "hybrid"

            answer = await advanced_summarization_chain.generate_answer(
                "What books did J.K. Rowling write?", mock_knowledge_base
            )

            assert "Hybrid answer" in answer
            mock_hybrid.assert_called_once()

    @pytest.mark.asyncio
    async def test_summarization_chain_initialization_different_llms(self):
        """Test summarization chain initialization with different LLM models."""
        config = SummarizationConfig(use_cot=True, max_tokens=1000)

        # Test Claude
        with patch("langchain_anthropic.ChatAnthropic") as mock_claude:
            mock_claude.return_value = Mock()
            chain = SummarizationChain(
                llm_model="claude-3-haiku", api_key="test_key", config=config
            )
            assert chain.llm_model == "claude-3-haiku"

        # Test Gemini
        with patch("langchain_google_genai.ChatGoogleGenerativeAI") as mock_gemini:
            mock_gemini.return_value = Mock()
            chain = SummarizationChain(
                llm_model="gemini-1.5-pro", api_key="test_key", config=config
            )
            assert chain.llm_model == "gemini-1.5-pro"

        # Test invalid model
        with pytest.raises(ValueError, match="Unsupported LLM model"):
            SummarizationChain(
                llm_model="invalid-model", api_key="test_key", config=config
            )

    @pytest.mark.asyncio
    async def test_summarization_chain_edge_cases(self, summarization_chain):
        """Test summarization chain with edge cases."""
        # Test with empty knowledge base
        empty_knowledge = RetrievalResult(entities=[], triples=[], metadata={})

        with patch.object(summarization_chain, "cot_chain") as mock_chain:
            mock_chain.ainvoke = AsyncMock(
                return_value="Answer: No information available."
            )

            answer = await summarization_chain.generate_answer(
                "What books did J.K. Rowling write?", empty_knowledge
            )

            assert "No information available" in answer

        # Test with very long question
        long_question = "What books did " + "J.K. Rowling " * 50 + "write?"

        with patch.object(summarization_chain, "cot_chain") as mock_chain:
            mock_chain.ainvoke = AsyncMock(
                return_value="Answer: J.K. Rowling wrote Harry Potter."
            )

            answer = await summarization_chain.generate_answer(
                long_question, empty_knowledge
            )

            assert "Harry Potter" in answer

    @pytest.mark.asyncio
    async def test_summarization_chain_config_validation(self):
        """Test summarization chain configuration validation."""
        # Test valid config
        valid_config = SummarizationConfig(
            use_cot=True, max_tokens=1000, temperature=0.1, strategy="cot"
        )

        with patch("langchain_openai.ChatOpenAI") as mock_openai:
            mock_openai.return_value = Mock()
            chain = SummarizationChain(
                llm_model="gpt-4", api_key="test_key", config=valid_config
            )
            assert chain.config.use_cot is True
            assert chain.config.max_tokens == 1000

        # Test invalid config
        with pytest.raises(ValueError, match="Invalid strategy"):
            invalid_config = SummarizationConfig(strategy="invalid_strategy")
            SummarizationChain(
                llm_model="gpt-4", api_key="test_key", config=invalid_config
            )

    @pytest.mark.asyncio
    async def test_summarization_chain_chain_creation(self, summarization_chain):
        """Test that summarization chains are properly created."""
        # Test chain creation
        assert hasattr(summarization_chain, "cot_chain")
        assert hasattr(summarization_chain, "confidence_chain")
        assert hasattr(summarization_chain, "direct_chain")

        # Test that chains are callable
        assert callable(summarization_chain.cot_chain.ainvoke)
        assert callable(summarization_chain.confidence_chain.ainvoke)
        assert callable(summarization_chain.direct_chain.ainvoke)


# Test utilities for summarization chain
def create_test_knowledge_base(entities: list, triples: list) -> RetrievalResult:
    """Create a test knowledge base for summarization tests."""
    return RetrievalResult(
        entities=entities,
        triples=triples,
        metadata={"source": "test", "timestamp": "2024-01-01"},
    )


def create_test_entity_for_summarization(
    entity_id: str, name: str, entity_type: str, attributes: dict = None
) -> Entity:
    """Create a test entity for summarization tests."""
    return Entity(
        id=entity_id,
        name=name,
        type=entity_type,
        attributes=attributes or {},
        neighbors=[],
    )


def create_test_triple_for_summarization(
    subject: str, predicate: str, obj: str, confidence: float = 0.9
) -> Triple:
    """Create a test triple for summarization tests."""
    return Triple(
        subject=subject, predicate=predicate, object=obj, confidence=confidence
    )


# Pytest configuration
@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()
