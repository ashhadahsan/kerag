"""
Test configuration for KERAG tests.
"""

import os
from unittest.mock import Mock, AsyncMock, patch
from core.types import (
    KERAGConfig,
    RetrievalConfig,
    FilteringConfig,
    SummarizationConfig,
    KnowledgeBaseType,
)

# Mock API key for testing
MOCK_API_KEY = "test_key_12345"


def get_test_config():
    """Get test configuration."""
    return KERAGConfig(
        knowledge_base_type=KnowledgeBaseType.SPARQL,
        endpoint_url="http://test-endpoint.com/sparql",
        llm_model="gpt-4",
        api_key=MOCK_API_KEY,
        retrieval_config=RetrievalConfig(
            max_hops=3,
            max_entities=100,
            max_triples=200,
            timeout=30,
            confidence_threshold=0.5,
        ),
        filtering_config=FilteringConfig(
            relevance_threshold=0.5, max_triples=500, use_llm_filtering=True
        ),
        summarization_config=SummarizationConfig(
            use_cot=True, max_tokens=4000, temperature=0.1
        ),
        debug=True,
        timeout=30,
    )


def mock_llm_response():
    """Mock LLM response for testing."""
    return {
        "entity_name": "Test Entity",
        "entity_type": "Person",
        "confidence": 0.95,
        "domain": "book",
        "relations": ["author", "wrote"],
        "max_hops": 2,
        "filters": {},
        "answer": "Test answer",
        "reasoning": "Test reasoning",
    }


def mock_retrieval_result():
    """Mock retrieval result for testing."""
    from core.types import Entity, Triple, RetrievalResult

    entity = Entity(
        id="test_entity", name="Test Entity", type="Person", attributes={}, neighbors=[]
    )

    triple = Triple(
        subject="test_entity", predicate="author", object="test_book", confidence=0.9
    )

    return RetrievalResult(
        entities=[entity],
        triples=[triple],
        metadata={
            "source_entity": "test_entity",
            "max_hops": 2,
            "relations": ["author", "wrote"],
            "total_entities": 1,
            "total_triples": 1,
            "expansion_method": "multi_hop_neighborhood",
        },
    )


def setup_test_environment():
    """Setup test environment with mocked components."""
    # Mock environment variables
    os.environ["OPENAI_API_KEY"] = MOCK_API_KEY
    os.environ["ANTHROPIC_API_KEY"] = MOCK_API_KEY
    os.environ["GOOGLE_API_KEY"] = MOCK_API_KEY
    os.environ["XAI_API_KEY"] = MOCK_API_KEY

    # Mock LLM responses
    mock_response = mock_llm_response()

    # Create mock LLM
    mock_llm = Mock()
    mock_llm.ainvoke = AsyncMock(return_value=mock_response)
    mock_llm.invoke = Mock(return_value=mock_response)

    return mock_llm, mock_response


def mock_knowledge_base():
    """Create mock knowledge base."""
    mock_kb = Mock()
    mock_kb.get_entity_info = AsyncMock(
        return_value=Entity(
            id="test_entity",
            name="Test Entity",
            type="Person",
            attributes={},
            neighbors=[],
        )
    )
    mock_kb.get_entity_neighbors = AsyncMock(return_value=mock_retrieval_result())
    mock_kb.search_entities = AsyncMock(return_value=[])
    mock_kb.get_relations = AsyncMock(return_value=["author", "wrote"])
    mock_kb.execute_query = AsyncMock(return_value=mock_retrieval_result())

    return mock_kb
