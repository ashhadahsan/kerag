"""
Pytest configuration and fixtures for KERAG tests.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from test_config import (
    get_test_config,
    mock_llm_response,
    mock_retrieval_result,
    setup_test_environment,
    mock_knowledge_base,
)
from core.types import Entity, Triple, RetrievalResult, RetrievalPlan, DomainType


@pytest.fixture
def test_config():
    """Test configuration fixture."""
    return get_test_config()


@pytest.fixture
def mock_llm():
    """Mock LLM fixture."""
    mock_llm, _ = setup_test_environment()
    return mock_llm


@pytest.fixture
def mock_kb():
    """Mock knowledge base fixture."""
    return mock_knowledge_base()


@pytest.fixture
def mock_retrieval_plan():
    """Mock retrieval plan fixture."""
    entity = Entity(
        id="test_entity", name="Test Entity", type="Person", attributes={}, neighbors=[]
    )

    return RetrievalPlan(
        domain=DomainType.BOOK,
        topic_entity=entity,
        relations=["author", "wrote"],
        max_hops=2,
        filters={},
    )


@pytest.fixture
def mock_entity():
    """Mock entity fixture."""
    return Entity(
        id="test_entity", name="Test Entity", type="Person", attributes={}, neighbors=[]
    )


@pytest.fixture
def mock_triple():
    """Mock triple fixture."""
    return Triple(
        subject="test_entity", predicate="author", object="test_book", confidence=0.9
    )


@pytest.fixture
def mock_retrieval_result_fixture():
    """Mock retrieval result fixture."""
    return mock_retrieval_result()


@pytest.fixture
def mock_planning_result():
    """Mock planning result fixture."""
    return {
        "domain": "book",
        "topic_entity": {
            "id": "test_entity",
            "name": "Test Entity",
            "type": "Person",
            "attributes": {},
            "neighbors": [],
        },
        "retrieval_plan": {
            "domain": "book",
            "topic_entity": {
                "id": "test_entity",
                "name": "Test Entity",
                "type": "Person",
                "attributes": {},
                "neighbors": [],
            },
            "relations": ["author", "wrote"],
            "max_hops": 2,
            "filters": {},
        },
        "success": True,
    }


@pytest.fixture
def mock_retrieval_result_dict():
    """Mock retrieval result as dictionary."""
    return {
        "retrieved_knowledge": {
            "entities": [
                {
                    "id": "test_entity",
                    "name": "Test Entity",
                    "type": "Person",
                    "attributes": {},
                    "neighbors": [],
                }
            ],
            "triples": [
                {
                    "subject": "test_entity",
                    "predicate": "author",
                    "object": "test_book",
                    "confidence": 0.9,
                }
            ],
            "metadata": {
                "source_entity": "test_entity",
                "max_hops": 2,
                "relations": ["author", "wrote"],
                "total_entities": 1,
                "total_triples": 1,
                "expansion_method": "multi_hop_neighborhood",
            },
        },
        "success": True,
    }


@pytest.fixture
def mock_filtering_result():
    """Mock filtering result."""
    return {
        "filtered_knowledge": {
            "entities": [
                {
                    "id": "test_entity",
                    "name": "Test Entity",
                    "type": "Person",
                    "attributes": {},
                    "neighbors": [],
                }
            ],
            "triples": [
                {
                    "subject": "test_entity",
                    "predicate": "author",
                    "object": "test_book",
                    "confidence": 0.9,
                }
            ],
            "metadata": {
                "source_entity": "test_entity",
                "max_hops": 2,
                "relations": ["author", "wrote"],
                "total_entities": 1,
                "total_triples": 1,
                "expansion_method": "multi_hop_neighborhood",
            },
        },
        "success": True,
    }


@pytest.fixture
def mock_summarization_result():
    """Mock summarization result."""
    return {
        "answer": "Test Entity wrote Test Book.",
        "confidence": 0.9,
        "success": True,
    }


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


# Mock all LLM calls to avoid API key issues
@pytest.fixture(autouse=True)
def mock_llm_calls():
    """Mock all LLM calls to avoid API key issues."""
    with patch("langchain_openai.ChatOpenAI") as mock_openai, patch(
        "langchain_anthropic.ChatAnthropic"
    ) as mock_anthropic, patch(
        "langchain_google_genai.ChatGoogleGenerativeAI"
    ) as mock_google:

        # Mock OpenAI
        mock_openai_instance = Mock()
        mock_openai_instance.ainvoke = AsyncMock(return_value=mock_llm_response())
        mock_openai_instance.invoke = Mock(return_value=mock_llm_response())
        mock_openai.return_value = mock_openai_instance

        # Mock Anthropic
        mock_anthropic_instance = Mock()
        mock_anthropic_instance.ainvoke = AsyncMock(return_value=mock_llm_response())
        mock_anthropic_instance.invoke = Mock(return_value=mock_llm_response())
        mock_anthropic.return_value = mock_anthropic_instance

        # Mock Google
        mock_google_instance = Mock()
        mock_google_instance.ainvoke = AsyncMock(return_value=mock_llm_response())
        mock_google_instance.invoke = Mock(return_value=mock_llm_response())
        mock_google.return_value = mock_google_instance

        yield {
            "openai": mock_openai_instance,
            "anthropic": mock_anthropic_instance,
            "google": mock_google_instance,
        }
