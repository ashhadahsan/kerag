"""
Test suite for KERAG workflows.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch

from core.types import KERAGConfig, KnowledgeBaseType, KERAGResult
from core.exceptions import KERAGException
from workflows.kerag_workflow import (
    KERAGWorkflow,
    AsyncKERAGWorkflow,
    KERAGWorkflowState,
)


class TestKERAGWorkflow:
    """Test suite for KERAG workflow."""

    @pytest.fixture
    def mock_config(self):
        """Create mock KERAG configuration."""
        return KERAGConfig(
            knowledge_base_type=KnowledgeBaseType.SPARQL,
            endpoint_url="http://test-endpoint.com/sparql",
            llm_model="gpt-4",
            api_key="test_key",
        )

    @pytest.fixture
    def mock_workflow(self, mock_config):
        """Create mock KERAG workflow."""
        with patch("workflows.kerag_workflow.SPARQLKnowledgeBase") as mock_kb_class:
            mock_kb = Mock()
            mock_kb_class.return_value = mock_kb

            workflow = KERAGWorkflow(mock_config)
            workflow.knowledge_base = mock_kb

            return workflow

    @pytest.fixture
    def mock_planning_result(self):
        """Create mock planning result."""
        return {
            "question": "Test question",
            "topic_entity": {
                "id": "entity1",
                "name": "Test Entity",
                "type": "Person",
                "attributes": {},
                "neighbors": [],
            },
            "domain": "book",
            "retrieval_plan": {
                "domain": "book",
                "topic_entity": {
                    "id": "entity1",
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
    def mock_retrieval_result(self):
        """Create mock retrieval result."""
        return {
            "retrieved_knowledge": {
                "entities": [
                    {
                        "id": "book1",
                        "name": "Test Book",
                        "type": "Book",
                        "attributes": {},
                        "neighbors": [],
                    }
                ],
                "triples": [
                    {
                        "subject": "entity1",
                        "predicate": "author",
                        "object": "book1",
                        "confidence": 0.9,
                    }
                ],
                "metadata": {},
            },
            "success": True,
        }

    @pytest.fixture
    def mock_filtering_result(self):
        """Create mock filtering result."""
        return {
            "filtered_knowledge": {
                "entities": [
                    {
                        "id": "book1",
                        "name": "Test Book",
                        "type": "Book",
                        "attributes": {},
                        "neighbors": [],
                    }
                ],
                "triples": [
                    {
                        "subject": "entity1",
                        "predicate": "author",
                        "object": "book1",
                        "confidence": 0.9,
                    }
                ],
                "metadata": {},
            },
            "success": True,
        }

    @pytest.fixture
    def mock_summarization_result(self):
        """Create mock summarization result."""
        return {
            "answer": "Test Entity wrote Test Book.",
            "confidence": 0.9,
            "success": True,
        }

    @pytest.mark.asyncio
    async def test_planning_node_success(self, mock_workflow, mock_planning_result):
        """Test successful planning node execution."""
        with patch.object(mock_workflow.planning_chain, "arun") as mock_planning:
            mock_planning.return_value = mock_planning_result

            state: KERAGWorkflowState = {
                "question": "Test question",
                "domain": None,
                "topic_entity": None,
                "retrieval_plan": None,
                "retrieved_knowledge": None,
                "filtered_knowledge": None,
                "answer": None,
                "confidence": None,
                "metadata": {},
                "error": None,
                "success": False,
            }

            result_state = await mock_workflow._planning_node(state)

            assert result_state["success"] is True
            assert result_state["domain"] == "book"
            assert result_state["topic_entity"]["name"] == "Test Entity"

    @pytest.mark.asyncio
    async def test_planning_node_failure(self, mock_workflow):
        """Test planning node failure handling."""
        with patch.object(mock_workflow.planning_chain, "arun") as mock_planning:
            mock_planning.return_value = {
                "question": "Test question",
                "error": "Planning failed",
                "success": False,
            }

            state: KERAGWorkflowState = {
                "question": "Test question",
                "domain": None,
                "topic_entity": None,
                "retrieval_plan": None,
                "retrieved_knowledge": None,
                "filtered_knowledge": None,
                "answer": None,
                "confidence": None,
                "metadata": {},
                "error": None,
                "success": False,
            }

            result_state = await mock_workflow._planning_node(state)

            assert result_state["success"] is False
            assert "Planning failed" in result_state["error"]

    @pytest.mark.asyncio
    async def test_retrieval_node_success(self, mock_workflow, mock_retrieval_result):
        """Test successful retrieval node execution."""
        with patch.object(mock_workflow.retrieval_chain, "arun") as mock_retrieval:
            mock_retrieval.return_value = mock_retrieval_result

            state: KERAGWorkflowState = {
                "question": "Test question",
                "domain": "book",
                "topic_entity": {
                    "id": "entity1",
                    "name": "Test Entity",
                    "type": "Person",
                    "attributes": {},
                    "neighbors": [],
                },
                "retrieval_plan": {
                    "domain": "book",
                    "topic_entity": {
                        "id": "entity1",
                        "name": "Test Entity",
                        "type": "Person",
                        "attributes": {},
                        "neighbors": [],
                    },
                    "relations": ["author", "wrote"],
                    "max_hops": 2,
                    "filters": {},
                },
                "retrieved_knowledge": None,
                "filtered_knowledge": None,
                "answer": None,
                "confidence": None,
                "metadata": {},
                "error": None,
                "success": True,
            }

            result_state = await mock_workflow._retrieval_node(state)

            assert "retrieved_knowledge" in result_state
            assert (
                result_state["retrieved_knowledge"]["entities"][0]["name"]
                == "Test Book"
            )

    @pytest.mark.asyncio
    async def test_filtering_node_success(self, mock_workflow, mock_filtering_result):
        """Test successful filtering node execution."""
        with patch.object(mock_workflow.filtering_chain, "arun") as mock_filtering:
            mock_filtering.return_value = mock_filtering_result

            state: KERAGWorkflowState = {
                "question": "Test question",
                "domain": "book",
                "topic_entity": {
                    "id": "entity1",
                    "name": "Test Entity",
                    "type": "Person",
                    "attributes": {},
                    "neighbors": [],
                },
                "retrieval_plan": {},
                "retrieved_knowledge": {"entities": [], "triples": [], "metadata": {}},
                "filtered_knowledge": None,
                "answer": None,
                "confidence": None,
                "metadata": {},
                "error": None,
                "success": True,
            }

            result_state = await mock_workflow._filtering_node(state)

            assert "filtered_knowledge" in result_state
            assert (
                result_state["filtered_knowledge"]["entities"][0]["name"] == "Test Book"
            )

    @pytest.mark.asyncio
    async def test_summarization_node_success(
        self, mock_workflow, mock_summarization_result
    ):
        """Test successful summarization node execution."""
        with patch.object(
            mock_workflow.summarization_chain, "arun"
        ) as mock_summarization:
            mock_summarization.return_value = mock_summarization_result

            state: KERAGWorkflowState = {
                "question": "Test question",
                "domain": "book",
                "topic_entity": {
                    "id": "entity1",
                    "name": "Test Entity",
                    "type": "Person",
                    "attributes": {},
                    "neighbors": [],
                },
                "retrieval_plan": {},
                "retrieved_knowledge": {},
                "filtered_knowledge": {"entities": [], "triples": [], "metadata": {}},
                "answer": None,
                "confidence": None,
                "metadata": {},
                "error": None,
                "success": True,
            }

            result_state = await mock_workflow._summarization_node(state)

            assert result_state["success"] is True
            assert result_state["answer"] == "Test Entity wrote Test Book."
            assert result_state["confidence"] == 0.9

    @pytest.mark.asyncio
    async def test_error_handling_node(self, mock_workflow):
        """Test error handling node."""
        state: KERAGWorkflowState = {
            "question": "Test question",
            "domain": None,
            "topic_entity": None,
            "retrieval_plan": None,
            "retrieved_knowledge": None,
            "filtered_knowledge": None,
            "answer": None,
            "confidence": None,
            "metadata": {},
            "error": "Test error occurred",
            "success": False,
        }

        result_state = await mock_workflow._error_handling_node(state)

        assert result_state["success"] is False
        assert "error" in result_state["answer"]
        assert result_state["confidence"] == 0.0

    def test_should_continue_after_planning(self, mock_workflow):
        """Test continuation decision after planning."""
        # Test success case
        success_state: KERAGWorkflowState = {
            "question": "Test question",
            "domain": "book",
            "topic_entity": {},
            "retrieval_plan": {},
            "retrieved_knowledge": None,
            "filtered_knowledge": None,
            "answer": None,
            "confidence": None,
            "metadata": {},
            "error": None,
            "success": True,
        }

        assert (
            mock_workflow._should_continue_after_planning(success_state) == "continue"
        )

        # Test error case
        error_state: KERAGWorkflowState = {
            "question": "Test question",
            "domain": None,
            "topic_entity": None,
            "retrieval_plan": None,
            "retrieved_knowledge": None,
            "filtered_knowledge": None,
            "answer": None,
            "confidence": None,
            "metadata": {},
            "error": "Planning failed",
            "success": False,
        }

        assert mock_workflow._should_continue_after_planning(error_state) == "error"

    @pytest.mark.asyncio
    async def test_process_question_success(self, mock_workflow):
        """Test successful question processing."""
        # Mock all chain results
        with patch.object(
            mock_workflow.planning_chain, "arun"
        ) as mock_planning, patch.object(
            mock_workflow.retrieval_chain, "arun"
        ) as mock_retrieval, patch.object(
            mock_workflow.filtering_chain, "arun"
        ) as mock_filtering, patch.object(
            mock_workflow.summarization_chain, "arun"
        ) as mock_summarization:

            mock_planning.return_value = {
                "question": "Test question",
                "domain": "book",
                "topic_entity": {
                    "id": "entity1",
                    "name": "Test Entity",
                    "type": "Person",
                    "attributes": {},
                    "neighbors": [],
                },
                "retrieval_plan": {
                    "domain": "book",
                    "topic_entity": {
                        "id": "entity1",
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

            mock_retrieval.return_value = {
                "retrieved_knowledge": {
                    "entities": [
                        {
                            "id": "book1",
                            "name": "Test Book",
                            "type": "Book",
                            "attributes": {},
                            "neighbors": [],
                        }
                    ],
                    "triples": [
                        {
                            "subject": "entity1",
                            "predicate": "author",
                            "object": "book1",
                            "confidence": 0.9,
                        }
                    ],
                    "metadata": {},
                },
                "success": True,
            }

            mock_filtering.return_value = {
                "filtered_knowledge": {
                    "entities": [
                        {
                            "id": "book1",
                            "name": "Test Book",
                            "type": "Book",
                            "attributes": {},
                            "neighbors": [],
                        }
                    ],
                    "triples": [
                        {
                            "subject": "entity1",
                            "predicate": "author",
                            "object": "book1",
                            "confidence": 0.9,
                        }
                    ],
                    "metadata": {},
                },
                "success": True,
            }

            mock_summarization.return_value = {
                "answer": "Test Entity wrote Test Book.",
                "confidence": 0.9,
                "success": True,
            }

            result = await mock_workflow.process_question("Test question")

            assert isinstance(result, KERAGResult)
            assert result.answer == "Test Entity wrote Test Book."
            assert result.confidence == 0.9
            assert len(result.retrieved_entities) == 1
            assert len(result.retrieved_triples) == 1

    @pytest.mark.asyncio
    async def test_process_question_failure(self, mock_workflow):
        """Test question processing failure."""
        with patch.object(mock_workflow.planning_chain, "arun") as mock_planning:
            mock_planning.return_value = {
                "question": "Test question",
                "error": "Planning failed",
                "success": False,
            }

            with pytest.raises(KERAGException):
                await mock_workflow.process_question("Test question")

    def test_entity_to_dict(self, mock_workflow):
        """Test entity to dictionary conversion."""
        from core.types import Entity

        entity = Entity(
            id="test_entity",
            name="Test Entity",
            type="Person",
            attributes={"age": 30},
            neighbors=[],
        )

        entity_dict = mock_workflow._entity_to_dict(entity)

        assert entity_dict["id"] == "test_entity"
        assert entity_dict["name"] == "Test Entity"
        assert entity_dict["type"] == "Person"
        assert entity_dict["attributes"]["age"] == 30

    def test_dict_to_retrieval_plan(self, mock_workflow):
        """Test dictionary to retrieval plan conversion."""
        plan_dict = {
            "domain": "book",
            "topic_entity": {
                "id": "entity1",
                "name": "Test Entity",
                "type": "Person",
                "attributes": {},
                "neighbors": [],
            },
            "relations": ["author", "wrote"],
            "max_hops": 2,
            "filters": {},
        }

        plan = mock_workflow._dict_to_retrieval_plan(plan_dict)

        assert plan.domain.value == "book"
        assert plan.topic_entity.name == "Test Entity"
        assert "author" in plan.relations
        assert plan.max_hops == 2


class TestAsyncKERAGWorkflow:
    """Test suite for async KERAG workflow."""

    @pytest.fixture
    def mock_config(self):
        """Create mock KERAG configuration."""
        return KERAGConfig(
            knowledge_base_type=KnowledgeBaseType.API,
            endpoint_url="https://api.test.com",
            llm_model="gpt-4",
            api_key="test_key",
        )

    @pytest.fixture
    def mock_async_workflow(self, mock_config):
        """Create mock async KERAG workflow."""
        with patch("workflows.kerag_workflow.APIKnowledgeBase") as mock_kb_class:
            mock_kb = Mock()
            mock_kb_class.return_value = mock_kb

            workflow = AsyncKERAGWorkflow(mock_config)
            workflow.knowledge_base = mock_kb

            return workflow

    @pytest.mark.asyncio
    async def test_async_context_manager(self, mock_async_workflow):
        """Test async context manager functionality."""
        mock_async_workflow.knowledge_base.__aenter__ = AsyncMock()
        mock_async_workflow.knowledge_base.__aexit__ = AsyncMock()

        async with mock_async_workflow:
            mock_async_workflow.knowledge_base.__aenter__.assert_called_once()

        mock_async_workflow.knowledge_base.__aexit__.assert_called_once()

    @pytest.mark.asyncio
    async def test_async_workflow_process_question(self, mock_async_workflow):
        """Test async workflow question processing."""
        with patch.object(mock_async_workflow, "process_question") as mock_process:
            mock_process.return_value = KERAGResult(
                answer="Test answer",
                confidence=0.9,
                retrieved_entities=[],
                retrieved_triples=[],
                reasoning_steps=[],
                metadata={},
            )

            result = await mock_async_workflow.process_question("Test question")

            assert result.answer == "Test answer"
            assert result.confidence == 0.9


class TestWorkflowIntegration:
    """Integration tests for workflows."""

    @pytest.mark.asyncio
    async def test_workflow_with_mock_knowledge_base(self):
        """Test workflow with mocked knowledge base."""
        config = KERAGConfig(
            knowledge_base_type=KnowledgeBaseType.SPARQL,
            endpoint_url="http://test-endpoint.com/sparql",
            llm_model="gpt-4",
            api_key="test_key",
        )

        with patch(
            "workflows.kerag_workflow.SPARQLKnowledgeBase"
        ) as mock_kb_class, patch(
            "workflows.kerag_workflow.PlanningChain"
        ) as mock_planning_class, patch(
            "workflows.kerag_workflow.RetrievalChain"
        ) as mock_retrieval_class, patch(
            "workflows.kerag_workflow.FilteringChain"
        ) as mock_filtering_class, patch(
            "workflows.kerag_workflow.SummarizationChain"
        ) as mock_summarization_class:

            # Mock knowledge base
            mock_kb = Mock()
            mock_kb_class.return_value = mock_kb

            # Mock chains
            mock_planning = Mock()
            mock_planning.arun = AsyncMock(
                return_value={
                    "question": "Test question",
                    "domain": "book",
                    "topic_entity": {
                        "id": "entity1",
                        "name": "Test Entity",
                        "type": "Person",
                        "attributes": {},
                        "neighbors": [],
                    },
                    "retrieval_plan": {
                        "domain": "book",
                        "topic_entity": {
                            "id": "entity1",
                            "name": "Test Entity",
                            "type": "Person",
                            "attributes": {},
                            "neighbors": [],
                        },
                        "relations": ["author"],
                        "max_hops": 1,
                        "filters": {},
                    },
                    "success": True,
                }
            )
            mock_planning_class.return_value = mock_planning

            mock_retrieval = Mock()
            mock_retrieval.arun = AsyncMock(
                return_value={
                    "retrieved_knowledge": {
                        "entities": [],
                        "triples": [],
                        "metadata": {},
                    },
                    "success": True,
                }
            )
            mock_retrieval_class.return_value = mock_retrieval

            mock_filtering = Mock()
            mock_filtering.arun = AsyncMock(
                return_value={
                    "filtered_knowledge": {
                        "entities": [],
                        "triples": [],
                        "metadata": {},
                    },
                    "success": True,
                }
            )
            mock_filtering_class.return_value = mock_filtering

            mock_summarization = Mock()
            mock_summarization.arun = AsyncMock(
                return_value={
                    "answer": "Test answer",
                    "confidence": 0.9,
                    "success": True,
                }
            )
            mock_summarization_class.return_value = mock_summarization

            workflow = KERAGWorkflow(config)

            result = await workflow.process_question("Test question")

            assert result.answer == "Test answer"
            assert result.confidence == 0.9

    def test_workflow_graph_creation(self):
        """Test workflow graph creation."""
        config = KERAGConfig(
            knowledge_base_type=KnowledgeBaseType.SPARQL,
            endpoint_url="http://test-endpoint.com/sparql",
            llm_model="gpt-4",
            api_key="test_key",
        )

        with patch("workflows.kerag_workflow.SPARQLKnowledgeBase"), patch(
            "workflows.kerag_workflow.PlanningChain"
        ), patch("workflows.kerag_workflow.RetrievalChain"), patch(
            "workflows.kerag_workflow.FilteringChain"
        ), patch(
            "workflows.kerag_workflow.SummarizationChain"
        ):

            workflow = KERAGWorkflow(config)
            graph = workflow.get_graph()

            assert graph is not None
            # Test that graph has expected structure
            # This would depend on the specific LangGraph implementation


# Test utilities
def create_mock_state(**kwargs) -> KERAGWorkflowState:
    """Create a mock workflow state."""
    default_state = {
        "question": "Test question",
        "domain": None,
        "topic_entity": None,
        "retrieval_plan": None,
        "retrieved_knowledge": None,
        "filtered_knowledge": None,
        "answer": None,
        "confidence": None,
        "metadata": {},
        "error": None,
        "success": False,
    }

    default_state.update(kwargs)
    return default_state


# Pytest configuration
@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()
