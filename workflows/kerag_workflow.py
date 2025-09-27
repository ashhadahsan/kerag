"""
LangGraph workflow implementation for KERAG pipeline.
"""

import asyncio
from typing import Dict, Any, List, Optional, TypedDict
from langgraph.graph import StateGraph, END

from core.interfaces import WorkflowInterface
from core.types import KERAGState, KERAGConfig, KERAGResult
from core.exceptions import (
    KERAGException,
    PlanningError,
    RetrievalError,
    FilteringError,
    SummarizationError,
)
from chains.planning import PlanningChain
from chains.retrieval import RetrievalChain
from chains.filtering import FilteringChain
from chains.summarization import SummarizationChain
from knowledge_bases.sparql_kb import SPARQLKnowledgeBase
from knowledge_bases.api_kb import APIKnowledgeBase
from config.settings import KnowledgeBaseType


class KERAGWorkflowState(TypedDict):
    """State type for KERAG workflow."""

    question: str
    domain: Optional[str]
    topic_entity: Optional[Dict[str, Any]]
    retrieval_plan: Optional[Dict[str, Any]]
    retrieved_knowledge: Optional[Dict[str, Any]]
    filtered_knowledge: Optional[Dict[str, Any]]
    answer: Optional[str]
    confidence: Optional[float]
    metadata: Dict[str, Any]
    error: Optional[str]
    success: bool


class KERAGWorkflow(WorkflowInterface):
    """LangGraph workflow for the complete KERAG pipeline."""

    def __init__(self, config: KERAGConfig):
        """
        Initialize KERAG workflow.

        Args:
            config: KERAG configuration
        """
        self.config = config

        # Initialize knowledge base
        self.knowledge_base = self._create_knowledge_base()

        # Initialize chains
        self.planning_chain = PlanningChain(
            llm_model=config.llm_model, api_key=config.api_key
        )

        self.retrieval_chain = RetrievalChain(
            knowledge_base=self.knowledge_base, config=config.retrieval_config
        )

        self.filtering_chain = FilteringChain(
            llm_model=config.llm_model,
            api_key=config.api_key,
            config=config.filtering_config,
        )

        self.summarization_chain = SummarizationChain(
            llm_model=config.llm_model,
            api_key=config.api_key,
            config=config.summarization_config,
        )

        # Create workflow graph
        self.graph = self._create_workflow_graph()

    def _create_knowledge_base(self):
        """Create knowledge base based on configuration."""
        if self.config.knowledge_base_type == KnowledgeBaseType.SPARQL:
            return SPARQLKnowledgeBase(
                endpoint_url=self.config.endpoint_url, timeout=self.config.timeout
            )
        elif self.config.knowledge_base_type == KnowledgeBaseType.API:
            return APIKnowledgeBase(
                base_url=self.config.endpoint_url,
                api_key=getattr(self.config, "api_key", None),
                timeout=self.config.timeout,
            )
        else:
            raise KERAGException(
                f"Unsupported knowledge base type: {self.config.knowledge_base_type}"
            )

    def _create_workflow_graph(self) -> StateGraph:
        """Create the LangGraph workflow."""

        # Define workflow graph
        workflow = StateGraph(KERAGWorkflowState)

        # Add nodes
        workflow.add_node("planning", self._planning_node)
        workflow.add_node("retrieval", self._retrieval_node)
        workflow.add_node("filtering", self._filtering_node)
        workflow.add_node("summarization", self._summarization_node)
        workflow.add_node("error_handling", self._error_handling_node)

        # Define edges
        workflow.add_edge("planning", "retrieval")
        workflow.add_edge("retrieval", "filtering")
        workflow.add_edge("filtering", "summarization")
        workflow.add_edge("summarization", END)

        # Add conditional edges for error handling
        workflow.add_conditional_edges(
            "planning",
            self._should_continue_after_planning,
            {"continue": "retrieval", "error": "error_handling"},
        )

        workflow.add_conditional_edges(
            "retrieval",
            self._should_continue_after_retrieval,
            {"continue": "filtering", "error": "error_handling"},
        )

        workflow.add_conditional_edges(
            "filtering",
            self._should_continue_after_filtering,
            {"continue": "summarization", "error": "error_handling"},
        )

        workflow.add_conditional_edges(
            "summarization",
            self._should_continue_after_summarization,
            {"continue": END, "error": "error_handling"},
        )

        # Set entry point
        workflow.set_entry_point("planning")

        return workflow.compile()

    async def _planning_node(self, state: KERAGWorkflowState) -> KERAGWorkflowState:
        """Planning phase node."""
        try:
            # Run planning chain
            planning_result = await self.planning_chain.arun(
                {"question": state["question"]}
            )

            if not planning_result.get("success", False):
                raise PlanningError(planning_result.get("error", "Planning failed"))

            # Update state (don't update question to avoid LangGraph conflicts)
            # Handle both Entity objects and dictionaries
            topic_entity = planning_result["topic_entity"]
            if hasattr(topic_entity, "id"):  # It's an Entity object
                topic_entity_dict = self._entity_to_dict(topic_entity)
            else:  # It's already a dictionary
                topic_entity_dict = topic_entity

            retrieval_plan = planning_result["retrieval_plan"]
            if hasattr(retrieval_plan, "domain"):  # It's a RetrievalPlan object
                retrieval_plan_dict = self._retrieval_plan_to_dict(retrieval_plan)
            else:  # It's already a dictionary
                retrieval_plan_dict = retrieval_plan

            return {
                "domain": planning_result["domain"],
                "topic_entity": topic_entity_dict,
                "retrieval_plan": retrieval_plan_dict,
                "success": True,
            }

        except Exception as e:
            return {"error": f"Planning failed: {str(e)}", "success": False}

    async def _retrieval_node(self, state: KERAGWorkflowState) -> KERAGWorkflowState:
        """Retrieval phase node."""
        try:
            # Reconstruct retrieval plan from state
            retrieval_plan = self._dict_to_retrieval_plan(state["retrieval_plan"])

            # Run retrieval chain
            retrieval_result = await self.retrieval_chain.arun(
                {"retrieval_plan": retrieval_plan}
            )

            if not retrieval_result.get("success", False):
                raise RetrievalError(retrieval_result.get("error", "Retrieval failed"))

            # Update state
            return {
                "retrieved_knowledge": self._retrieval_result_to_dict(
                    retrieval_result["retrieved_knowledge"]
                ),
                "success": True,
            }

        except Exception as e:
            return {"error": f"Retrieval failed: {str(e)}", "success": False}

    async def _filtering_node(self, state: KERAGWorkflowState) -> KERAGWorkflowState:
        """Filtering phase node."""
        try:
            # Reconstruct retrieved knowledge from state
            retrieved_knowledge = self._dict_to_retrieval_result(
                state["retrieved_knowledge"]
            )

            # Run filtering chain
            filtering_result = await self.filtering_chain.arun(
                {
                    "question": state["question"],
                    "retrieved_knowledge": retrieved_knowledge,
                }
            )

            if not filtering_result.get("success", False):
                raise FilteringError(filtering_result.get("error", "Filtering failed"))

            # Update state
            return {
                "filtered_knowledge": self._retrieval_result_to_dict(
                    filtering_result["filtered_knowledge"]
                ),
                "success": True,
            }

        except Exception as e:
            return {"error": f"Filtering failed: {str(e)}", "success": False}

    async def _summarization_node(
        self, state: KERAGWorkflowState
    ) -> KERAGWorkflowState:
        """Summarization phase node."""
        try:
            # Reconstruct filtered knowledge from state
            filtered_knowledge = self._dict_to_retrieval_result(
                state["filtered_knowledge"]
            )

            # Run summarization chain
            summarization_result = await self.summarization_chain.arun(
                {
                    "question": state["question"],
                    "filtered_knowledge": filtered_knowledge,
                }
            )

            if not summarization_result.get("success", False):
                raise SummarizationError(
                    summarization_result.get("error", "Summarization failed")
                )

            # Update state
            return {
                "answer": summarization_result["answer"],
                "confidence": summarization_result["confidence"],
                "success": True,
            }

        except Exception as e:
            return {
                "error": f"Summarization failed: {str(e)}",
                "success": False,
            }

    async def _error_handling_node(
        self, state: KERAGWorkflowState
    ) -> KERAGWorkflowState:
        """Error handling node."""
        # Provide fallback answer
        return {
            "answer": f"I encountered an error while processing your question: {state.get('error', 'Unknown error')}. Please try rephrasing your question or check if the knowledge base is accessible.",
            "confidence": 0.0,
            "success": False,
        }

    def _should_continue_after_planning(self, state: KERAGWorkflowState) -> str:
        """Determine next step after planning."""
        return "continue" if not state.get("error") else "error"

    def _should_continue_after_retrieval(self, state: KERAGWorkflowState) -> str:
        """Determine next step after retrieval."""
        return "continue" if not state.get("error") else "error"

    def _should_continue_after_filtering(self, state: KERAGWorkflowState) -> str:
        """Determine next step after filtering."""
        return "continue" if not state.get("error") else "error"

    def _should_continue_after_summarization(self, state: KERAGWorkflowState) -> str:
        """Determine next step after summarization."""
        return "continue" if not state.get("error") else "error"

    # Helper methods for state serialization
    def _entity_to_dict(self, entity) -> Dict[str, Any]:
        """Convert Entity to dictionary for state storage."""
        return {
            "id": entity.id,
            "name": entity.name,
            "type": entity.type,
            "attributes": entity.attributes,
            "neighbors": [{"id": n.id, "name": n.name} for n in entity.neighbors],
        }

    def _retrieval_plan_to_dict(self, plan) -> Dict[str, Any]:
        """Convert RetrievalPlan to dictionary for state storage."""
        return {
            "domain": plan.domain.value,
            "topic_entity": self._entity_to_dict(plan.topic_entity),
            "relations": plan.relations,
            "max_hops": plan.max_hops,
            "filters": plan.filters,
        }

    def _retrieval_result_to_dict(self, result) -> Dict[str, Any]:
        """Convert RetrievalResult to dictionary for state storage."""
        # Handle both RetrievalResult objects and dictionaries
        if hasattr(result, "entities"):  # It's a RetrievalResult object
            return {
                "entities": [self._entity_to_dict(e) for e in result.entities],
                "triples": [
                    {
                        "subject": t.subject,
                        "predicate": t.predicate,
                        "object": t.object,
                        "confidence": t.confidence,
                    }
                    for t in result.triples
                ],
                "metadata": result.metadata,
            }
        else:  # It's already a dictionary
            return result

    def _dict_to_retrieval_plan(self, plan_dict: Dict[str, Any]):
        """Convert dictionary to RetrievalPlan."""
        from core.types import Entity, RetrievalPlan, DomainType

        topic_entity = Entity(
            id=plan_dict["topic_entity"]["id"],
            name=plan_dict["topic_entity"]["name"],
            type=plan_dict["topic_entity"]["type"],
            attributes=plan_dict["topic_entity"]["attributes"],
            neighbors=[],
        )

        return RetrievalPlan(
            domain=DomainType(plan_dict["domain"]),
            topic_entity=topic_entity,
            relations=plan_dict["relations"],
            max_hops=plan_dict["max_hops"],
            filters=plan_dict["filters"],
        )

    def _dict_to_retrieval_result(self, result_dict: Dict[str, Any]):
        """Convert dictionary to RetrievalResult."""
        from core.types import Entity, Triple, RetrievalResult

        entities = [
            Entity(
                id=e["id"],
                name=e["name"],
                type=e["type"],
                attributes=e["attributes"],
                neighbors=[],
            )
            for e in result_dict["entities"]
        ]

        triples = [
            Triple(
                subject=t["subject"],
                predicate=t["predicate"],
                object=t["object"],
                confidence=t.get("confidence"),
            )
            for t in result_dict["triples"]
        ]

        return RetrievalResult(
            entities=entities, triples=triples, metadata=result_dict["metadata"]
        )

    async def process_question(self, question: str) -> KERAGResult:
        """Process a question through the entire KERAG pipeline."""
        try:
            # Initialize state
            initial_state: KERAGWorkflowState = {
                "question": question,
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

            # Run workflow
            final_state = await self.graph.ainvoke(initial_state)

            # Create result
            if not final_state.get("error"):
                return KERAGResult(
                    answer=final_state["answer"],
                    confidence=final_state["confidence"],
                    retrieved_entities=self._dict_to_retrieval_result(
                        final_state["retrieved_knowledge"]
                    ).entities,
                    retrieved_triples=self._dict_to_retrieval_result(
                        final_state["retrieved_knowledge"]
                    ).triples,
                    reasoning_steps=[],  # Could be extracted from CoT reasoning
                    metadata={
                        **final_state["metadata"],
                        "domain": final_state["domain"],
                        "topic_entity": final_state["topic_entity"],
                    },
                )
            else:
                raise KERAGException(final_state.get("error", "Workflow failed"))

        except Exception as e:
            raise KERAGException(f"Failed to process question: {str(e)}")

    def get_graph(self):
        """Get the LangGraph workflow."""
        return self.graph

    def get_state_schema(self):
        """Get the state schema for the workflow."""
        return KERAGWorkflowState


class AsyncKERAGWorkflow(KERAGWorkflow):
    """Async-compatible KERAG workflow."""

    async def __aenter__(self):
        """Async context manager entry."""
        if hasattr(self.knowledge_base, "__aenter__"):
            await self.knowledge_base.__aenter__()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if hasattr(self.knowledge_base, "__aexit__"):
            await self.knowledge_base.__aexit__(exc_type, exc_val, exc_tb)
