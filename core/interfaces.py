"""
Abstract interfaces for KERAG components.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from .types import Entity, Triple, RetrievalResult, RetrievalPlan


class KnowledgeBaseInterface(ABC):
    """Abstract interface for knowledge base implementations."""

    @abstractmethod
    async def get_entity_info(self, entity_id: str) -> Entity:
        """Get detailed information about an entity."""
        pass

    @abstractmethod
    async def get_entity_neighbors(
        self, entity_id: str, relations: List[str] = None, max_hops: int = 1
    ) -> RetrievalResult:
        """Get neighbors of an entity up to max_hops."""
        pass

    @abstractmethod
    async def search_entities(
        self, query: str, entity_type: str = None, limit: int = 10
    ) -> List[Entity]:
        """Search for entities matching a query."""
        pass

    @abstractmethod
    async def get_relations(self, entity_id: str) -> List[str]:
        """Get all relations for an entity."""
        pass

    @abstractmethod
    async def execute_query(self, query: str) -> RetrievalResult:
        """Execute a custom query against the knowledge base."""
        pass


class PlanningInterface(ABC):
    """Abstract interface for planning phase."""

    @abstractmethod
    async def identify_topic_entity(self, question: str) -> Entity:
        """Identify the main topic entity from the question."""
        pass

    @abstractmethod
    async def detect_domain(self, question: str, topic_entity: Entity) -> str:
        """Detect the domain of the question."""
        pass

    @abstractmethod
    async def generate_retrieval_plan(
        self, question: str, topic_entity: Entity, domain: str
    ) -> RetrievalPlan:
        """Generate a retrieval plan based on question analysis."""
        pass


class RetrievalInterface(ABC):
    """Abstract interface for retrieval phase."""

    @abstractmethod
    async def retrieve_knowledge(self, plan: RetrievalPlan) -> RetrievalResult:
        """Retrieve knowledge based on the retrieval plan."""
        pass

    @abstractmethod
    async def expand_neighborhood(
        self, entity: Entity, relations: List[str], max_hops: int
    ) -> RetrievalResult:
        """Expand entity neighborhood with multi-hop exploration."""
        pass


class FilteringInterface(ABC):
    """Abstract interface for filtering phase."""

    @abstractmethod
    async def filter_knowledge(
        self, question: str, retrieved_knowledge: RetrievalResult
    ) -> RetrievalResult:
        """Filter retrieved knowledge based on relevance to the question."""
        pass

    @abstractmethod
    async def compute_relevance_scores(
        self, question: str, entities: List[Entity], triples: List[Triple]
    ) -> Dict[str, float]:
        """Compute relevance scores for entities and triples."""
        pass


class SummarizationInterface(ABC):
    """Abstract interface for summarization phase."""

    @abstractmethod
    async def generate_answer(
        self, question: str, filtered_knowledge: RetrievalResult
    ) -> str:
        """Generate final answer using Chain-of-Thought reasoning."""
        pass

    @abstractmethod
    async def compute_confidence(
        self, question: str, answer: str, knowledge: RetrievalResult
    ) -> float:
        """Compute confidence score for the generated answer."""
        pass


class ChainInterface(ABC):
    """Abstract interface for LangChain LCEL chains."""

    @abstractmethod
    async def arun(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Asynchronously run the chain."""
        pass

    @abstractmethod
    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Synchronously run the chain."""
        pass

    @abstractmethod
    def get_chain(self):
        """Get the underlying LangChain chain."""
        pass


class WorkflowInterface(ABC):
    """Abstract interface for LangGraph workflows."""

    @abstractmethod
    async def process_question(self, question: str) -> Dict[str, Any]:
        """Process a question through the entire KERAG pipeline."""
        pass

    @abstractmethod
    def get_graph(self):
        """Get the LangGraph workflow."""
        pass

    @abstractmethod
    def get_state_schema(self):
        """Get the state schema for the workflow."""
        pass
