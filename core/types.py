"""
Type definitions for KERAG implementation.
"""

from typing import Dict, List, Optional, Union, Any, Tuple
from dataclasses import dataclass
from enum import Enum
from pydantic import BaseModel, Field


class KnowledgeBaseType(str, Enum):
    """Supported knowledge base types."""

    SPARQL = "sparql"
    API = "api"


class DomainType(str, Enum):
    """Common domain types for question answering."""

    MOVIE = "movie"
    MUSIC = "music"
    SPORTS = "sports"
    FINANCE = "finance"
    BOOK = "book"
    PERSON = "person"
    ORGANIZATION = "organization"
    LOCATION = "location"
    GENERAL = "general"


@dataclass
class Entity:
    """Represents a knowledge graph entity."""

    id: str
    name: str
    type: str
    attributes: Dict[str, Any]
    neighbors: List["Entity"]


@dataclass
class Triple:
    """Represents a knowledge graph triple."""

    subject: str
    predicate: str
    object: Union[str, Entity]
    confidence: Optional[float] = None


@dataclass
class RetrievalPlan:
    """Represents a retrieval plan for KERAG."""

    domain: DomainType
    topic_entity: Entity
    relations: List[str]
    max_hops: int
    filters: Dict[str, Any]


@dataclass
class RetrievalResult:
    """Represents the result of knowledge retrieval."""

    entities: List[Entity]
    triples: List[Triple]
    metadata: Dict[str, Any]


@dataclass
class FilteredKnowledge:
    """Represents filtered knowledge for summarization."""

    relevant_entities: List[Entity]
    relevant_triples: List[Triple]
    relevance_scores: Dict[str, float]
    metadata: Dict[str, Any]


class KERAGState(BaseModel):
    """State management for KERAG workflow."""

    question: str = Field(description="The input question")
    domain: Optional[DomainType] = Field(default=None, description="Detected domain")
    topic_entity: Optional[Entity] = Field(
        default=None, description="Identified topic entity"
    )
    retrieval_plan: Optional[RetrievalPlan] = Field(
        default=None, description="Generated retrieval plan"
    )
    retrieved_knowledge: Optional[RetrievalResult] = Field(
        default=None, description="Retrieved knowledge"
    )
    filtered_knowledge: Optional[FilteredKnowledge] = Field(
        default=None, description="Filtered knowledge"
    )
    answer: Optional[str] = Field(default=None, description="Final answer")
    confidence: Optional[float] = Field(default=None, description="Answer confidence")
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata"
    )
    error: Optional[str] = Field(default=None, description="Error message if any")


class PlanningResult(BaseModel):
    """Result from the planning phase."""

    domain: DomainType
    topic_entity: Entity
    relations: List[str]
    max_hops: int
    confidence: float


class RetrievalConfig(BaseModel):
    """Configuration for retrieval phase."""

    max_hops: int = Field(
        default=3, description="Maximum number of hops for neighborhood expansion"
    )
    max_entities: int = Field(
        default=1000, description="Maximum number of entities to retrieve"
    )
    max_triples: int = Field(
        default=500, description="Maximum number of triples to retrieve"
    )
    relation_filters: List[str] = Field(
        default_factory=list, description="Relations to filter out"
    )
    include_attributes: bool = Field(
        default=True, description="Whether to include entity attributes"
    )
    timeout: int = Field(default=30, description="Timeout for retrieval operations")
    confidence_threshold: float = Field(
        default=0.5, description="Minimum confidence threshold for triples"
    )


class FilteringConfig(BaseModel):
    """Configuration for filtering phase."""

    relevance_threshold: float = Field(
        default=0.5, description="Minimum relevance score"
    )
    max_triples: int = Field(
        default=500, description="Maximum number of triples to keep"
    )
    max_entities: int = Field(
        default=1000, description="Maximum number of entities to keep"
    )
    use_llm_filtering: bool = Field(
        default=True, description="Whether to use LLM for filtering"
    )


class SummarizationConfig(BaseModel):
    """Configuration for summarization phase."""

    use_cot: bool = Field(
        default=True, description="Whether to use Chain-of-Thought reasoning"
    )
    max_tokens: int = Field(default=4000, description="Maximum tokens for LLM input")
    temperature: float = Field(
        default=0.1, description="LLM temperature for generation"
    )
    strategy: str = Field(
        default="cot", description="Summarization strategy (cot, direct, hybrid)"
    )


class KERAGConfig(BaseModel):
    """Main configuration for KERAG pipeline."""

    knowledge_base_type: KnowledgeBaseType
    endpoint_url: str
    llm_model: str = Field(default="gpt-4", description="LLM model to use")
    api_key: Optional[str] = Field(default=None, description="API key for LLM")

    # Phase-specific configurations
    retrieval_config: RetrievalConfig = Field(default_factory=RetrievalConfig)
    filtering_config: FilteringConfig = Field(default_factory=FilteringConfig)
    summarization_config: SummarizationConfig = Field(
        default_factory=SummarizationConfig
    )

    # Additional settings
    debug: bool = Field(default=False, description="Enable debug mode")
    timeout: int = Field(default=30, description="Timeout for API calls")


class KERAGResult(BaseModel):
    """Final result from KERAG pipeline."""

    answer: str
    confidence: float
    retrieved_entities: List[Entity]
    retrieved_triples: List[Triple]
    reasoning_steps: List[str]
    metadata: Dict[str, Any]
