"""
Filtering chain implementation for KERAG using LangChain LCEL.
"""

import asyncio
from typing import Dict, Any, List, Optional
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI

from core.interfaces import FilteringInterface, ChainInterface
from core.types import (
    Entity,
    Triple,
    RetrievalResult,
    FilteredKnowledge,
    FilteringConfig,
)
from core.exceptions import FilteringError
from config.settings import PromptConfig


class FilteringChain(FilteringInterface, ChainInterface):
    """Filtering chain for relevance-based knowledge filtering."""

    def __init__(
        self,
        llm_model: str = "gpt-4",
        api_key: Optional[str] = None,
        config: Optional[FilteringConfig] = None,
    ):
        """
        Initialize filtering chain.

        Args:
            llm_model: LLM model to use
            api_key: API key for LLM
            config: Filtering configuration
        """
        self.llm_model = llm_model
        self.config = config or FilteringConfig()

        # Validate configuration
        self._validate_config()

        # Initialize LLM
        if llm_model.startswith("gpt"):
            self.llm = ChatOpenAI(model=llm_model, temperature=0.1, api_key=api_key)
        elif llm_model.startswith("claude"):
            self.llm = ChatAnthropic(model=llm_model, temperature=0.1, api_key=api_key)
        elif llm_model.startswith("gemini"):
            self.llm = ChatGoogleGenerativeAI(
                model=llm_model, temperature=0.1, google_api_key=api_key
            )
        elif llm_model.startswith("grok"):
            # For Grok, we'll use OpenAI-compatible interface since xAI supports it
            self.llm = ChatOpenAI(
                model=llm_model,
                temperature=0.1,
                api_key=api_key,
                base_url="https://api.x.ai/v1",  # xAI's OpenAI-compatible endpoint
            )
        else:
            self.llm = ChatOpenAI(model=llm_model, temperature=0.1, api_key=api_key)

        # Setup filtering chain
        self._setup_chain()

    def _validate_config(self):
        """Validate filtering configuration."""
        if not isinstance(self.config, FilteringConfig):
            raise ValueError("Invalid filtering configuration")

        if not (0.0 <= self.config.relevance_threshold <= 1.0):
            raise ValueError("Relevance threshold must be between 0.0 and 1.0")

        if self.config.max_triples is not None and self.config.max_triples < 0:
            raise ValueError("Max triples must be non-negative")

    def _setup_chain(self):
        """Setup LangChain LCEL filtering chain."""

        # Relevance scoring prompt
        relevance_prompt = ChatPromptTemplate.from_template(
            PromptConfig.FILTERING_PROMPTS["relevance_scoring"]
        )

        self.relevance_chain = relevance_prompt | self.llm | JsonOutputParser()

        # Main filtering chain
        self.filtering_chain = RunnablePassthrough.assign(
            filtered_knowledge=RunnableLambda(self._filter_knowledge)
        )

    async def filter_knowledge(
        self, question: str, retrieved_knowledge: RetrievalResult
    ) -> RetrievalResult:
        """Filter retrieved knowledge based on relevance to the question."""
        try:
            # Compute relevance scores
            relevance_scores = await self.compute_relevance_scores(
                question, retrieved_knowledge.entities, retrieved_knowledge.triples
            )

            # Filter based on scores
            filtered_entities = []
            filtered_triples = []

            # Filter entities
            for entity in retrieved_knowledge.entities:
                entity_score = relevance_scores.get(f"entity_{entity.id}", 0.0)
                if entity_score >= self.config.relevance_threshold:
                    filtered_entities.append(entity)

            # Filter triples
            for i, triple in enumerate(retrieved_knowledge.triples):
                triple_score = relevance_scores.get(f"triple_{i}", 0.0)
                if triple_score >= self.config.relevance_threshold:
                    filtered_triples.append(triple)

            # Limit results if configured
            if self.config.max_triples:
                filtered_triples = filtered_triples[: self.config.max_triples]

            if self.config.max_entities:
                filtered_entities = filtered_entities[: self.config.max_entities]

            # Create filtered knowledge object
            filtered_knowledge = FilteredKnowledge(
                relevant_entities=filtered_entities,
                relevant_triples=filtered_triples,
                relevance_scores=relevance_scores,
                metadata={
                    "original_entities": len(retrieved_knowledge.entities),
                    "original_triples": len(retrieved_knowledge.triples),
                    "filtered_entities": len(filtered_entities),
                    "filtered_triples": len(filtered_triples),
                    "relevance_threshold": self.config.relevance_threshold,
                    "filtering_method": "llm_based",
                },
            )

            # Return as RetrievalResult for compatibility
            return RetrievalResult(
                entities=filtered_entities,
                triples=filtered_triples,
                metadata={
                    **retrieved_knowledge.metadata,
                    **filtered_knowledge.metadata,
                },
            )

        except Exception as e:
            raise FilteringError(f"Failed to filter knowledge: {str(e)}")

    async def compute_relevance_scores(
        self, question: str, entities: List[Entity], triples: List[Triple]
    ) -> Dict[str, float]:
        """Compute relevance scores for entities and triples."""
        try:
            # Prepare knowledge representation for LLM
            knowledge_text = self._prepare_knowledge_text(entities, triples)

            # Get relevance scores from LLM
            result = await self.relevance_chain.ainvoke(
                {"question": question, "knowledge": knowledge_text}
            )

            # Parse scores
            entity_scores = result.get("entity_scores", {})
            triple_scores = result.get("triple_scores", {})

            # Create comprehensive score dictionary
            scores = {}

            # Add entity scores
            for entity in entities:
                entity_key = f"entity_{entity.id}"
                scores[entity_key] = entity_scores.get(entity.id, 0.0)

            # Add triple scores
            for i, triple in enumerate(triples):
                triple_key = f"triple_{i}"
                scores[triple_key] = triple_scores.get(str(i), 0.0)

            return scores

        except Exception as e:
            # Fallback to simple keyword-based scoring
            return await self._compute_fallback_scores(question, entities, triples)

    def _prepare_knowledge_text(
        self, entities: List[Entity], triples: List[Triple]
    ) -> str:
        """Prepare knowledge representation for LLM processing."""
        text_parts = []

        # Add entities
        text_parts.append("ENTITIES:")
        for entity in entities:
            text_parts.append(f"- {entity.id}: {entity.name} ({entity.type})")
            for attr_key, attr_value in entity.attributes.items():
                text_parts.append(f"  {attr_key}: {attr_value}")

        # Add triples
        text_parts.append("\nTRIPLES:")
        for i, triple in enumerate(triples):
            text_parts.append(
                f"- {i}: {triple.subject} --{triple.predicate}--> {triple.object}"
            )
            if triple.confidence:
                text_parts.append(f"  confidence: {triple.confidence}")

        return "\n".join(text_parts)

    async def _compute_fallback_scores(
        self, question: str, entities: List[Entity], triples: List[Triple]
    ) -> Dict[str, float]:
        """Compute fallback relevance scores using keyword matching."""
        scores = {}
        question_words = set(question.lower().split())

        # Score entities based on name and attribute matching
        for entity in entities:
            entity_key = f"entity_{entity.id}"
            score = 0.0

            # Check entity name
            entity_words = set(entity.name.lower().split())
            name_overlap = len(question_words.intersection(entity_words))
            score += min(name_overlap * 0.3, 1.0)

            # Check entity attributes
            for attr_value in entity.attributes.values():
                if isinstance(attr_value, str):
                    attr_words = set(attr_value.lower().split())
                    attr_overlap = len(question_words.intersection(attr_words))
                    score += min(attr_overlap * 0.1, 0.5)

            scores[entity_key] = min(score, 1.0)

        # Score triples based on content matching
        for i, triple in enumerate(triples):
            triple_key = f"triple_{i}"
            score = 0.0

            # Check triple components
            triple_text = f"{triple.subject} {triple.predicate} {triple.object}".lower()
            triple_words = set(triple_text.split())
            overlap = len(question_words.intersection(triple_words))
            score = min(overlap * 0.2, 1.0)

            scores[triple_key] = score

        return scores

    async def arun(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Asynchronously run the filtering chain."""
        try:
            question = input_data["question"]
            retrieved_knowledge = input_data["retrieved_knowledge"]

            # Filter knowledge
            filtered_knowledge = await self.filter_knowledge(
                question, retrieved_knowledge
            )

            return {
                **input_data,
                "filtered_knowledge": filtered_knowledge,
                "success": True,
            }

        except Exception as e:
            return {**input_data, "error": str(e), "success": False}

    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Synchronously run the filtering chain."""
        return asyncio.run(self.arun(input_data))

    def get_chain(self):
        """Get the underlying LangChain chain."""
        return self.filtering_chain

    async def _filter_knowledge(self, input_data: Dict[str, Any]) -> RetrievalResult:
        """Internal method for chain execution."""
        try:
            question = input_data.get("question")
            retrieved_knowledge = input_data.get("retrieved_knowledge")

            if not question or not retrieved_knowledge:
                raise FilteringError("Missing question or retrieved knowledge")

            return await self.filter_knowledge(question, retrieved_knowledge)
        except Exception as e:
            raise FilteringError(f"Chain filtering failed: {str(e)}")


class AdvancedFilteringChain(FilteringChain):
    """Advanced filtering chain with multiple filtering strategies."""

    async def filter_knowledge(
        self, question: str, retrieved_knowledge: RetrievalResult
    ) -> RetrievalResult:
        """Advanced filtering with multiple strategies."""
        try:
            # Strategy 1: LLM-based filtering
            llm_result = await super().filter_knowledge(question, retrieved_knowledge)

            # Strategy 2: Semantic similarity filtering
            semantic_result = await self._semantic_filtering(
                question, retrieved_knowledge
            )

            # Strategy 3: Graph-based filtering
            graph_result = await self._graph_based_filtering(
                question, retrieved_knowledge
            )

            # Combine results using ensemble approach
            combined_result = await self._combine_filtering_results(
                question, [llm_result, semantic_result, graph_result]
            )

            return combined_result

        except Exception as e:
            raise FilteringError(f"Advanced filtering failed: {str(e)}")

    async def _semantic_filtering(
        self, question: str, retrieved_knowledge: RetrievalResult
    ) -> RetrievalResult:
        """Semantic similarity-based filtering."""
        # This would use embeddings or other semantic similarity methods
        # For now, return the original result
        return retrieved_knowledge

    async def _graph_based_filtering(
        self, question: str, retrieved_knowledge: RetrievalResult
    ) -> RetrievalResult:
        """Graph-based filtering using connectivity patterns."""
        # This would analyze graph connectivity and centrality
        # For now, return the original result
        return retrieved_knowledge

    async def _combine_filtering_results(
        self, question: str, results: List[RetrievalResult]
    ) -> RetrievalResult:
        """Combine multiple filtering results using ensemble approach."""
        # Simple intersection for now - could be more sophisticated
        if not results:
            return RetrievalResult(entities=[], triples=[], metadata={})

        # Start with first result
        combined_entities = set(entity.id for entity in results[0].entities)
        combined_triples = results[0].triples.copy()

        # Intersect with other results
        for result in results[1:]:
            result_entities = set(entity.id for entity in result.entities)
            combined_entities = combined_entities.intersection(result_entities)

            # For triples, keep those that appear in multiple results
            combined_triples = [
                triple
                for triple in combined_triples
                if any(
                    triple.subject == t.subject
                    and triple.predicate == t.predicate
                    and triple.object == t.object
                    for t in result.triples
                )
            ]

        # Reconstruct entities
        final_entities = []
        for entity in results[0].entities:
            if entity.id in combined_entities:
                final_entities.append(entity)

        return RetrievalResult(
            entities=final_entities,
            triples=combined_triples,
            metadata={
                "filtering_strategy": "ensemble",
                "num_strategies": len(results),
                "final_entities": len(final_entities),
                "final_triples": len(combined_triples),
            },
        )

    async def _hybrid_filtering(
        self, question: str, retrieved_knowledge: RetrievalResult
    ) -> RetrievalResult:
        """Hybrid filtering combining multiple strategies."""
        try:
            # Combine LLM and semantic filtering
            llm_result = await super().filter_knowledge(question, retrieved_knowledge)
            semantic_result = await self._semantic_filtering(
                question, retrieved_knowledge
            )

            # Use intersection of both results
            return await self._combine_filtering_results(
                question, [llm_result, semantic_result]
            )
        except Exception:
            # Fallback to original result
            return retrieved_knowledge
