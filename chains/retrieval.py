"""
Retrieval chain implementation for KERAG using LangChain LCEL.
"""

import asyncio
from typing import Dict, Any, List, Optional
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate

from core.interfaces import RetrievalInterface, ChainInterface
from core.types import Entity, RetrievalPlan, RetrievalResult, RetrievalConfig
from core.exceptions import RetrievalError
from knowledge_bases.sparql_kb import SPARQLKnowledgeBase
from knowledge_bases.api_kb import APIKnowledgeBase
from config.settings import KnowledgeBaseType


class RetrievalChain(RetrievalInterface, ChainInterface):
    """Retrieval chain for multi-hop neighborhood expansion."""

    def __init__(self, knowledge_base: Any, config: RetrievalConfig):
        """
        Initialize retrieval chain.

        Args:
            knowledge_base: Knowledge base implementation
            config: Retrieval configuration
        """
        self.knowledge_base = knowledge_base
        self.config = config

        # Validate configuration
        self._validate_config()

        # Setup retrieval chain
        self._setup_chain()

    def _validate_config(self):
        """Validate retrieval configuration."""
        if not isinstance(self.config, RetrievalConfig):
            raise ValueError("Invalid retrieval configuration")

        if self.config.max_hops < 0:
            raise ValueError("Max hops must be non-negative")

        if self.config.max_entities < 0:
            raise ValueError("Max entities must be non-negative")

        if self.config.max_triples is not None and self.config.max_triples < 0:
            raise ValueError("Max triples must be non-negative")

        if self.config.timeout < 0:
            raise ValueError("Timeout must be non-negative")

    def _setup_chain(self):
        """Setup LangChain LCEL retrieval chain."""
        self.retrieval_chain = RunnablePassthrough.assign(
            retrieved_knowledge=RunnableLambda(self._retrieve_knowledge)
        )

    async def retrieve_knowledge(self, plan: RetrievalPlan) -> RetrievalResult:
        """Retrieve knowledge based on the retrieval plan."""
        try:
            # Start with the topic entity
            source_entity = plan.topic_entity

            # Get neighbors with multi-hop expansion
            result = await self.expand_neighborhood(
                source_entity, plan.relations, plan.max_hops
            )

            # Apply additional filtering based on plan filters
            filtered_result = await self._apply_plan_filters(result, plan)

            return filtered_result

        except Exception as e:
            raise RetrievalError(f"Failed to retrieve knowledge: {str(e)}")

    async def expand_neighborhood(
        self,
        entity: Entity,
        relations: List[str],
        max_hops: int,
        filters: Dict[str, Any] = None,
    ) -> RetrievalResult:
        """Expand entity neighborhood with multi-hop exploration."""
        try:
            all_entities = [entity]
            all_triples = []
            seen_entities = {entity.id}

            current_entities = [entity]

            for hop in range(max_hops):
                next_entities = []

                for current_entity in current_entities:
                    try:
                        # Get neighbors for current entity
                        hop_result = await self.knowledge_base.get_entity_neighbors(
                            current_entity.id, relations=relations, max_hops=1
                        )

                        # Add new entities (avoid duplicates)
                        for neighbor_entity in hop_result.entities:
                            if neighbor_entity.id not in seen_entities:
                                all_entities.append(neighbor_entity)
                                next_entities.append(neighbor_entity)
                                seen_entities.add(neighbor_entity.id)

                        # Add triples
                        all_triples.extend(hop_result.triples)

                        # Limit entities if configured
                        if len(all_entities) >= self.config.max_entities:
                            break

                    except Exception as e:
                        # Continue with other entities if one fails
                        continue

                current_entities = next_entities

                # Stop if no new entities found or limit reached
                if (
                    not current_entities
                    or len(all_entities) >= self.config.max_entities
                ):
                    break

            return RetrievalResult(
                entities=all_entities[: self.config.max_entities],
                triples=all_triples,
                metadata={
                    "source_entity": entity.id,
                    "max_hops": max_hops,
                    "relations": relations,
                    "relations_filter": relations,
                    "total_entities": len(all_entities),
                    "total_triples": len(all_triples),
                    "expansion_method": "multi_hop_neighborhood",
                },
            )

        except Exception as e:
            raise RetrievalError(f"Failed to expand neighborhood: {str(e)}")

    async def _apply_plan_filters(
        self, result: RetrievalResult, plan: RetrievalPlan
    ) -> RetrievalResult:
        """Apply additional filters based on the retrieval plan."""
        try:
            filtered_entities = result.entities.copy()
            filtered_triples = result.triples.copy()

            # Apply relation filters if specified
            if plan.filters.get("relations"):
                filtered_triples = [
                    triple
                    for triple in filtered_triples
                    if any(rel in triple.predicate for rel in plan.filters["relations"])
                ]

            # Apply type filters if specified
            if plan.filters.get("entity_types"):
                allowed_types = plan.filters["entity_types"]
                filtered_entities = [
                    entity
                    for entity in filtered_entities
                    if entity.type in allowed_types
                ]

            # Apply confidence filters if specified
            if plan.filters.get("min_confidence"):
                min_conf = plan.filters["min_confidence"]
                filtered_triples = [
                    triple
                    for triple in filtered_triples
                    if triple.confidence is None or triple.confidence >= min_conf
                ]

            return RetrievalResult(
                entities=filtered_entities,
                triples=filtered_triples,
                metadata={
                    **result.metadata,
                    "applied_filters": plan.filters,
                    "filtered_entities": len(filtered_entities),
                    "filtered_triples": len(filtered_triples),
                },
            )

        except Exception as e:
            # Return original result if filtering fails
            return result

    async def arun(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Asynchronously run the retrieval chain."""
        try:
            retrieval_plan = input_data["retrieval_plan"]

            # Retrieve knowledge based on plan
            retrieved_knowledge = await self.retrieve_knowledge(retrieval_plan)

            return {
                **input_data,
                "retrieved_knowledge": retrieved_knowledge,
                "success": True,
            }

        except Exception as e:
            return {**input_data, "error": str(e), "success": False}

    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Synchronously run the retrieval chain."""
        return asyncio.run(self.arun(input_data))

    def get_chain(self):
        """Get the underlying LangChain chain."""
        return self.retrieval_chain

    async def _retrieve_knowledge(self, input_data: Dict[str, Any]) -> RetrievalResult:
        """Internal method for chain execution."""
        try:
            retrieval_plan = input_data.get("retrieval_plan")
            if not retrieval_plan:
                raise RetrievalError("No retrieval plan provided")

            return await self.retrieve_knowledge(retrieval_plan)
        except Exception as e:
            raise RetrievalError(f"Chain retrieval failed: {str(e)}")


class AdaptiveRetrievalChain(RetrievalChain):
    """Adaptive retrieval chain that adjusts strategy based on entity characteristics."""

    async def expand_neighborhood(
        self, entity: Entity, relations: List[str], max_hops: int
    ) -> RetrievalResult:
        """Adaptive neighborhood expansion with dynamic strategy selection."""
        try:
            # Analyze entity characteristics
            entity_analysis = await self._analyze_entity(entity)

            # Adjust strategy based on analysis
            if entity_analysis["is_highly_connected"]:
                # Use more conservative expansion for highly connected entities
                adjusted_max_hops = min(max_hops, 2)
                adjusted_relations = relations[:5]  # Limit relations
            elif entity_analysis["is_isolated"]:
                # Use more aggressive expansion for isolated entities
                adjusted_max_hops = min(max_hops + 1, 4)
                adjusted_relations = relations
            else:
                # Use standard expansion
                adjusted_max_hops = max_hops
                adjusted_relations = relations

            # Perform adaptive retrieval
            result = await super().expand_neighborhood(
                entity, adjusted_relations, adjusted_max_hops
            )

            # Add analysis metadata
            result.metadata.update(
                {
                    "entity_analysis": entity_analysis,
                    "adjusted_max_hops": adjusted_max_hops,
                    "adjusted_relations": adjusted_relations,
                    "strategy": "adaptive",
                }
            )

            return result

        except Exception as e:
            raise RetrievalError(f"Adaptive retrieval failed: {str(e)}")

    async def _analyze_entity(self, entity: Entity) -> Dict[str, Any]:
        """Analyze entity characteristics for adaptive retrieval."""
        try:
            # Get entity relations
            relations = await self.knowledge_base.get_relations(entity.id)

            # Analyze connectivity
            relation_count = len(relations)
            is_highly_connected = relation_count > 20
            is_isolated = relation_count < 3

            # Analyze relation types
            relation_types = set()
            for rel in relations:
                # Extract relation type (simplified)
                if "/" in rel:
                    rel_type = rel.split("/")[-1]
                    relation_types.add(rel_type)

            return {
                "relation_count": relation_count,
                "relation_types": list(relation_types),
                "is_highly_connected": is_highly_connected,
                "is_isolated": is_isolated,
                "diversity_score": len(relation_types) / max(relation_count, 1),
            }

        except Exception:
            # Return default analysis if KB access fails
            return {
                "relation_count": 0,
                "relation_types": [],
                "is_highly_connected": False,
                "is_isolated": True,
                "diversity_score": 0.0,
            }


class BatchRetrievalChain(RetrievalChain):
    """Batch retrieval chain for processing multiple entities efficiently."""

    async def retrieve_knowledge_batch(
        self, plans: List[RetrievalPlan]
    ) -> List[RetrievalResult]:
        """Retrieve knowledge for multiple plans in batch."""
        try:
            # Group plans by similarity for efficient processing
            grouped_plans = await self._group_similar_plans(plans)

            results = []
            for group in grouped_plans:
                # Process each group
                group_results = await asyncio.gather(
                    *[self.retrieve_knowledge(plan) for plan in group],
                    return_exceptions=True,
                )

                # Handle exceptions
                for i, result in enumerate(group_results):
                    if isinstance(result, Exception):
                        # Create empty result for failed retrievals
                        empty_result = RetrievalResult(
                            entities=[],
                            triples=[],
                            metadata={"error": str(result), "plan_index": i},
                        )
                        results.append(empty_result)
                    else:
                        results.append(result)

            return results

        except Exception as e:
            raise RetrievalError(f"Batch retrieval failed: {str(e)}")

    async def _group_similar_plans(
        self, plans: List[RetrievalPlan]
    ) -> List[List[RetrievalPlan]]:
        """Group similar retrieval plans for efficient processing."""
        # Simple grouping by domain and max_hops
        groups = {}

        for plan in plans:
            key = (plan.domain, plan.max_hops)
            if key not in groups:
                groups[key] = []
            groups[key].append(plan)

        return list(groups.values())
