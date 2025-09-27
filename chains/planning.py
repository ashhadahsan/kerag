"""
Planning chain implementation for KERAG using LangChain LCEL.
"""

import json
import asyncio
from typing import Dict, Any, List, Optional
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_openai import ChatOpenAI
from langchain_community.llms import OpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI

from core.interfaces import PlanningInterface, ChainInterface
from core.types import Entity, DomainType, RetrievalPlan, PlanningResult
from core.exceptions import PlanningError
from config.settings import PromptConfig, LLMConfig


class PlanningChain(PlanningInterface, ChainInterface):
    """Planning chain for entity identification, domain detection, and retrieval planning."""

    def __init__(
        self,
        llm_model: str = "gpt-4",
        api_key: Optional[str] = None,
        temperature: float = 0.1,
    ):
        """
        Initialize planning chain.

        Args:
            llm_model: LLM model to use
            api_key: API key for LLM
            temperature: LLM temperature
        """
        self.llm_model = llm_model
        self.temperature = temperature

        # Initialize LLM
        if llm_model.startswith("gpt"):
            self.llm = ChatOpenAI(
                model=llm_model, temperature=temperature, api_key=api_key
            )
        elif llm_model.startswith("claude"):
            self.llm = ChatAnthropic(
                model=llm_model, temperature=temperature, api_key=api_key
            )
        elif llm_model.startswith("gemini"):
            self.llm = ChatGoogleGenerativeAI(
                model=llm_model, temperature=temperature, google_api_key=api_key
            )
        elif llm_model.startswith("grok"):
            # For Grok, we'll use OpenAI-compatible interface since xAI supports it
            self.llm = ChatOpenAI(
                model=llm_model,
                temperature=temperature,
                api_key=api_key,
                base_url="https://api.x.ai/v1",  # xAI's OpenAI-compatible endpoint
            )
        else:
            self.llm = OpenAI(model=llm_model, temperature=temperature, api_key=api_key)

        # Initialize chains
        self._setup_chains()

    def _setup_chains(self):
        """Setup LangChain LCEL chains."""

        # Entity Identification Chain
        entity_prompt = ChatPromptTemplate.from_template(
            PromptConfig.PLANNING_PROMPTS["entity_identification"]
        )
        self.entity_chain = entity_prompt | self.llm | JsonOutputParser()

        # Domain Detection Chain
        domain_prompt = ChatPromptTemplate.from_template(
            PromptConfig.PLANNING_PROMPTS["domain_detection"]
        )
        self.domain_chain = domain_prompt | self.llm | JsonOutputParser()

        # Retrieval Planning Chain
        retrieval_prompt = ChatPromptTemplate.from_template(
            PromptConfig.PLANNING_PROMPTS["retrieval_planning"]
        )
        self.retrieval_planning_chain = retrieval_prompt | self.llm | JsonOutputParser()

        # Combined planning chain
        self.planning_chain = RunnablePassthrough.assign(
            entity_info=RunnableLambda(self._extract_entity_info),
            domain_info=RunnableLambda(self._extract_domain_info),
            retrieval_plan_info=RunnableLambda(self._extract_retrieval_plan_info),
        )

    async def identify_topic_entity(self, question: str) -> Entity:
        """Identify the main topic entity from the question."""
        try:
            result = await self.entity_chain.ainvoke({"question": question})

            # Validate and create entity
            if not result.get("entity_name"):
                raise PlanningError("Failed to identify entity name")

            confidence = result.get("confidence", 0.5)

            # Check confidence threshold
            if confidence < 0.3:
                raise PlanningError("Low confidence in entity identification")

            entity = Entity(
                id=f"entity_{result['entity_name'].replace(' ', '_')}",
                name=result["entity_name"],
                type=result.get("entity_type", "Unknown"),
                attributes={
                    "confidence": confidence,
                    "detection_method": "llm_identification",
                },
                neighbors=[],
            )

            return entity

        except Exception as e:
            raise PlanningError(f"Failed to identify topic entity: {str(e)}")

    async def detect_domain(self, question: str, topic_entity: Entity) -> str:
        """Detect the domain of the question."""
        try:
            result = await self.domain_chain.ainvoke(
                {
                    "question": question,
                    "topic_entity": f"{topic_entity.name} ({topic_entity.type})",
                }
            )

            domain = result.get("domain", "general").lower()
            confidence = result.get("confidence", 0.5)

            # Check confidence threshold
            if confidence < 0.3:
                raise PlanningError("Low confidence in domain detection")

            # Validate domain
            try:
                domain_enum = DomainType(domain)
                return domain_enum.value
            except ValueError:
                # Map common variations to valid domains
                domain_mapping = {
                    "film": "movie",
                    "cinema": "movie",
                    "movie": "movie",
                    "music": "music",
                    "song": "music",
                    "album": "music",
                    "artist": "music",
                    "sport": "sports",
                    "sports": "sports",
                    "financial": "finance",
                    "finance": "finance",
                    "book": "book",
                    "novel": "book",
                    "author": "book",
                    "person": "person",
                    "people": "person",
                    "organization": "organization",
                    "company": "organization",
                    "location": "location",
                    "place": "location",
                    "mixed": "general",  # Handle mixed domains
                    "unknown": "general",  # Handle unknown domains
                }

                mapped_domain = domain_mapping.get(domain, "general")
                return mapped_domain

        except Exception as e:
            raise PlanningError(f"Failed to detect domain: {str(e)}")

    async def generate_retrieval_plan(
        self, question: str, topic_entity: Entity, domain: str
    ) -> RetrievalPlan:
        """Generate a retrieval plan based on question analysis."""
        try:
            # Validate domain before proceeding
            try:
                DomainType(domain)
            except ValueError:
                raise PlanningError(f"Invalid domain: {domain}")

            result = await self.retrieval_planning_chain.ainvoke(
                {
                    "question": question,
                    "topic_entity": f"{topic_entity.name} ({topic_entity.type})",
                    "domain": domain,
                }
            )

            # Extract plan components
            relations = result.get("relations", [])
            max_hops = result.get("max_hops", 2)
            filters = result.get("filters", {})

            # Validate and create retrieval plan
            plan = RetrievalPlan(
                domain=DomainType(domain),
                topic_entity=topic_entity,
                relations=relations,
                max_hops=max_hops,
                filters=filters,
            )

            return plan

        except Exception as e:
            raise PlanningError(f"Failed to generate retrieval plan: {str(e)}")

    async def arun(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Asynchronously run the planning chain."""
        try:
            question = input_data["question"]

            # Step 1: Identify topic entity
            topic_entity = await self.identify_topic_entity(question)

            # Step 2: Detect domain
            domain = await self.detect_domain(question, topic_entity)

            # Step 3: Generate retrieval plan
            retrieval_plan = await self.generate_retrieval_plan(
                question, topic_entity, domain
            )

            # Create planning result
            planning_result = PlanningResult(
                domain=DomainType(domain),
                topic_entity=topic_entity,
                relations=retrieval_plan.relations,
                max_hops=retrieval_plan.max_hops,
                confidence=min(
                    topic_entity.attributes.get("confidence", 0.5),
                    retrieval_plan.filters.get("confidence", 0.5),
                ),
            )

            return {
                "question": question,
                "topic_entity": topic_entity,
                "domain": domain,
                "retrieval_plan": retrieval_plan,
                "planning_result": planning_result,
                "success": True,
            }

        except Exception as e:
            return {
                "question": input_data.get("question", ""),
                "error": str(e),
                "success": False,
            }

    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Synchronously run the planning chain."""
        return asyncio.run(self.arun(input_data))

    def get_chain(self):
        """Get the underlying LangChain chain."""
        return self.planning_chain

    async def _extract_entity_info(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract entity information using the entity chain."""
        try:
            result = await self.entity_chain.ainvoke(
                {"question": input_data["question"]}
            )
            return {"entity_info": result}
        except Exception as e:
            return {"entity_info": {"error": str(e)}}

    async def _extract_domain_info(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract domain information using the domain chain."""
        try:
            entity_info = input_data.get("entity_info", {})
            if "error" in entity_info:
                return {"domain_info": {"error": "Entity identification failed"}}

            entity_name = entity_info.get("entity_name", "")
            entity_type = entity_info.get("entity_type", "")

            result = await self.domain_chain.ainvoke(
                {
                    "question": input_data["question"],
                    "topic_entity": f"{entity_name} ({entity_type})",
                }
            )
            return {"domain_info": result}
        except Exception as e:
            return {"domain_info": {"error": str(e)}}

    async def _extract_retrieval_plan_info(
        self, input_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Extract retrieval plan information using the retrieval planning chain."""
        try:
            entity_info = input_data.get("entity_info", {})
            domain_info = input_data.get("domain_info", {})

            if "error" in entity_info or "error" in domain_info:
                return {"retrieval_plan_info": {"error": "Previous steps failed"}}

            entity_name = entity_info.get("entity_name", "")
            entity_type = entity_info.get("entity_type", "")
            domain = domain_info.get("domain", "general")

            result = await self.retrieval_planning_chain.ainvoke(
                {
                    "question": input_data["question"],
                    "topic_entity": f"{entity_name} ({entity_type})",
                    "domain": domain,
                }
            )
            return {"retrieval_plan_info": result}
        except Exception as e:
            return {"retrieval_plan_info": {"error": str(e)}}


class AdvancedPlanningChain(PlanningChain):
    """Advanced planning chain with enhanced entity resolution and domain detection."""

    def __init__(
        self,
        llm_model: str = "gpt-4",
        api_key: Optional[str] = None,
        temperature: float = 0.1,
        knowledge_base: Optional[Any] = None,
    ):
        """
        Initialize advanced planning chain.

        Args:
            llm_model: LLM model to use
            api_key: API key for LLM
            temperature: LLM temperature
            knowledge_base: Knowledge base for entity resolution
        """
        super().__init__(llm_model, api_key, temperature)
        self.knowledge_base = knowledge_base

    async def identify_topic_entity_with_resolution(self, question: str) -> Entity:
        """Identify topic entity with knowledge base resolution."""
        try:
            # First, get LLM-based entity identification
            llm_entity = await self.identify_topic_entity(question)

            if self.knowledge_base:
                # Try to resolve the entity in the knowledge base
                try:
                    search_results = await self.knowledge_base.search_entities(
                        llm_entity.name, limit=5
                    )

                    if search_results:
                        # Use the best match from knowledge base
                        resolved_entity = search_results[0]
                        resolved_entity.attributes.update(llm_entity.attributes)
                        resolved_entity.attributes["resolution_method"] = (
                            "kb_resolution"
                        )
                        return resolved_entity

                except Exception:
                    # Fall back to LLM entity if KB resolution fails
                    pass

            # Return LLM entity with resolution method
            llm_entity.attributes["resolution_method"] = "llm_only"
            return llm_entity

        except Exception as e:
            raise PlanningError(
                f"Failed to identify topic entity with resolution: {str(e)}"
            )

    async def detect_domain_with_context(
        self, question: str, topic_entity: Entity
    ) -> str:
        """Detect domain with additional context from knowledge base."""
        try:
            # Get base domain detection
            domain = await self.detect_domain(question, topic_entity)

            # Enhance with knowledge base context if available
            if self.knowledge_base and hasattr(topic_entity, "id"):
                try:
                    entity_relations = await self.knowledge_base.get_relations(
                        topic_entity.id
                    )

                    # Use relations to refine domain detection
                    domain_hints = self._analyze_relations_for_domain(entity_relations)
                    if domain_hints and domain_hints != domain:
                        # Update domain based on relation analysis
                        domain = domain_hints

                except Exception:
                    # Continue with original domain if KB analysis fails
                    pass

            return domain

        except Exception as e:
            raise PlanningError(f"Failed to detect domain with context: {str(e)}")

    def _analyze_relations_for_domain(self, relations: List[str]) -> Optional[str]:
        """Analyze entity relations to determine domain hints."""
        domain_relation_mapping = {
            "movie": ["actor", "director", "genre", "release_date", "box_office"],
            "music": ["artist", "album", "genre", "release_date", "label"],
            "sports": ["player", "team", "league", "position", "statistics"],
            "finance": ["company", "stock", "revenue", "market_cap", "sector"],
            "book": ["author", "publisher", "genre", "publication_date", "series"],
        }

        relation_scores = {}
        for domain, domain_relations in domain_relation_mapping.items():
            score = sum(
                1
                for rel in relations
                if any(dr in rel.lower() for dr in domain_relations)
            )
            relation_scores[domain] = score

        # Return domain with highest score if significant
        if relation_scores:
            best_domain = max(relation_scores, key=relation_scores.get)
            if relation_scores[best_domain] > 0:
                return best_domain

        return None
