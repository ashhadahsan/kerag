"""
Summarization chain implementation for KERAG using LangChain LCEL.
"""

import asyncio
from typing import Dict, Any, List, Optional
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI

from core.interfaces import SummarizationInterface, ChainInterface
from core.types import RetrievalResult, SummarizationConfig
from core.exceptions import SummarizationError
from config.settings import PromptConfig


class SummarizationChain(SummarizationInterface, ChainInterface):
    """Summarization chain for Chain-of-Thought reasoning and answer generation."""

    def __init__(
        self,
        llm_model: str = "gpt-4",
        api_key: Optional[str] = None,
        config: Optional[SummarizationConfig] = None,
    ):
        """
        Initialize summarization chain.

        Args:
            llm_model: LLM model to use
            api_key: API key for LLM
            config: Summarization configuration
        """
        self.llm_model = llm_model
        self.config = config or SummarizationConfig()

        # Initialize LLM
        if llm_model.startswith("gpt"):
            self.llm = ChatOpenAI(
                model=llm_model,
                temperature=self.config.temperature,
                api_key=api_key,
                max_tokens=self.config.max_tokens,
            )
        elif llm_model.startswith("claude"):
            self.llm = ChatAnthropic(
                model=llm_model,
                temperature=self.config.temperature,
                api_key=api_key,
                max_tokens=self.config.max_tokens,
            )
        elif llm_model.startswith("gemini"):
            self.llm = ChatGoogleGenerativeAI(
                model=llm_model,
                temperature=self.config.temperature,
                google_api_key=api_key,
                max_tokens=self.config.max_tokens,
            )
        elif llm_model.startswith("grok"):
            # For Grok, we'll use OpenAI-compatible interface since xAI supports it
            self.llm = ChatOpenAI(
                model=llm_model,
                temperature=self.config.temperature,
                api_key=api_key,
                max_tokens=self.config.max_tokens,
                base_url="https://api.x.ai/v1",  # xAI's OpenAI-compatible endpoint
            )
        else:
            self.llm = ChatOpenAI(
                model=llm_model,
                temperature=self.config.temperature,
                api_key=api_key,
                max_tokens=self.config.max_tokens,
            )

        # Setup summarization chain
        self._setup_chain()

    def _setup_chain(self):
        """Setup LangChain LCEL summarization chain."""

        # Chain-of-Thought reasoning prompt
        cot_prompt = ChatPromptTemplate.from_template(
            PromptConfig.SUMMARIZATION_PROMPTS["cot_reasoning"]
        )

        self.cot_chain = cot_prompt | self.llm | StrOutputParser()

        # Confidence scoring prompt
        confidence_prompt = ChatPromptTemplate.from_template(
            PromptConfig.SUMMARIZATION_PROMPTS["confidence_scoring"]
        )

        self.confidence_chain = confidence_prompt | self.llm | StrOutputParser()

        # Direct answer chain
        direct_prompt = ChatPromptTemplate.from_template(
            "Answer the following question based on the provided knowledge.\n\n"
            "Question: {question}\n\n"
            "Knowledge:\n{knowledge}\n\n"
            "Answer:"
        )
        self.direct_chain = direct_prompt | self.llm | StrOutputParser()

        # Main summarization chain
        self.summarization_chain = RunnablePassthrough.assign(
            answer=RunnableLambda(self._generate_answer),
            confidence=RunnableLambda(self._compute_confidence),
        )

    async def generate_answer(
        self, question: str, filtered_knowledge: RetrievalResult
    ) -> str:
        """Generate final answer using Chain-of-Thought reasoning."""
        try:
            # Prepare knowledge for summarization
            knowledge_text = self._prepare_knowledge_for_summarization(
                filtered_knowledge
            )

            # Generate answer using CoT reasoning
            if self.config.use_cot:
                response = await self.cot_chain.ainvoke(
                    {"question": question, "knowledge": knowledge_text}
                )

                # Extract answer from CoT response
                answer = self._extract_answer_from_cot(response)
            else:
                # Direct answer generation without CoT
                answer = await self._generate_direct_answer(question, knowledge_text)

            return answer

        except Exception as e:
            raise SummarizationError(f"Failed to generate answer: {str(e)}")

    async def compute_confidence(
        self, question: str, answer: str, knowledge: RetrievalResult
    ) -> float:
        """Compute confidence score for the generated answer."""
        try:
            # Prepare knowledge for confidence assessment
            knowledge_text = self._prepare_knowledge_for_summarization(knowledge)

            # Get confidence assessment from LLM
            response = await self.confidence_chain.ainvoke(
                {"question": question, "answer": answer, "knowledge": knowledge_text}
            )

            # Extract confidence score
            confidence = self._extract_confidence_score(response)

            return confidence

        except Exception as e:
            # Return default confidence if assessment fails
            return 0.5

    def _prepare_knowledge_for_summarization(self, knowledge: RetrievalResult) -> str:
        """Prepare knowledge for LLM summarization."""
        text_parts = []

        # Add context information
        text_parts.append("CONTEXT INFORMATION:")
        text_parts.append(f"Total entities: {len(knowledge.entities)}")
        text_parts.append(f"Total triples: {len(knowledge.triples)}")

        # Add entities with their attributes
        text_parts.append("\nENTITIES:")
        for entity in knowledge.entities:
            text_parts.append(f"- {entity.name} ({entity.type})")
            for attr_key, attr_value in entity.attributes.items():
                if isinstance(attr_value, str) and len(attr_value) < 200:
                    text_parts.append(f"  {attr_key}: {attr_value}")

        # Add triples
        text_parts.append("\nRELATIONSHIPS:")
        for triple in knowledge.triples:
            subject_name = self._get_entity_name(triple.subject, knowledge.entities)
            object_name = self._get_entity_name(triple.object, knowledge.entities)
            predicate_name = self._simplify_predicate(triple.predicate)

            text_parts.append(f"- {subject_name} {predicate_name} {object_name}")
            if triple.confidence:
                text_parts.append(f"  (confidence: {triple.confidence})")

        # Limit text length to fit in context window
        full_text = "\n".join(text_parts)
        if (
            len(full_text) > self.config.max_tokens * 4
        ):  # Rough character to token ratio
            # Truncate while preserving structure
            truncated_parts = []
            current_length = 0
            max_length = self.config.max_tokens * 3

            for part in text_parts:
                if current_length + len(part) > max_length:
                    truncated_parts.append("... (truncated)")
                    break
                truncated_parts.append(part)
                current_length += len(part)

            full_text = "\n".join(truncated_parts)

        return full_text

    def _get_entity_name(self, entity_id: str, entities: List[Any]) -> str:
        """Get entity name from ID."""
        for entity in entities:
            if entity.id == entity_id:
                return entity.name
        return entity_id

    def _simplify_predicate(self, predicate: str) -> str:
        """Simplify predicate URI to readable text."""
        if "/" in predicate:
            return predicate.split("/")[-1].replace("_", " ")
        return predicate

    def _extract_answer_from_cot(self, cot_response: str) -> str:
        """Extract final answer from Chain-of-Thought response."""
        lines = cot_response.split("\n")

        # Look for "Answer:" marker
        for i, line in enumerate(lines):
            if line.strip().startswith("Answer:"):
                # Extract answer from this line or following lines
                answer = line.replace("Answer:", "").strip()

                # If answer is very short, check next lines
                if len(answer) < 10 and i + 1 < len(lines):
                    next_line = lines[i + 1].strip()
                    if next_line and not next_line.startswith(("Reasoning:", "Step")):
                        answer += " " + next_line

                return answer

        # Fallback: return the last non-empty line
        for line in reversed(lines):
            if line.strip() and not line.strip().startswith(("Reasoning:", "Step")):
                return line.strip()

        # Final fallback: return the whole response
        return cot_response.strip()

    async def _generate_direct_answer(self, question: str, knowledge_text: str) -> str:
        """Generate answer directly without Chain-of-Thought."""
        direct_prompt = ChatPromptTemplate.from_template(
            "Answer the following question based on the provided knowledge.\n\n"
            "Question: {question}\n\n"
            "Knowledge:\n{knowledge}\n\n"
            "Answer:"
        )

        direct_chain = direct_prompt | self.llm | StrOutputParser()

        return await direct_chain.ainvoke(
            {"question": question, "knowledge": knowledge_text}
        )

    def _extract_confidence_score(self, confidence_response: str) -> float:
        """Extract confidence score from LLM response."""
        try:
            # Look for numeric confidence scores
            import re

            # Pattern to find confidence scores
            patterns = [
                r"confidence[:\s]+(\d+\.?\d*)",
                r"score[:\s]+(\d+\.?\d*)",
                r"(\d+\.?\d*)\s*(?:out of 1|/1|%)",
                r"(\d+\.?\d*)",
            ]

            for pattern in patterns:
                matches = re.findall(pattern, confidence_response.lower())
                if matches:
                    score = float(matches[0])

                    # Normalize to 0-1 range
                    if score > 1 and score <= 100:
                        score = score / 100
                    elif score > 100:
                        score = min(score / 100, 1.0)

                    return min(max(score, 0.0), 1.0)

            # Fallback: analyze sentiment of response
            positive_words = ["high", "confident", "certain", "strong", "reliable"]
            negative_words = ["low", "uncertain", "doubtful", "weak", "unreliable"]

            response_lower = confidence_response.lower()
            positive_count = sum(1 for word in positive_words if word in response_lower)
            negative_count = sum(1 for word in negative_words if word in response_lower)

            if positive_count > negative_count:
                return 0.8
            elif negative_count > positive_count:
                return 0.3
            else:
                return 0.5

        except Exception:
            return 0.5

    async def arun(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Asynchronously run the summarization chain."""
        try:
            question = input_data["question"]
            filtered_knowledge = input_data["filtered_knowledge"]

            # Generate answer
            answer = await self.generate_answer(question, filtered_knowledge)

            # Compute confidence
            confidence = await self.compute_confidence(
                question, answer, filtered_knowledge
            )

            return {
                **input_data,
                "answer": answer,
                "confidence": confidence,
                "success": True,
            }

        except Exception as e:
            return {**input_data, "error": str(e), "success": False}

    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Synchronously run the summarization chain."""
        return asyncio.run(self.arun(input_data))

    def get_chain(self):
        """Get the underlying LangChain chain."""
        return self.summarization_chain

    async def _generate_answer(self, input_data: Dict[str, Any]) -> str:
        """Internal method for chain execution."""
        try:
            question = input_data.get("question")
            filtered_knowledge = input_data.get("filtered_knowledge")

            if not question or not filtered_knowledge:
                raise SummarizationError("Missing question or filtered knowledge")

            return await self.generate_answer(question, filtered_knowledge)
        except Exception as e:
            raise SummarizationError(f"Chain answer generation failed: {str(e)}")

    async def _compute_confidence(self, input_data: Dict[str, Any]) -> float:
        """Internal method for chain execution."""
        try:
            question = input_data.get("question")
            answer = input_data.get("answer")
            filtered_knowledge = input_data.get("filtered_knowledge")

            if not all([question, answer, filtered_knowledge]):
                return 0.5

            return await self.compute_confidence(question, answer, filtered_knowledge)
        except Exception as e:
            return 0.5


class AdvancedSummarizationChain(SummarizationChain):
    """Advanced summarization chain with multiple reasoning strategies."""

    async def generate_answer(
        self, question: str, filtered_knowledge: RetrievalResult
    ) -> str:
        """Generate answer using multiple reasoning strategies."""
        try:
            # Strategy 1: Chain-of-Thought reasoning
            cot_answer = await super().generate_answer(question, filtered_knowledge)

            # Strategy 2: Direct extraction
            extraction_answer = await self._extraction_based_answer(
                question, filtered_knowledge
            )

            # Strategy 3: Template-based answer
            template_answer = await self._template_based_answer(
                question, filtered_knowledge
            )

            # Combine answers using ensemble approach
            final_answer = await self._combine_answers(
                question, [cot_answer, extraction_answer, template_answer]
            )

            return final_answer

        except Exception as e:
            raise SummarizationError(f"Advanced answer generation failed: {str(e)}")

    async def _extraction_based_answer(
        self, question: str, knowledge: RetrievalResult
    ) -> str:
        """Generate answer by direct extraction from knowledge."""
        # Simple extraction based on question type
        question_lower = question.lower()

        if "what" in question_lower:
            # Extract relevant facts
            facts = []
            for entity in knowledge.entities:
                if entity.attributes:
                    facts.append(f"{entity.name}: {entity.attributes}")
            return ". ".join(facts[:3])  # Limit to 3 facts

        elif "who" in question_lower:
            # Extract person entities
            persons = [
                entity.name
                for entity in knowledge.entities
                if "person" in entity.type.lower() or "human" in entity.type.lower()
            ]
            return ", ".join(persons[:3])

        elif "when" in question_lower:
            # Extract date/time information
            dates = []
            for entity in knowledge.entities:
                for attr_key, attr_value in entity.attributes.items():
                    if any(
                        time_word in attr_key.lower()
                        for time_word in ["date", "time", "year"]
                    ):
                        dates.append(str(attr_value))
            return ", ".join(dates[:3])

        else:
            return "Information extracted from knowledge base."

    async def _template_based_answer(
        self, question: str, knowledge: RetrievalResult
    ) -> str:
        """Generate answer using predefined templates."""
        # Simple template matching
        question_lower = question.lower()

        if "how many" in question_lower:
            count = len(knowledge.entities)
            return f"There are {count} relevant items found."

        elif "list" in question_lower or "what are" in question_lower:
            items = [entity.name for entity in knowledge.entities[:5]]
            return "The relevant items are: " + ", ".join(items)

        else:
            return f"Based on the available information, there are {len(knowledge.entities)} relevant entities and {len(knowledge.triples)} relationships."

    async def _combine_answers(self, question: str, answers: List[str]) -> str:
        """Combine multiple answers using LLM."""
        try:
            combine_prompt = ChatPromptTemplate.from_template(
                "Given the following question and multiple answer candidates, "
                "provide the best single answer by combining the most relevant information.\n\n"
                "Question: {question}\n\n"
                "Answer candidates:\n{candidates}\n\n"
                "Best answer:"
            )

            candidates_text = "\n".join(
                [f"{i+1}. {answer}" for i, answer in enumerate(answers)]
            )

            combine_chain = combine_prompt | self.llm | StrOutputParser()

            return await combine_chain.ainvoke(
                {"question": question, "candidates": candidates_text}
            )

        except Exception:
            # Fallback to first answer if combination fails
            return answers[0] if answers else "No answer available."

    async def _hybrid_answer(self, question: str, knowledge: RetrievalResult) -> str:
        """Generate answer using hybrid approach combining extraction and template methods."""
        try:
            # Get extraction-based answer
            extraction_answer = await self._extraction_based_answer(question, knowledge)

            # Get template-based answer
            template_answer = await self._template_based_answer(question, knowledge)

            # Combine using simple logic
            if len(extraction_answer) > len(template_answer):
                return extraction_answer
            else:
                return template_answer

        except Exception:
            return "Hybrid answer generation failed."
