"""
Parallel Optimized LangGraph Agent for Knowledge-Enhanced Question Answering.

This agent implements a sophisticated multi-agent system with parallel processing
capabilities, designed to compete with and complement KERAG's entity-level retrieval.
"""

import asyncio
import time
from typing import Dict, Any, List, Optional, TypedDict, Union
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
import json

from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import BaseMessage

from core.types import KERAGConfig, KnowledgeBaseType
from core.exceptions import KERAGException
from knowledge_bases.sparql_kb import SPARQLKnowledgeBase
from knowledge_bases.api_kb import APIKnowledgeBase


@dataclass
class AgentResult:
    """Result from an individual agent."""

    agent_name: str
    result: str
    confidence: float
    reasoning_steps: List[str]
    metadata: Dict[str, Any]
    processing_time: float


@dataclass
class ParallelAgentResult:
    """Result from the parallel agent system."""

    final_answer: str
    confidence: float
    agent_results: List[AgentResult]
    reasoning_chain: List[str]
    metadata: Dict[str, Any]
    total_processing_time: float


class ParallelLangGraphAgentState(TypedDict):
    """State for the parallel LangGraph agent."""

    question: str
    domain: Optional[str]
    entities: List[Dict[str, Any]]
    agent_results: List[AgentResult]
    final_answer: Optional[str]
    confidence: Optional[float]
    reasoning_chain: List[str]
    metadata: Dict[str, Any]
    error: Optional[str]
    success: bool


class ParallelLangGraphAgent:
    """
    Parallel Optimized LangGraph Agent for complex question answering.

    This agent uses multiple specialized sub-agents working in parallel
    to provide comprehensive answers to complex questions.
    """

    def __init__(self, config: KERAGConfig):
        """
        Initialize the parallel LangGraph agent.

        Args:
            config: Configuration for the agent
        """
        self.config = config
        self.llm = self._create_llm()
        self.knowledge_base = self._create_knowledge_base()

        # Create specialized agents
        self.agents = {
            "entity_extractor": EntityExtractionAgent(self.llm),
            "relationship_analyzer": RelationshipAnalysisAgent(self.llm),
            "reasoning_agent": ReasoningAgent(self.llm),
            "fact_checker": FactCheckingAgent(self.llm),
            "synthesizer": AnswerSynthesisAgent(self.llm),
        }

        # Create the workflow graph
        self.graph = self._create_workflow_graph()

    def _create_llm(self):
        """Create the language model based on configuration."""
        if "gpt" in self.config.llm_model.lower():
            return ChatOpenAI(
                model=self.config.llm_model,
                api_key=self.config.api_key,
                temperature=0.1,
            )
        elif "claude" in self.config.llm_model.lower():
            return ChatAnthropic(
                model=self.config.llm_model,
                api_key=self.config.api_key,
                temperature=0.1,
            )
        elif "gemini" in self.config.llm_model.lower():
            return ChatGoogleGenerativeAI(
                model=self.config.llm_model,
                api_key=self.config.api_key,
                temperature=0.1,
            )
        else:
            raise KERAGException(f"Unsupported LLM model: {self.config.llm_model}")

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
        """Create the LangGraph workflow for parallel processing."""

        workflow = StateGraph(ParallelLangGraphAgentState)

        # Add nodes
        workflow.add_node("entity_extraction", self._entity_extraction_node)
        workflow.add_node("parallel_analysis", self._parallel_analysis_node)
        workflow.add_node("synthesis", self._synthesis_node)
        workflow.add_node("validation", self._validation_node)
        workflow.add_node("error_handling", self._error_handling_node)

        # Define edges
        workflow.add_edge("entity_extraction", "parallel_analysis")
        workflow.add_edge("parallel_analysis", "synthesis")
        workflow.add_edge("synthesis", "validation")
        workflow.add_edge("validation", END)

        # Add conditional edges for error handling
        workflow.add_conditional_edges(
            "entity_extraction",
            self._should_continue_after_extraction,
            {"continue": "parallel_analysis", "error": "error_handling"},
        )

        workflow.add_conditional_edges(
            "parallel_analysis",
            self._should_continue_after_analysis,
            {"continue": "synthesis", "error": "error_handling"},
        )

        workflow.add_conditional_edges(
            "synthesis",
            self._should_continue_after_synthesis,
            {"continue": "validation", "error": "error_handling"},
        )

        # Set entry point
        workflow.set_entry_point("entity_extraction")

        return workflow.compile()

    async def _entity_extraction_node(
        self, state: ParallelLangGraphAgentState
    ) -> ParallelLangGraphAgentState:
        """Extract entities and domain from the question."""
        try:
            start_time = time.time()

            # Use entity extraction agent
            result = await self.agents["entity_extractor"].process(
                question=state["question"]
            )

            processing_time = time.time() - start_time

            return {
                **state,
                "domain": result.get("domain"),
                "entities": result.get("entities", []),
                "metadata": {
                    **state.get("metadata", {}),
                    "entity_extraction_time": processing_time,
                    "extracted_entities": len(result.get("entities", [])),
                },
                "success": True,
            }

        except Exception as e:
            return {
                **state,
                "error": f"Entity extraction failed: {str(e)}",
                "success": False,
            }

    async def _parallel_analysis_node(
        self, state: ParallelLangGraphAgentState
    ) -> ParallelLangGraphAgentState:
        """Run parallel analysis using multiple agents."""
        try:
            start_time = time.time()

            # Prepare tasks for parallel execution
            tasks = []

            # Relationship analysis
            if state.get("entities"):
                tasks.append(
                    self._run_agent_async(
                        "relationship_analyzer",
                        question=state["question"],
                        entities=state["entities"],
                    )
                )

            # Reasoning analysis
            tasks.append(
                self._run_agent_async(
                    "reasoning_agent",
                    question=state["question"],
                    domain=state.get("domain"),
                )
            )

            # Fact checking
            tasks.append(
                self._run_agent_async("fact_checker", question=state["question"])
            )

            # Execute all tasks in parallel
            agent_results = await asyncio.gather(*tasks, return_exceptions=True)

            # Filter out exceptions and create AgentResult objects
            valid_results = []
            for i, result in enumerate(agent_results):
                if isinstance(result, Exception):
                    continue
                if result and result.get("success"):
                    agent_name = list(self.agents.keys())[i]
                    valid_results.append(
                        AgentResult(
                            agent_name=agent_name,
                            result=result.get("result", ""),
                            confidence=result.get("confidence", 0.0),
                            reasoning_steps=result.get("reasoning_steps", []),
                            metadata=result.get("metadata", {}),
                            processing_time=result.get("processing_time", 0.0),
                        )
                    )

            processing_time = time.time() - start_time

            return {
                **state,
                "agent_results": [asdict(r) for r in valid_results],
                "metadata": {
                    **state.get("metadata", {}),
                    "parallel_analysis_time": processing_time,
                    "successful_agents": len(valid_results),
                },
                "success": True,
            }

        except Exception as e:
            return {
                **state,
                "error": f"Parallel analysis failed: {str(e)}",
                "success": False,
            }

    async def _synthesis_node(
        self, state: ParallelLangGraphAgentState
    ) -> ParallelLangGraphAgentState:
        """Synthesize results from all agents."""
        try:
            start_time = time.time()

            # Convert agent results back to objects
            agent_results = []
            for result_dict in state.get("agent_results", []):
                agent_results.append(AgentResult(**result_dict))

            # Use synthesis agent
            synthesis_result = await self.agents["synthesizer"].process(
                question=state["question"], agent_results=agent_results
            )

            processing_time = time.time() - start_time

            return {
                **state,
                "final_answer": synthesis_result.get("answer", ""),
                "confidence": synthesis_result.get("confidence", 0.0),
                "reasoning_chain": synthesis_result.get("reasoning_chain", []),
                "metadata": {
                    **state.get("metadata", {}),
                    "synthesis_time": processing_time,
                },
                "success": True,
            }

        except Exception as e:
            return {**state, "error": f"Synthesis failed: {str(e)}", "success": False}

    async def _validation_node(
        self, state: ParallelLangGraphAgentState
    ) -> ParallelLangGraphAgentState:
        """Validate the final answer."""
        try:
            # Simple validation - in practice, this could be more sophisticated
            if not state.get("final_answer") or len(state.get("final_answer", "")) < 10:
                return {**state, "error": "Answer too short or empty", "success": False}

            return {**state, "success": True}

        except Exception as e:
            return {**state, "error": f"Validation failed: {str(e)}", "success": False}

    async def _error_handling_node(
        self, state: ParallelLangGraphAgentState
    ) -> ParallelLangGraphAgentState:
        """Handle errors gracefully."""
        error_msg = state.get("error", "Unknown error")

        return {
            **state,
            "final_answer": f"I encountered an error while processing your question: {error_msg}. Please try rephrasing your question.",
            "confidence": 0.0,
            "success": False,
        }

    async def _run_agent_async(self, agent_name: str, **kwargs) -> Dict[str, Any]:
        """Run an agent asynchronously."""
        try:
            start_time = time.time()
            result = await self.agents[agent_name].process(**kwargs)
            result["processing_time"] = time.time() - start_time
            return result
        except Exception as e:
            return {"success": False, "error": str(e), "processing_time": 0.0}

    def _should_continue_after_extraction(
        self, state: ParallelLangGraphAgentState
    ) -> str:
        """Determine next step after entity extraction."""
        return "continue" if state.get("success", False) else "error"

    def _should_continue_after_analysis(
        self, state: ParallelLangGraphAgentState
    ) -> str:
        """Determine next step after parallel analysis."""
        return "continue" if state.get("success", False) else "error"

    def _should_continue_after_synthesis(
        self, state: ParallelLangGraphAgentState
    ) -> str:
        """Determine next step after synthesis."""
        return "continue" if state.get("success", False) else "error"

    async def process_question(self, question: str) -> ParallelAgentResult:
        """Process a question through the parallel agent system."""
        try:
            # Initialize state
            initial_state: ParallelLangGraphAgentState = {
                "question": question,
                "domain": None,
                "entities": [],
                "agent_results": [],
                "final_answer": None,
                "confidence": None,
                "reasoning_chain": [],
                "metadata": {},
                "error": None,
                "success": False,
            }

            # Run workflow
            final_state = await self.graph.ainvoke(initial_state)

            # Create result
            if final_state.get("success", False):
                return ParallelAgentResult(
                    final_answer=final_state["final_answer"],
                    confidence=final_state["confidence"],
                    agent_results=[
                        AgentResult(**r) for r in final_state.get("agent_results", [])
                    ],
                    reasoning_chain=final_state.get("reasoning_chain", []),
                    metadata=final_state.get("metadata", {}),
                    total_processing_time=sum(
                        r.get("processing_time", 0)
                        for r in final_state.get("agent_results", [])
                    )
                    + final_state.get("metadata", {}).get("synthesis_time", 0),
                )
            else:
                raise KERAGException(
                    final_state.get("error", "Agent processing failed")
                )

        except Exception as e:
            raise KERAGException(f"Failed to process question: {str(e)}")

    async def __aenter__(self):
        """Async context manager entry."""
        if hasattr(self.knowledge_base, "__aenter__"):
            await self.knowledge_base.__aenter__()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if hasattr(self.knowledge_base, "__aexit__"):
            await self.knowledge_base.__aexit__(exc_type, exc_val, exc_tb)


class BaseAgent:
    """Base class for specialized agents."""

    def __init__(self, llm):
        self.llm = llm

    async def process(self, **kwargs) -> Dict[str, Any]:
        """Process input and return result."""
        raise NotImplementedError


class EntityExtractionAgent(BaseAgent):
    """Agent for extracting entities and determining domain."""

    async def process(self, question: str) -> Dict[str, Any]:
        """Extract entities and domain from question."""
        try:
            prompt = f"""
            Analyze the following question and extract:
            1. Key entities mentioned
            2. Domain/category of the question
            3. Type of reasoning required
            
            Question: {question}
            
            Return your analysis in JSON format with keys: entities, domain, reasoning_type
            """

            response = await self.llm.ainvoke([HumanMessage(content=prompt)])
            result = json.loads(response.content)

            return {
                "success": True,
                "entities": result.get("entities", []),
                "domain": result.get("domain"),
                "reasoning_type": result.get("reasoning_type"),
                "metadata": {"extraction_method": "llm_analysis"},
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "entities": [],
                "domain": "general",
            }


class RelationshipAnalysisAgent(BaseAgent):
    """Agent for analyzing entity relationships."""

    async def process(
        self, question: str, entities: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Analyze relationships between entities."""
        try:
            prompt = f"""
            Analyze the relationships between entities in this question:
            Question: {question}
            Entities: {entities}
            
            Identify:
            1. Direct relationships between entities
            2. Indirect relationships (through 2-3 hops)
            3. Missing connections that might be relevant
            
            Return your analysis in JSON format.
            """

            response = await self.llm.ainvoke([HumanMessage(content=prompt)])
            result = json.loads(response.content)

            return {
                "success": True,
                "result": result.get("analysis", ""),
                "confidence": result.get("confidence", 0.8),
                "reasoning_steps": result.get("reasoning_steps", []),
                "metadata": {"analysis_type": "relationship"},
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "result": "Unable to analyze relationships",
                "confidence": 0.0,
            }


class ReasoningAgent(BaseAgent):
    """Agent for complex reasoning."""

    async def process(
        self, question: str, domain: Optional[str] = None
    ) -> Dict[str, Any]:
        """Perform complex reasoning on the question."""
        try:
            prompt = f"""
            Perform step-by-step reasoning for this question:
            Question: {question}
            Domain: {domain or 'general'}
            
            Provide:
            1. Step-by-step reasoning process
            2. Key insights and connections
            3. Potential answer approach
            4. Confidence level
            
            Return your analysis in JSON format.
            """

            response = await self.llm.ainvoke([HumanMessage(content=prompt)])
            result = json.loads(response.content)

            return {
                "success": True,
                "result": result.get("reasoning", ""),
                "confidence": result.get("confidence", 0.7),
                "reasoning_steps": result.get("steps", []),
                "metadata": {"reasoning_type": "step_by_step"},
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "result": "Unable to perform reasoning",
                "confidence": 0.0,
            }


class FactCheckingAgent(BaseAgent):
    """Agent for fact checking and validation."""

    async def process(self, question: str) -> Dict[str, Any]:
        """Check facts and validate information."""
        try:
            prompt = f"""
            Fact-check and validate information for this question:
            Question: {question}
            
            Provide:
            1. Fact-checking results
            2. Potential sources of information
            3. Confidence in available facts
            4. Areas that need verification
            
            Return your analysis in JSON format.
            """

            response = await self.llm.ainvoke([HumanMessage(content=prompt)])
            result = json.loads(response.content)

            return {
                "success": True,
                "result": result.get("fact_check", ""),
                "confidence": result.get("confidence", 0.6),
                "reasoning_steps": result.get("verification_steps", []),
                "metadata": {"fact_check_type": "comprehensive"},
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "result": "Unable to fact-check",
                "confidence": 0.0,
            }


class AnswerSynthesisAgent(BaseAgent):
    """Agent for synthesizing final answers."""

    async def process(
        self, question: str, agent_results: List[AgentResult]
    ) -> Dict[str, Any]:
        """Synthesize results from all agents into a final answer."""
        try:
            # Prepare context from all agent results
            context = []
            for result in agent_results:
                context.append(f"{result.agent_name}: {result.result}")

            prompt = f"""
            Synthesize the following agent results into a comprehensive answer:
            Question: {question}
            
            Agent Results:
            {chr(10).join(context)}
            
            Provide:
            1. A comprehensive final answer
            2. Confidence level
            3. Reasoning chain
            4. Key insights from each agent
            
            Return your synthesis in JSON format.
            """

            response = await self.llm.ainvoke([HumanMessage(content=prompt)])
            result = json.loads(response.content)

            return {
                "success": True,
                "answer": result.get("final_answer", ""),
                "confidence": result.get("confidence", 0.7),
                "reasoning_chain": result.get("reasoning_chain", []),
                "metadata": {"synthesis_method": "multi_agent"},
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "answer": "Unable to synthesize answer",
                "confidence": 0.0,
            }


# Helper function for dataclass serialization
def asdict(obj):
    """Convert dataclass to dictionary."""
    if hasattr(obj, "__dict__"):
        return {k: v for k, v in obj.__dict__.items()}
    return obj
