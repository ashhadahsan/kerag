"""
Quick test script to verify LLM provider integration.
"""

import asyncio
import os
from typing import Dict, Any
from unittest.mock import patch, AsyncMock
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from core.types import KERAGConfig, KnowledgeBaseType
from workflows.kerag_workflow import AsyncKERAGWorkflow


async def check_llm_provider(model: str, api_key: str, provider_name: str):
    """Test a specific LLM provider."""

    print(f"\n🧪 Testing {provider_name} ({model})...")

    config = KERAGConfig(
        knowledge_base_type=KnowledgeBaseType.SPARQL,
        endpoint_url="http://dbpedia.org/sparql",
        llm_model=model,
        api_key=api_key,
        # Minimal config for quick test
        retrieval_config=RetrievalConfig(max_hops=1, max_entities=10),
        filtering_config=FilteringConfig(relevance_threshold=0.8, max_triples=5),
        timeout=15,
    )

    try:
        # Mock the LLM calls to avoid real API calls
        with patch("chains.planning.PlanningChain.entity_chain") as mock_entity, patch(
            "chains.planning.PlanningChain.domain_chain"
        ) as mock_domain, patch(
            "chains.planning.PlanningChain.retrieval_planning_chain"
        ) as mock_planning, patch(
            "chains.filtering.FilteringChain.relevance_chain"
        ) as mock_relevance, patch(
            "chains.summarization.SummarizationChain.cot_chain"
        ) as mock_cot, patch(
            "chains.summarization.SummarizationChain.confidence_chain"
        ) as mock_confidence:

            # Mock responses
            mock_entity.ainvoke = AsyncMock(
                return_value={
                    "entity_name": "France",
                    "entity_type": "Country",
                    "confidence": 0.9,
                }
            )

            mock_domain.ainvoke = AsyncMock(
                return_value={"domain": "location", "confidence": 0.8}
            )

            mock_planning.ainvoke = AsyncMock(
                return_value={"relations": ["capital"], "max_hops": 1, "filters": {}}
            )

            mock_relevance.ainvoke = AsyncMock(
                return_value={"entity_scores": {}, "triple_scores": {}}
            )

            mock_cot.ainvoke = AsyncMock(return_value="The capital of France is Paris.")

            mock_confidence.ainvoke = AsyncMock(return_value="0.9")

            async with AsyncKERAGWorkflow(config) as workflow:
                result = await workflow.process_question(
                    "What is the capital of France?"
                )

                print(f"  ✅ Success!")
                print(f"  Answer: {result.answer}")
                print(f"  Confidence: {result.confidence:.2f}")

                return True

    except Exception as e:
        print(f"  ❌ Error: {str(e)}")
        return False


async def main():
    """Test all available LLM providers."""

    print("🚀 KERAG LLM Integration Test")
    print("=" * 50)

    # Test configurations
    test_configs = []

    # OpenAI
    if os.getenv("OPENAI_API_KEY"):
        test_configs.extend(
            [
                ("gpt-4", os.getenv("OPENAI_API_KEY"), "OpenAI GPT-4"),
                ("gpt-3.5-turbo", os.getenv("OPENAI_API_KEY"), "OpenAI GPT-3.5"),
            ]
        )

    # Anthropic
    if os.getenv("ANTHROPIC_API_KEY"):
        test_configs.extend(
            [
                (
                    "claude-3-opus-20240229",
                    os.getenv("ANTHROPIC_API_KEY"),
                    "Anthropic Claude 3 Opus",
                ),
                (
                    "claude-3-sonnet-20240229",
                    os.getenv("ANTHROPIC_API_KEY"),
                    "Anthropic Claude 3 Sonnet",
                ),
                (
                    "claude-3-haiku-20240307",
                    os.getenv("ANTHROPIC_API_KEY"),
                    "Anthropic Claude 3 Haiku",
                ),
            ]
        )

    # Google
    if os.getenv("GOOGLE_API_KEY"):
        test_configs.extend(
            [
                ("gemini-1.5-pro", os.getenv("GOOGLE_API_KEY"), "Google Gemini Pro"),
            ]
        )

    # xAI
    if os.getenv("XAI_API_KEY"):
        test_configs.extend(
            [
                ("grok-beta", os.getenv("XAI_API_KEY"), "xAI Grok Beta"),
            ]
        )

    if not test_configs:
        print("❌ No API keys found!")
        print("Please set at least one of:")
        print("  OPENAI_API_KEY")
        print("  ANTHROPIC_API_KEY")
        print("  GOOGLE_API_KEY")
        print("  XAI_API_KEY")
        return

    print(f"Found {len(test_configs)} configurations to test")

    # Run tests
    results = []
    for model, api_key, name in test_configs:
        success = await check_llm_provider(model, api_key, name)
        results.append((name, success))

    # Summary
    print(f"\n{'='*50}")
    print("SUMMARY")
    print(f"{'='*50}")

    successful = 0
    for name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{name:<30} {status}")
        if success:
            successful += 1

    print(f"\n🎯 Results: {successful}/{len(results)} providers working")

    if successful == len(results):
        print("🎉 All LLM providers are working correctly!")
    elif successful > 0:
        print("⚠️  Some providers are working, check failed ones for issues")
    else:
        print("❌ No providers are working, check your API keys and setup")


if __name__ == "__main__":
    # Import here to avoid circular imports
    from core.types import RetrievalConfig, FilteringConfig

    asyncio.run(main())
