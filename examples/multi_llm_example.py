"""
Example implementation of KERAG with multiple LLM providers.
Supports OpenAI GPT, Anthropic Claude, Google Gemini, and xAI Grok.
"""

import asyncio
import os
from typing import Dict, Any
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from core.types import (
    KERAGConfig,
    KnowledgeBaseType,
    RetrievalConfig,
    FilteringConfig,
    SummarizationConfig,
)
from workflows.kerag_workflow import AsyncKERAGWorkflow


async def test_all_llm_providers():
    """Test all available LLM providers with KERAG."""

    # Define configurations for different LLM providers
    llm_configs = [
        {
            "name": "OpenAI GPT-4",
            "model": "gpt-4",
            "api_key": os.getenv("OPENAI_API_KEY"),
            "api_key_name": "OPENAI_API_KEY",
        },
        {
            "name": "OpenAI GPT-3.5 Turbo",
            "model": "gpt-3.5-turbo",
            "api_key": os.getenv("OPENAI_API_KEY"),
            "api_key_name": "OPENAI_API_KEY",
        },
        {
            "name": "Anthropic Claude 3 Opus",
            "model": "claude-3-opus-20240229",
            "api_key": os.getenv("ANTHROPIC_API_KEY"),
            "api_key_name": "ANTHROPIC_API_KEY",
        },
        {
            "name": "Anthropic Claude 3 Sonnet",
            "model": "claude-3-sonnet-20240229",
            "api_key": os.getenv("ANTHROPIC_API_KEY"),
            "api_key_name": "ANTHROPIC_API_KEY",
        },
        {
            "name": "Anthropic Claude 3 Haiku",
            "model": "claude-3-haiku-20240307",
            "api_key": os.getenv("ANTHROPIC_API_KEY"),
            "api_key_name": "ANTHROPIC_API_KEY",
        },
        {
            "name": "Google Gemini Pro",
            "model": "gemini-1.5-pro",
            "api_key": os.getenv("GOOGLE_API_KEY"),
            "api_key_name": "GOOGLE_API_KEY",
        },
        {
            "name": "Google Gemini Pro Vision",
            "model": "gemini-pro-vision",
            "api_key": os.getenv("GOOGLE_API_KEY"),
            "api_key_name": "GOOGLE_API_KEY",
        },
        {
            "name": "xAI Grok Beta",
            "model": "grok-beta",
            "api_key": os.getenv("XAI_API_KEY"),
            "api_key_name": "XAI_API_KEY",
        },
    ]

    question = "Which books written by J.K. Rowling are related to magic?"

    print("KERAG Multi-LLM Provider Test")
    print("=" * 60)
    print(f"Question: {question}")
    print("=" * 60)

    results = []

    for config_info in llm_configs:
        print(f"\nTesting {config_info['name']}...")
        print("-" * 40)

        # Check if API key is available
        if not config_info["api_key"]:
            print(
                f"  ⚠️  Skipping {config_info['name']} - {config_info['api_key_name']} not found"
            )
            results.append(
                {
                    "name": config_info["name"],
                    "status": "skipped",
                    "reason": f"{config_info['api_key_name']} not found",
                }
            )
            continue

        # Create KERAG configuration
        config = KERAGConfig(
            knowledge_base_type=KnowledgeBaseType.SPARQL,
            endpoint_url="http://dbpedia.org/sparql",
            llm_model=config_info["model"],
            api_key=config_info["api_key"],
            # Optimize for speed
            retrieval_config=RetrievalConfig(max_hops=1, max_entities=100),
            filtering_config=FilteringConfig(relevance_threshold=0.7, max_triples=50),
            summarization_config=SummarizationConfig(
                use_cot=True, max_tokens=2000, temperature=0.1
            ),
            timeout=30,
        )

        try:
            async with AsyncKERAGWorkflow(config) as workflow:
                import time

                start_time = time.time()

                result = await workflow.process_question(question)

                end_time = time.time()
                processing_time = end_time - start_time

                print(f"  ✅ Success!")
                print(f"  Time: {processing_time:.2f}s")
                print(f"  Confidence: {result.confidence:.2f}")
                print(f"  Answer: {result.answer[:100]}...")
                print(f"  Entities: {len(result.retrieved_entities)}")
                print(f"  Triples: {len(result.retrieved_triples)}")

                results.append(
                    {
                        "name": config_info["name"],
                        "status": "success",
                        "time": processing_time,
                        "confidence": result.confidence,
                        "answer_length": len(result.answer),
                        "entities": len(result.retrieved_entities),
                        "triples": len(result.retrieved_triples),
                    }
                )

        except Exception as e:
            print(f"  ❌ Error: {str(e)[:100]}...")
            results.append(
                {"name": config_info["name"], "status": "error", "error": str(e)}
            )

    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(
        f"{'Provider':<25} {'Status':<10} {'Time(s)':<10} {'Confidence':<12} {'Entities':<10}"
    )
    print("-" * 80)

    for result in results:
        if result["status"] == "success":
            print(
                f"{result['name']:<25} {result['status']:<10} {result['time']:<10.2f} {result['confidence']:<12.2f} {result['entities']:<10}"
            )
        elif result["status"] == "skipped":
            print(
                f"{result['name']:<25} {result['status']:<10} {'N/A':<10} {'N/A':<12} {'N/A':<10}"
            )
        else:
            print(
                f"{result['name']:<25} {result['status']:<10} {'N/A':<10} {'N/A':<12} {'N/A':<10}"
            )

    # Find best performers
    successful_results = [r for r in results if r["status"] == "success"]
    if successful_results:
        fastest = min(successful_results, key=lambda x: x["time"])
        most_confident = max(successful_results, key=lambda x: x["confidence"])

        print(f"\n🏆 Best Performance:")
        print(f"  Fastest: {fastest['name']} ({fastest['time']:.2f}s)")
        print(
            f"  Most Confident: {most_confident['name']} ({most_confident['confidence']:.2f})"
        )

    return results


async def test_gemini_specific():
    """Test Google Gemini with specific configurations."""

    if not os.getenv("GOOGLE_API_KEY"):
        print("⚠️  GOOGLE_API_KEY not found, skipping Gemini test")
        return

    gemini_models = ["gemini-pro", "gemini-pro-vision"]

    question = "What are the main themes in J.K. Rowling's Harry Potter series?"

    print("\n" + "=" * 60)
    print("GOOGLE GEMINI SPECIFIC TEST")
    print("=" * 60)
    print(f"Question: {question}")
    print("-" * 60)

    for model in gemini_models:
        print(f"\nTesting {model}...")

        config = KERAGConfig(
            knowledge_base_type=KnowledgeBaseType.SPARQL,
            endpoint_url="http://dbpedia.org/sparql",
            llm_model=model,
            api_key=os.getenv("GOOGLE_API_KEY"),
            # Optimize for Gemini
            retrieval_config=RetrievalConfig(max_hops=2, max_entities=200),
            filtering_config=FilteringConfig(relevance_threshold=0.6, max_triples=100),
            summarization_config=SummarizationConfig(
                use_cot=True, max_tokens=3000, temperature=0.2
            ),
        )

        try:
            async with AsyncKERAGWorkflow(config) as workflow:
                result = await workflow.process_question(question)

                print(f"✅ {model} Success!")
                print(f"Answer: {result.answer}")
                print(f"Confidence: {result.confidence:.2f}")

        except Exception as e:
            print(f"❌ {model} Error: {str(e)}")


async def test_grok_specific():
    """Test xAI Grok with specific configurations."""

    if not os.getenv("XAI_API_KEY"):
        print("⚠️  XAI_API_KEY not found, skipping Grok test")
        return

    grok_models = ["grok-beta", "grok-2-1212"]

    question = "What makes J.K. Rowling's writing style unique?"

    print("\n" + "=" * 60)
    print("XAI GROK SPECIFIC TEST")
    print("=" * 60)
    print(f"Question: {question}")
    print("-" * 60)

    for model in grok_models:
        print(f"\nTesting {model}...")

        config = KERAGConfig(
            knowledge_base_type=KnowledgeBaseType.SPARQL,
            endpoint_url="http://dbpedia.org/sparql",
            llm_model=model,
            api_key=os.getenv("XAI_API_KEY"),
            # Optimize for Grok
            retrieval_config=RetrievalConfig(max_hops=2, max_entities=150),
            filtering_config=FilteringConfig(relevance_threshold=0.7, max_triples=75),
            summarization_config=SummarizationConfig(
                use_cot=True, max_tokens=2500, temperature=0.1
            ),
        )

        try:
            async with AsyncKERAGWorkflow(config) as workflow:
                result = await workflow.process_question(question)

                print(f"✅ {model} Success!")
                print(f"Answer: {result.answer}")
                print(f"Confidence: {result.confidence:.2f}")

        except Exception as e:
            print(f"❌ {model} Error: {str(e)}")


async def performance_benchmark():
    """Run performance benchmark across all available providers."""

    print("\n" + "=" * 80)
    print("PERFORMANCE BENCHMARK")
    print("=" * 80)

    # Simple question for benchmarking
    benchmark_question = "What is the capital of France?"

    # Get available providers
    available_providers = []

    if os.getenv("OPENAI_API_KEY"):
        available_providers.extend(
            [
                {
                    "name": "GPT-4",
                    "model": "gpt-4",
                    "api_key": os.getenv("OPENAI_API_KEY"),
                },
                {
                    "name": "GPT-3.5",
                    "model": "gpt-3.5-turbo",
                    "api_key": os.getenv("OPENAI_API_KEY"),
                },
            ]
        )

    if os.getenv("ANTHROPIC_API_KEY"):
        available_providers.extend(
            [
                {
                    "name": "Claude Opus",
                    "model": "claude-3-opus-20240229",
                    "api_key": os.getenv("ANTHROPIC_API_KEY"),
                },
                {
                    "name": "Claude Sonnet",
                    "model": "claude-3-sonnet-20240229",
                    "api_key": os.getenv("ANTHROPIC_API_KEY"),
                },
                {
                    "name": "Claude Haiku",
                    "model": "claude-3-haiku-20240307",
                    "api_key": os.getenv("ANTHROPIC_API_KEY"),
                },
            ]
        )

    if os.getenv("GOOGLE_API_KEY"):
        available_providers.append(
            {
                "name": "Gemini Pro",
                "model": "gemini-pro",
                "api_key": os.getenv("GOOGLE_API_KEY"),
            }
        )

    if os.getenv("XAI_API_KEY"):
        available_providers.append(
            {
                "name": "Grok Beta",
                "model": "grok-beta",
                "api_key": os.getenv("XAI_API_KEY"),
            }
        )

    if not available_providers:
        print("❌ No API keys found. Please set environment variables:")
        print("   OPENAI_API_KEY, ANTHROPIC_API_KEY, GOOGLE_API_KEY, or XAI_API_KEY")
        return

    print(f"Running benchmark with {len(available_providers)} providers...")
    print(f"Question: {benchmark_question}")
    print("-" * 80)

    benchmark_results = []

    for provider in available_providers:
        print(f"\nBenchmarking {provider['name']}...")

        config = KERAGConfig(
            knowledge_base_type=KnowledgeBaseType.SPARQL,
            endpoint_url="http://dbpedia.org/sparql",
            llm_model=provider["model"],
            api_key=provider["api_key"],
            # Optimize for speed
            retrieval_config=RetrievalConfig(max_hops=1, max_entities=50),
            filtering_config=FilteringConfig(relevance_threshold=0.8, max_triples=25),
            timeout=20,
        )

        try:
            async with AsyncKERAGWorkflow(config) as workflow:
                import time

                start_time = time.time()

                result = await workflow.process_question(benchmark_question)

                end_time = time.time()
                processing_time = end_time - start_time

                benchmark_results.append(
                    {
                        "name": provider["name"],
                        "model": provider["model"],
                        "time": processing_time,
                        "confidence": result.confidence,
                        "answer": result.answer,
                        "success": True,
                    }
                )

                print(f"  ⏱️  Time: {processing_time:.2f}s")
                print(f"  🎯 Confidence: {result.confidence:.2f}")
                print(f"  📝 Answer: {result.answer[:80]}...")

        except Exception as e:
            print(f"  ❌ Error: {str(e)[:50]}...")
            benchmark_results.append(
                {
                    "name": provider["name"],
                    "model": provider["model"],
                    "time": 0,
                    "confidence": 0,
                    "answer": "",
                    "success": False,
                    "error": str(e),
                }
            )

    # Benchmark summary
    print(f"\n{'='*80}")
    print("BENCHMARK SUMMARY")
    print(f"{'='*80}")
    print(
        f"{'Provider':<15} {'Model':<20} {'Time(s)':<10} {'Confidence':<12} {'Status':<8}"
    )
    print("-" * 80)

    successful_benchmarks = [r for r in benchmark_results if r["success"]]

    for result in benchmark_results:
        status = "✅" if result["success"] else "❌"
        time_str = f"{result['time']:.2f}" if result["success"] else "N/A"
        conf_str = f"{result['confidence']:.2f}" if result["success"] else "N/A"

        print(
            f"{result['name']:<15} {result['model']:<20} {time_str:<10} {conf_str:<12} {status:<8}"
        )

    if successful_benchmarks:
        # Rankings
        fastest = min(successful_benchmarks, key=lambda x: x["time"])
        most_confident = max(successful_benchmarks, key=lambda x: x["confidence"])

        print(f"\n🏆 RANKINGS:")
        print(f"  🚀 Fastest: {fastest['name']} ({fastest['time']:.2f}s)")
        print(
            f"  🎯 Most Confident: {most_confident['name']} ({most_confident['confidence']:.2f})"
        )

        # Average performance
        avg_time = sum(r["time"] for r in successful_benchmarks) / len(
            successful_benchmarks
        )
        avg_confidence = sum(r["confidence"] for r in successful_benchmarks) / len(
            successful_benchmarks
        )

        print(f"  📊 Average Time: {avg_time:.2f}s")
        print(f"  📊 Average Confidence: {avg_confidence:.2f}")


async def main():
    """Main function to run all tests."""

    print("🚀 KERAG Multi-LLM Provider Test Suite")
    print("=" * 60)

    # Check environment variables
    env_vars = ["OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GOOGLE_API_KEY", "XAI_API_KEY"]
    available_keys = [var for var in env_vars if os.getenv(var)]

    if not available_keys:
        print("❌ No API keys found!")
        print("Please set at least one of the following environment variables:")
        for var in env_vars:
            print(f"   export {var}=your_api_key_here")
        return

    print(f"✅ Found API keys for: {', '.join(available_keys)}")

    # Run tests
    await test_all_llm_providers()
    await test_gemini_specific()
    await test_grok_specific()
    await performance_benchmark()

    print(f"\n🎉 All tests completed!")


if __name__ == "__main__":
    asyncio.run(main())
