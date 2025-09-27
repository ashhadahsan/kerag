#!/usr/bin/env python3
"""
Run benchmark with complex scenarios that properly test KERAG's capabilities.
"""

import asyncio
import os
import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from core.types import KERAGConfig, KnowledgeBaseType
from benchmarking.benchmark_framework import BenchmarkFramework
from benchmarking.complex_scenarios import get_complex_benchmark_datasets


async def main():
    """Run complex benchmark comparison."""

    print("🧠 KERAG vs Traditional RAG - Complex Scenarios Benchmark")
    print("=" * 60)

    # Check for API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("⚠️  Warning: OPENAI_API_KEY not found in environment variables")
        print("   Set your API key: export OPENAI_API_KEY='your-key-here'")
        print("   Using placeholder key for demonstration...")
        api_key = "your-api-key-here"

    # Get complex scenarios
    complex_datasets = get_complex_benchmark_datasets()

    print(f"📊 Found {len(complex_datasets)} complex scenario categories:")
    for category, scenarios in complex_datasets.items():
        print(f"   {category.replace('_', ' ').title()}: {len(scenarios)} scenarios")

    # Create benchmark framework with complex datasets
    benchmark = BenchmarkFramework()
    benchmark.datasets = complex_datasets  # Override with complex scenarios

    # Run benchmark on complex scenarios
    print("\n🚀 Starting Complex Scenarios Benchmark")
    print("=" * 60)

    results = await benchmark.run_comprehensive_benchmark(
        kerag_config=KERAGConfig(
            knowledge_base_type=KnowledgeBaseType.API,
            endpoint_url="https://api.example.com",  # Placeholder for API KB
            api_key=api_key,
            llm_model="gpt-4",
        )
    )

    # Print results
    print("\n🏆 COMPLEX SCENARIOS BENCHMARK RESULTS")
    print("=" * 60)

    # Group results by complexity
    complexity_results = {}
    for result in results:
        complexity = result.metadata.get("complexity", "unknown")
        if complexity not in complexity_results:
            complexity_results[complexity] = {"KERAG": [], "Traditional_RAG": []}
        complexity_results[complexity][result.method].append(result)

    # Print results by complexity
    for complexity in ["simple", "medium", "complex", "expert"]:
        if complexity in complexity_results:
            kerag_results = complexity_results[complexity]["KERAG"]
            trad_results = complexity_results[complexity]["Traditional_RAG"]

            if kerag_results and trad_results:
                kerag_accuracy = sum(r.accuracy_score for r in kerag_results) / len(
                    kerag_results
                )
                trad_accuracy = sum(r.accuracy_score for r in trad_results) / len(
                    trad_results
                )

                kerag_coverage = sum(r.coverage_score for r in kerag_results) / len(
                    kerag_results
                )
                trad_coverage = sum(r.coverage_score for r in trad_results) / len(
                    trad_results
                )

                kerag_recall = sum(r.retrieval_recall for r in kerag_results) / len(
                    kerag_results
                )
                trad_recall = sum(r.retrieval_recall for r in trad_results) / len(
                    trad_results
                )

                improvement = (
                    ((kerag_accuracy - trad_accuracy) / trad_accuracy * 100)
                    if trad_accuracy > 0
                    else 0
                )

                print(
                    f"\n📈 {complexity.upper()} COMPLEXITY ({len(kerag_results)} scenarios)"
                )
                print("-" * 50)
                print(
                    f"Accuracy:    KERAG {kerag_accuracy:.3f} vs Traditional {trad_accuracy:.3f} ({improvement:+.1f}%)"
                )
                print(
                    f"Coverage:    KERAG {kerag_coverage:.3f} vs Traditional {trad_coverage:.3f}"
                )
                print(
                    f"Retrieval:   KERAG {kerag_recall:.3f} vs Traditional {trad_recall:.3f}"
                )

    # Overall results
    all_kerag = [r for r in results if r.method == "KERAG"]
    all_trad = [r for r in results if r.method == "Traditional_RAG"]

    if all_kerag and all_trad:
        kerag_avg_accuracy = sum(r.accuracy_score for r in all_kerag) / len(all_kerag)
        trad_avg_accuracy = sum(r.accuracy_score for r in all_trad) / len(all_trad)

        kerag_avg_recall = sum(r.retrieval_recall for r in all_kerag) / len(all_kerag)
        trad_avg_recall = sum(r.retrieval_recall for r in all_trad) / len(all_trad)

        overall_improvement = (
            ((kerag_avg_accuracy - trad_avg_accuracy) / trad_avg_accuracy * 100)
            if trad_avg_accuracy > 0
            else 0
        )

        print(f"\n🎯 OVERALL COMPLEX SCENARIOS RESULTS")
        print("=" * 60)
        print(
            f"Average Accuracy: KERAG {kerag_avg_accuracy:.3f} vs Traditional {trad_avg_accuracy:.3f} ({overall_improvement:+.1f}%)"
        )
        print(
            f"Average Retrieval: KERAG {kerag_avg_recall:.3f} vs Traditional {trad_avg_recall:.3f}"
        )
        print(f"Total Scenarios: {len(all_kerag)} complex reasoning questions")

        if overall_improvement > 0:
            print(
                f"\n✅ KERAG shows {overall_improvement:.1f}% improvement on complex scenarios!"
            )
        else:
            print(
                f"\n⚠️  KERAG shows {overall_improvement:.1f}% performance on complex scenarios"
            )
            print(
                "   This suggests the scenarios may still not be complex enough for KERAG's strengths"
            )

    print(f"\n💾 Results saved to: benchmark_results/")
    print("✅ Complex scenarios benchmark completed!")


if __name__ == "__main__":
    asyncio.run(main())
