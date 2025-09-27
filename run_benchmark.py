#!/usr/bin/env python3
"""
Quick benchmark runner for KERAG vs Traditional RAG comparison.
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


async def main():
    """Run a quick benchmark comparison."""

    print("🚀 KERAG vs Traditional RAG Benchmark")
    print("=" * 50)

    # Check for API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("⚠️  Warning: OPENAI_API_KEY not found in environment variables")
        print("   Set your API key: export OPENAI_API_KEY='your-key-here'")
        print("   Using placeholder key for demonstration...")
        api_key = "your-api-key-here"

    # Configure KERAG
    kerag_config = KERAGConfig(
        knowledge_base_type=KnowledgeBaseType.SPARQL,
        endpoint_url="http://dbpedia.org/sparql",
        llm_model="gpt-4",
        api_key=api_key,
        debug=False,  # Set to True for detailed logging
        timeout=30,
    )

    # Initialize benchmark framework
    benchmark = BenchmarkFramework(output_dir="benchmark_results")

    try:
        # Run comprehensive benchmark
        print("\n📊 Starting benchmark comparison...")
        results = await benchmark.run_comprehensive_benchmark(
            kerag_config, save_results=True
        )

        print(f"\n✅ Benchmark completed successfully!")
        print(f"   Total datasets: {len(benchmark.datasets)}")
        print(f"   Total methods compared: 2 (KERAG vs Traditional RAG)")
        print(f"   Results saved to: benchmark_results/")

        # Print quick summary
        print(f"\n📈 Quick Summary:")
        kerag_results = [s for k, s in results.items() if "KERAG" in k]
        traditional_results = [s for k, s in results.items() if "Traditional_RAG" in k]

        if kerag_results and traditional_results:
            avg_kerag_accuracy = sum(s.average_accuracy for s in kerag_results) / len(
                kerag_results
            )
            avg_traditional_accuracy = sum(
                s.average_accuracy for s in traditional_results
            ) / len(traditional_results)

            improvement = (
                (avg_kerag_accuracy - avg_traditional_accuracy)
                / max(avg_traditional_accuracy, 0.001)
            ) * 100

            print(f"   KERAG Average Accuracy: {avg_kerag_accuracy:.3f}")
            print(
                f"   Traditional RAG Average Accuracy: {avg_traditional_accuracy:.3f}"
            )
            print(f"   Improvement: {improvement:+.1f}%")

    except Exception as e:
        print(f"❌ Benchmark failed: {str(e)}")
        print("   Make sure you have:")
        print("   1. Valid API key set in environment")
        print("   2. Internet connection for SPARQL endpoint")
        print("   3. All dependencies installed")
        return 1

    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
