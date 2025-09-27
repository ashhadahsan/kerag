"""
Example implementation of KERAG with Anthropic's Claude models.
"""

import asyncio
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


async def main():
    """Main example function for Anthropic-based KERAG."""

    # Configuration for Anthropic-based KERAG
    config = KERAGConfig(
        knowledge_base_type=KnowledgeBaseType.SPARQL,
        endpoint_url="http://dbpedia.org/sparql",  # DBpedia SPARQL endpoint
        llm_model="claude-3-opus-20240229",  # Anthropic's Claude model
        api_key="your-anthropic-api-key",  # Replace with your actual API key
        # Customize configurations
        retrieval_config=RetrievalConfig(
            max_hops=2, max_entities=500, include_attributes=True
        ),
        filtering_config=FilteringConfig(
            relevance_threshold=0.6, max_triples=300, use_llm_filtering=True
        ),
        summarization_config=SummarizationConfig(
            use_cot=True, max_tokens=3000, temperature=0.1
        ),
        debug=True,
    )

    # Example questions to test
    example_questions = [
        "Which books written by J.K. Rowling are related to magic?",
        "What movies has Tom Hanks starred in?",
        "Who is the president of the United States?",
        "What is the population of Tokyo?",
        "Which companies are headquartered in Silicon Valley?",
    ]

    # Initialize KERAG workflow
    async with AsyncKERAGWorkflow(config) as workflow:
        print("KERAG Anthropic Claude Example")
        print("=" * 50)

        for i, question in enumerate(example_questions, 1):
            print(f"\n{i}. Question: {question}")
            print("-" * 30)

            try:
                # Process question
                result = await workflow.process_question(question)

                # Display results
                print(f"Answer: {result.answer}")
                print(f"Confidence: {result.confidence:.2f}")
                print(f"Retrieved Entities: {len(result.retrieved_entities)}")
                print(f"Retrieved Triples: {len(result.retrieved_triples)}")

                if result.metadata:
                    print(f"Domain: {result.metadata.get('domain', 'Unknown')}")
                    if result.metadata.get("topic_entity"):
                        topic = result.metadata["topic_entity"]
                        print(
                            f"Topic Entity: {topic.get('name', 'Unknown')} ({topic.get('type', 'Unknown')})"
                        )

                # Show some retrieved entities
                if result.retrieved_entities:
                    print("\nSample Retrieved Entities:")
                    for entity in result.retrieved_entities[:3]:
                        print(f"  - {entity.name} ({entity.type})")

                # Show some retrieved triples
                if result.retrieved_triples:
                    print("\nSample Retrieved Triples:")
                    for triple in result.retrieved_triples[:3]:
                        print(
                            f"  - {triple.subject} --{triple.predicate}--> {triple.object}"
                        )

            except Exception as e:
                print(f"Error: {str(e)}")

            print("\n" + "=" * 50)


async def test_claude_models():
    """Test different Claude models."""

    claude_models = [
        "claude-3-opus-20240229",  # Most capable
        "claude-3-sonnet-20240229",  # Balanced
        "claude-3-haiku-20240307",  # Fastest
    ]

    question = "Which books written by J.K. Rowling are related to magic?"

    for model in claude_models:
        print(f"\nTesting {model}")
        print("-" * 40)

        config = KERAGConfig(
            knowledge_base_type=KnowledgeBaseType.SPARQL,
            endpoint_url="http://dbpedia.org/sparql",
            llm_model=model,
            api_key="your-anthropic-api-key",
            # Optimize for speed
            retrieval_config=RetrievalConfig(max_hops=1, max_entities=100),
            filtering_config=FilteringConfig(relevance_threshold=0.7, max_triples=50),
            timeout=20,
        )

        async with AsyncKERAGWorkflow(config) as workflow:
            try:
                import time

                start_time = time.time()

                result = await workflow.process_question(question)

                end_time = time.time()
                processing_time = end_time - start_time

                print(f"Answer: {result.answer}")
                print(f"Confidence: {result.confidence:.2f}")
                print(f"Processing Time: {processing_time:.2f}s")
                print(f"Retrieved Entities: {len(result.retrieved_entities)}")

            except Exception as e:
                print(f"Error with {model}: {str(e)}")


async def test_anthropic_with_api_kb():
    """Test Anthropic with API-based knowledge base."""

    config = KERAGConfig(
        knowledge_base_type=KnowledgeBaseType.API,
        endpoint_url="https://api.example-knowledge-base.com",
        llm_model="claude-3-sonnet-20240229",  # Good balance for API calls
        api_key="your-anthropic-api-key",
        # Optimize for API-based KB
        retrieval_config=RetrievalConfig(
            max_hops=3, max_entities=1000, include_attributes=True
        ),
        filtering_config=FilteringConfig(
            relevance_threshold=0.5, max_triples=500, use_llm_filtering=True
        ),
        summarization_config=SummarizationConfig(
            use_cot=True, max_tokens=4000, temperature=0.1
        ),
        debug=True,
    )

    api_questions = [
        "Which movies has Leonardo DiCaprio starred in?",
        "What are the top-grossing films of 2023?",
        "Who are the current members of the Beatles?",
        "What companies are in the Fortune 500?",
        "Which cities have hosted the Olympics?",
    ]

    async with AsyncKERAGWorkflow(config) as workflow:
        print("KERAG Anthropic + API Knowledge Base Example")
        print("=" * 60)

        for i, question in enumerate(api_questions, 1):
            print(f"\n{i}. Question: {question}")
            print("-" * 40)

            try:
                result = await workflow.process_question(question)

                print(f"Answer: {result.answer}")
                print(f"Confidence: {result.confidence:.2f}")

                # Show API-specific information
                if result.metadata:
                    print(f"Domain: {result.metadata.get('domain', 'Unknown')}")
                    if result.metadata.get("topic_entity"):
                        topic = result.metadata["topic_entity"]
                        print(f"Topic Entity: {topic.get('name', 'Unknown')}")

            except Exception as e:
                print(f"Error: {str(e)}")

            print()


async def performance_comparison():
    """Compare performance between different LLM providers."""
    import time

    question = "What is the capital of France?"

    # Test configurations
    configs = [
        {
            "name": "Claude 3 Opus",
            "model": "claude-3-opus-20240229",
            "api_key": "your-anthropic-api-key",
        },
        {
            "name": "Claude 3 Sonnet",
            "model": "claude-3-sonnet-20240229",
            "api_key": "your-anthropic-api-key",
        },
        {
            "name": "Claude 3 Haiku",
            "model": "claude-3-haiku-20240307",
            "api_key": "your-anthropic-api-key",
        },
        {"name": "GPT-4", "model": "gpt-4", "api_key": "your-openai-api-key"},
    ]

    results = []

    for config_info in configs:
        print(f"\nTesting {config_info['name']}...")

        config = KERAGConfig(
            knowledge_base_type=KnowledgeBaseType.SPARQL,
            endpoint_url="http://dbpedia.org/sparql",
            llm_model=config_info["model"],
            api_key=config_info["api_key"],
            # Optimize for speed comparison
            retrieval_config=RetrievalConfig(max_hops=1, max_entities=50),
            filtering_config=FilteringConfig(relevance_threshold=0.7, max_triples=25),
            timeout=15,
        )

        async with AsyncKERAGWorkflow(config) as workflow:
            try:
                start_time = time.time()
                result = await workflow.process_question(question)
                end_time = time.time()

                processing_time = end_time - start_time

                results.append(
                    {
                        "model": config_info["name"],
                        "time": processing_time,
                        "confidence": result.confidence,
                        "answer_length": len(result.answer),
                        "success": True,
                    }
                )

                print(f"  Time: {processing_time:.2f}s")
                print(f"  Confidence: {result.confidence:.2f}")
                print(f"  Answer: {result.answer[:100]}...")

            except Exception as e:
                results.append(
                    {
                        "model": config_info["name"],
                        "time": 0,
                        "confidence": 0,
                        "answer_length": 0,
                        "success": False,
                        "error": str(e),
                    }
                )
                print(f"  Error: {str(e)}")

    # Summary
    print(f"\n{'='*60}")
    print("PERFORMANCE COMPARISON SUMMARY")
    print(f"{'='*60}")
    print(f"{'Model':<20} {'Time(s)':<10} {'Confidence':<12} {'Success':<8}")
    print("-" * 60)

    for result in results:
        if result["success"]:
            print(
                f"{result['model']:<20} {result['time']:<10.2f} {result['confidence']:<12.2f} {'Yes':<8}"
            )
        else:
            print(f"{result['model']:<20} {'N/A':<10} {'N/A':<12} {'No':<8}")

    # Find fastest successful model
    successful_results = [r for r in results if r["success"]]
    if successful_results:
        fastest = min(successful_results, key=lambda x: x["time"])
        print(
            f"\nFastest successful model: {fastest['model']} ({fastest['time']:.2f}s)"
        )

        most_confident = max(successful_results, key=lambda x: x["confidence"])
        print(
            f"Most confident model: {most_confident['model']} ({most_confident['confidence']:.2f})"
        )


if __name__ == "__main__":
    # Run the main example
    print("Running KERAG Anthropic Example...")
    asyncio.run(main())

    print("\n" + "=" * 60 + "\n")

    # Test different Claude models
    print("Testing Different Claude Models...")
    asyncio.run(test_claude_models())

    print("\n" + "=" * 60 + "\n")

    # Test with API knowledge base
    print("Testing Anthropic with API Knowledge Base...")
    asyncio.run(test_anthropic_with_api_kb())

    print("\n" + "=" * 60 + "\n")

    # Performance comparison
    print("Running Performance Comparison...")
    asyncio.run(performance_comparison())
