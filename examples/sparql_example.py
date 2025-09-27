"""
Example implementation of KERAG with SPARQL-based knowledge base.
"""

import asyncio
from typing import Dict, Any

from core.types import (
    KERAGConfig,
    KnowledgeBaseType,
    RetrievalConfig,
    FilteringConfig,
    SummarizationConfig,
)
from workflows.kerag_workflow import AsyncKERAGWorkflow


async def main():
    """Main example function."""

    # Configuration for SPARQL-based KERAG
    config = KERAGConfig(
        knowledge_base_type=KnowledgeBaseType.SPARQL,
        endpoint_url="http://dbpedia.org/sparql",  # DBpedia SPARQL endpoint
        llm_model="gpt-4",
        api_key="your-openai-api-key",  # Replace with your actual API key
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
        print("KERAG SPARQL Example")
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


async def test_specific_question():
    """Test a specific question in detail."""

    config = KERAGConfig(
        knowledge_base_type=KnowledgeBaseType.SPARQL,
        endpoint_url="http://dbpedia.org/sparql",
        llm_model="gpt-4",
        api_key="your-openai-api-key",
        debug=True,
    )

    question = "Which books written by J.K. Rowling are related to magic?"

    async with AsyncKERAGWorkflow(config) as workflow:
        print("Detailed KERAG Analysis")
        print("=" * 50)
        print(f"Question: {question}")
        print()

        # Get the workflow graph for inspection
        graph = workflow.get_graph()
        print("Workflow Graph Nodes:", list(graph.get_graph().nodes()))
        print()

        # Process the question step by step
        result = await workflow.process_question(question)

        print("Final Result:")
        print(f"Answer: {result.answer}")
        print(f"Confidence: {result.confidence}")
        print()

        print("Detailed Metadata:")
        for key, value in result.metadata.items():
            if isinstance(value, dict):
                print(f"  {key}:")
                for sub_key, sub_value in value.items():
                    print(f"    {sub_key}: {sub_value}")
            else:
                print(f"  {key}: {value}")
        print()

        print("All Retrieved Entities:")
        for entity in result.retrieved_entities:
            print(f"  - {entity.name} ({entity.type})")
            for attr_key, attr_value in entity.attributes.items():
                if isinstance(attr_value, str) and len(attr_value) < 100:
                    print(f"    {attr_key}: {attr_value}")
        print()

        print("All Retrieved Triples:")
        for triple in result.retrieved_triples:
            print(f"  - {triple.subject} --{triple.predicate}--> {triple.object}")
            if triple.confidence:
                print(f"    (confidence: {triple.confidence})")


async def test_error_handling():
    """Test error handling scenarios."""

    config = KERAGConfig(
        knowledge_base_type=KnowledgeBaseType.SPARQL,
        endpoint_url="http://invalid-endpoint.com/sparql",  # Invalid endpoint
        llm_model="gpt-4",
        api_key="invalid-api-key",  # Invalid API key
        debug=True,
    )

    question = "What is the meaning of life?"

    async with AsyncKERAGWorkflow(config) as workflow:
        print("Error Handling Test")
        print("=" * 50)
        print(f"Question: {question}")
        print()

        try:
            result = await workflow.process_question(question)
            print(f"Unexpected success: {result.answer}")
        except Exception as e:
            print(f"Expected error: {str(e)}")


async def performance_test():
    """Test performance with multiple questions."""
    import time

    config = KERAGConfig(
        knowledge_base_type=KnowledgeBaseType.SPARQL,
        endpoint_url="http://dbpedia.org/sparql",
        llm_model="gpt-4",
        api_key="your-openai-api-key",
        # Optimize for performance
        retrieval_config=RetrievalConfig(max_hops=1, max_entities=100),
        filtering_config=FilteringConfig(relevance_threshold=0.7, max_triples=50),
        timeout=15,
    )

    questions = [
        "Who is the author of Harry Potter?",
        "What is the capital of France?",
        "Who directed the movie Inception?",
        "What is the largest planet in our solar system?",
        "Who painted the Mona Lisa?",
    ]

    async with AsyncKERAGWorkflow(config) as workflow:
        print("Performance Test")
        print("=" * 50)

        total_time = 0
        successful_questions = 0

        for i, question in enumerate(questions, 1):
            start_time = time.time()

            try:
                result = await workflow.process_question(question)
                end_time = time.time()

                processing_time = end_time - start_time
                total_time += processing_time
                successful_questions += 1

                print(f"{i}. {question}")
                print(f"   Answer: {result.answer}")
                print(f"   Time: {processing_time:.2f}s")
                print(f"   Confidence: {result.confidence:.2f}")
                print()

            except Exception as e:
                end_time = time.time()
                processing_time = end_time - start_time
                total_time += processing_time

                print(f"{i}. {question}")
                print(f"   Error: {str(e)}")
                print(f"   Time: {processing_time:.2f}s")
                print()

        if successful_questions > 0:
            avg_time = total_time / len(questions)
            success_rate = successful_questions / len(questions)

            print(f"Performance Summary:")
            print(f"  Total Questions: {len(questions)}")
            print(f"  Successful: {successful_questions}")
            print(f"  Success Rate: {success_rate:.2%}")
            print(f"  Average Time: {avg_time:.2f}s")
            print(f"  Total Time: {total_time:.2f}s")


if __name__ == "__main__":
    # Run the main example
    print("Running KERAG SPARQL Example...")
    asyncio.run(main())

    print("\n" + "=" * 60 + "\n")

    # Run detailed analysis
    print("Running Detailed Analysis...")
    asyncio.run(test_specific_question())

    print("\n" + "=" * 60 + "\n")

    # Run error handling test
    print("Running Error Handling Test...")
    asyncio.run(test_error_handling())

    print("\n" + "=" * 60 + "\n")

    # Run performance test
    print("Running Performance Test...")
    asyncio.run(performance_test())
