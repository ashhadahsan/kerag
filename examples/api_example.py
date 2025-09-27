"""
Example implementation of KERAG with API-based knowledge base.
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
    """Main example function for API-based KERAG."""

    # Configuration for API-based KERAG
    config = KERAGConfig(
        knowledge_base_type=KnowledgeBaseType.API,
        endpoint_url="https://api.example-knowledge-base.com",  # Replace with actual API endpoint
        llm_model="gpt-4",
        api_key="your-openai-api-key",  # Replace with your actual API key
        # Customize configurations
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

    # Example questions to test
    example_questions = [
        "Which movies has Leonardo DiCaprio starred in?",
        "What are the top-grossing films of 2023?",
        "Who are the current members of the Beatles?",
        "What companies are in the Fortune 500?",
        "Which cities have hosted the Olympics?",
    ]

    # Initialize KERAG workflow
    async with AsyncKERAGWorkflow(config) as workflow:
        print("KERAG API Example")
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
                        # Show some attributes
                        for attr_key, attr_value in list(entity.attributes.items())[:2]:
                            if isinstance(attr_value, str) and len(attr_value) < 100:
                                print(f"    {attr_key}: {attr_value}")

                # Show some retrieved triples
                if result.retrieved_triples:
                    print("\nSample Retrieved Triples:")
                    for triple in result.retrieved_triples[:3]:
                        print(
                            f"  - {triple.subject} --{triple.predicate}--> {triple.object}"
                        )
                        if triple.confidence:
                            print(f"    (confidence: {triple.confidence})")

            except Exception as e:
                print(f"Error: {str(e)}")

            print("\n" + "=" * 50)


async def test_movie_domain():
    """Test specific movie domain questions."""

    config = KERAGConfig(
        knowledge_base_type=KnowledgeBaseType.API,
        endpoint_url="https://api.example-knowledge-base.com",
        llm_model="gpt-4",
        api_key="your-openai-api-key",
        # Optimize for movie domain
        retrieval_config=RetrievalConfig(
            max_hops=2,
            max_entities=300,
            relation_filters=[
                "actor",
                "director",
                "genre",
                "release_date",
                "box_office",
            ],
        ),
        filtering_config=FilteringConfig(relevance_threshold=0.6, max_triples=200),
        debug=True,
    )

    movie_questions = [
        "What movies has Tom Hanks directed?",
        "Which actors starred in the movie Titanic?",
        "What is the highest-grossing movie of all time?",
        "Who composed the music for the Lord of the Rings movies?",
        "What genre is the movie Inception?",
    ]

    async with AsyncKERAGWorkflow(config) as workflow:
        print("Movie Domain Test")
        print("=" * 50)

        for i, question in enumerate(movie_questions, 1):
            print(f"\n{i}. Question: {question}")
            print("-" * 30)

            try:
                result = await workflow.process_question(question)

                print(f"Answer: {result.answer}")
                print(f"Confidence: {result.confidence:.2f}")

                # Show movie-specific information
                if result.metadata.get("domain") == "movie":
                    print("Movie-specific analysis completed")

                # Show relevant entities
                movie_entities = [
                    e
                    for e in result.retrieved_entities
                    if "movie" in e.type.lower() or "film" in e.type.lower()
                ]
                if movie_entities:
                    print(f"Movie entities found: {len(movie_entities)}")
                    for entity in movie_entities[:2]:
                        print(f"  - {entity.name}")

            except Exception as e:
                print(f"Error: {str(e)}")

            print()


async def test_music_domain():
    """Test specific music domain questions."""

    config = KERAGConfig(
        knowledge_base_type=KnowledgeBaseType.API,
        endpoint_url="https://api.example-knowledge-base.com",
        llm_model="gpt-4",
        api_key="your-openai-api-key",
        # Optimize for music domain
        retrieval_config=RetrievalConfig(
            max_hops=2,
            max_entities=400,
            relation_filters=[
                "artist",
                "album",
                "song",
                "genre",
                "release_date",
                "label",
            ],
        ),
        filtering_config=FilteringConfig(relevance_threshold=0.5, max_triples=250),
        debug=True,
    )

    music_questions = [
        "What albums has Taylor Swift released?",
        "Who are the members of the band Queen?",
        "What genre is the song Bohemian Rhapsody?",
        "Which record label signed The Beatles?",
        "What year was the album Abbey Road released?",
    ]

    async with AsyncKERAGWorkflow(config) as workflow:
        print("Music Domain Test")
        print("=" * 50)

        for i, question in enumerate(music_questions, 1):
            print(f"\n{i}. Question: {question}")
            print("-" * 30)

            try:
                result = await workflow.process_question(question)

                print(f"Answer: {result.answer}")
                print(f"Confidence: {result.confidence:.2f}")

                # Show music-specific information
                if result.metadata.get("domain") == "music":
                    print("Music-specific analysis completed")

                # Show relevant entities
                music_entities = [
                    e
                    for e in result.retrieved_entities
                    if any(
                        keyword in e.type.lower()
                        for keyword in ["music", "song", "album", "artist", "band"]
                    )
                ]
                if music_entities:
                    print(f"Music entities found: {len(music_entities)}")
                    for entity in music_entities[:2]:
                        print(f"  - {entity.name} ({entity.type})")

            except Exception as e:
                print(f"Error: {str(e)}")

            print()


async def test_sports_domain():
    """Test specific sports domain questions."""

    config = KERAGConfig(
        knowledge_base_type=KnowledgeBaseType.API,
        endpoint_url="https://api.example-knowledge-base.com",
        llm_model="gpt-4",
        api_key="your-openai-api-key",
        # Optimize for sports domain
        retrieval_config=RetrievalConfig(
            max_hops=2,
            max_entities=500,
            relation_filters=[
                "player",
                "team",
                "league",
                "position",
                "statistics",
                "game",
                "season",
            ],
        ),
        filtering_config=FilteringConfig(relevance_threshold=0.6, max_triples=300),
        debug=True,
    )

    sports_questions = [
        "Which team does LeBron James play for?",
        "What is the record for most points scored in an NBA game?",
        "Who won the FIFA World Cup in 2022?",
        "Which player has the most career home runs in MLB?",
        "What is the fastest time ever recorded in the 100m dash?",
    ]

    async with AsyncKERAGWorkflow(config) as workflow:
        print("Sports Domain Test")
        print("=" * 50)

        for i, question in enumerate(sports_questions, 1):
            print(f"\n{i}. Question: {question}")
            print("-" * 30)

            try:
                result = await workflow.process_question(question)

                print(f"Answer: {result.answer}")
                print(f"Confidence: {result.confidence:.2f}")

                # Show sports-specific information
                if result.metadata.get("domain") == "sports":
                    print("Sports-specific analysis completed")

                # Show relevant entities
                sports_entities = [
                    e
                    for e in result.retrieved_entities
                    if any(
                        keyword in e.type.lower()
                        for keyword in ["player", "team", "league", "sport", "game"]
                    )
                ]
                if sports_entities:
                    print(f"Sports entities found: {len(sports_entities)}")
                    for entity in sports_entities[:2]:
                        print(f"  - {entity.name} ({entity.type})")

            except Exception as e:
                print(f"Error: {str(e)}")

            print()


async def test_finance_domain():
    """Test specific finance domain questions."""

    config = KERAGConfig(
        knowledge_base_type=KnowledgeBaseType.API,
        endpoint_url="https://api.example-knowledge-base.com",
        llm_model="gpt-4",
        api_key="your-openai-api-key",
        # Optimize for finance domain
        retrieval_config=RetrievalConfig(
            max_hops=2,
            max_entities=300,
            relation_filters=[
                "company",
                "stock",
                "revenue",
                "market_cap",
                "sector",
                "ceo",
                "founded",
            ],
        ),
        filtering_config=FilteringConfig(relevance_threshold=0.7, max_triples=200),
        debug=True,
    )

    finance_questions = [
        "What is the market cap of Apple Inc?",
        "Who is the CEO of Microsoft?",
        "Which sector does Tesla belong to?",
        "What was Amazon's revenue in 2023?",
        "Which companies are in the S&P 500?",
    ]

    async with AsyncKERAGWorkflow(config) as workflow:
        print("Finance Domain Test")
        print("=" * 50)

        for i, question in enumerate(finance_questions, 1):
            print(f"\n{i}. Question: {question}")
            print("-" * 30)

            try:
                result = await workflow.process_question(question)

                print(f"Answer: {result.answer}")
                print(f"Confidence: {result.confidence:.2f}")

                # Show finance-specific information
                if result.metadata.get("domain") == "finance":
                    print("Finance-specific analysis completed")

                # Show relevant entities
                finance_entities = [
                    e
                    for e in result.retrieved_entities
                    if any(
                        keyword in e.type.lower()
                        for keyword in ["company", "stock", "corporation", "business"]
                    )
                ]
                if finance_entities:
                    print(f"Finance entities found: {len(finance_entities)}")
                    for entity in finance_entities[:2]:
                        print(f"  - {entity.name} ({entity.type})")

            except Exception as e:
                print(f"Error: {str(e)}")

            print()


async def test_batch_processing():
    """Test batch processing of multiple questions."""
    import time

    config = KERAGConfig(
        knowledge_base_type=KnowledgeBaseType.API,
        endpoint_url="https://api.example-knowledge-base.com",
        llm_model="gpt-4",
        api_key="your-openai-api-key",
        # Optimize for batch processing
        retrieval_config=RetrievalConfig(max_hops=1, max_entities=200),
        filtering_config=FilteringConfig(relevance_threshold=0.6, max_triples=100),
        timeout=20,
    )

    batch_questions = [
        "What is the capital of Japan?",
        "Who wrote the novel 1984?",
        "What is the chemical symbol for gold?",
        "Which planet is closest to the Sun?",
        "What is the largest ocean on Earth?",
        "Who painted the Starry Night?",
        "What is the speed of light?",
        "Which country has the most population?",
        "What is the currency of Brazil?",
        "Who invented the telephone?",
    ]

    async with AsyncKERAGWorkflow(config) as workflow:
        print("Batch Processing Test")
        print("=" * 50)

        start_time = time.time()
        results = []

        for i, question in enumerate(batch_questions, 1):
            try:
                result = await workflow.process_question(question)
                results.append(
                    {
                        "question": question,
                        "answer": result.answer,
                        "confidence": result.confidence,
                        "success": True,
                    }
                )
                print(f"{i:2d}. ✓ {question[:50]}...")

            except Exception as e:
                results.append(
                    {
                        "question": question,
                        "answer": None,
                        "confidence": 0.0,
                        "success": False,
                        "error": str(e),
                    }
                )
                print(f"{i:2d}. ✗ {question[:50]}... - {str(e)[:30]}")

        end_time = time.time()
        total_time = end_time - start_time

        # Summary
        successful = sum(1 for r in results if r["success"])
        avg_confidence = sum(r["confidence"] for r in results if r["success"]) / max(
            successful, 1
        )

        print(f"\nBatch Processing Summary:")
        print(f"  Total Questions: {len(batch_questions)}")
        print(f"  Successful: {successful}")
        print(f"  Success Rate: {successful/len(batch_questions):.2%}")
        print(f"  Average Confidence: {avg_confidence:.2f}")
        print(f"  Total Time: {total_time:.2f}s")
        print(f"  Average Time per Question: {total_time/len(batch_questions):.2f}s")


if __name__ == "__main__":
    # Run the main example
    print("Running KERAG API Example...")
    asyncio.run(main())

    print("\n" + "=" * 60 + "\n")

    # Run domain-specific tests
    print("Running Movie Domain Test...")
    asyncio.run(test_movie_domain())

    print("\n" + "=" * 60 + "\n")

    print("Running Music Domain Test...")
    asyncio.run(test_music_domain())

    print("\n" + "=" * 60 + "\n")

    print("Running Sports Domain Test...")
    asyncio.run(test_sports_domain())

    print("\n" + "=" * 60 + "\n")

    print("Running Finance Domain Test...")
    asyncio.run(test_finance_domain())

    print("\n" + "=" * 60 + "\n")

    print("Running Batch Processing Test...")
    asyncio.run(test_batch_processing())
