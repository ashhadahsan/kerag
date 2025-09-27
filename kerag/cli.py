"""
Command-line interface for KERAG.
"""

import asyncio
import argparse
import sys
from typing import Optional

from .workflows.kerag_workflow import KERAGWorkflow
from .knowledge_bases.sparql_kb import SPARQLKnowledgeBase
from .knowledge_bases.api_kb import APIKnowledgeBase
from .core.types import RetrievalConfig


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="KERAG: Knowledge-Enhanced Retrieval-Augmented Generation"
    )

    parser.add_argument("question", help="The question to answer")

    parser.add_argument(
        "--kb-type",
        choices=["sparql", "api"],
        default="sparql",
        help="Knowledge base type",
    )

    parser.add_argument("--kb-url", required=True, help="Knowledge base URL")

    parser.add_argument("--api-key", help="API key for the knowledge base")

    parser.add_argument("--llm-model", default="gpt-4", help="LLM model to use")

    parser.add_argument("--llm-api-key", help="API key for the LLM")

    parser.add_argument(
        "--max-hops",
        type=int,
        default=3,
        help="Maximum number of hops for neighborhood expansion",
    )

    parser.add_argument(
        "--max-entities",
        type=int,
        default=1000,
        help="Maximum number of entities to retrieve",
    )

    args = parser.parse_args()

    # Run the async main function
    asyncio.run(async_main(args))


async def async_main(args):
    """Async main function."""
    try:
        # Initialize knowledge base
        if args.kb_type == "sparql":
            kb = SPARQLKnowledgeBase(args.kb_url)
        else:
            kb = APIKnowledgeBase(args.kb_url, args.api_key)

        # Configure retrieval
        config = RetrievalConfig(
            max_hops=args.max_hops,
            max_entities=args.max_entities,
            max_triples=500,
            timeout=30,
            confidence_threshold=0.5,
        )

        # Initialize workflow
        workflow = KERAGWorkflow(
            knowledge_base=kb,
            retrieval_config=config,
            llm_model=args.llm_model,
            api_key=args.llm_api_key,
        )

        # Process question
        result = await workflow.process_question(args.question)

        if result.get("error"):
            print(f"Error: {result['error']}", file=sys.stderr)
            sys.exit(1)

        print(f"Answer: {result['answer']}")
        print(f"Confidence: {result['confidence']}")

    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

