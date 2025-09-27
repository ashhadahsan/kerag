# KERAG: Knowledge-Enhanced Retrieval-Augmented Generation

[![Coverage](https://img.shields.io/badge/coverage-52%25-yellow)](https://github.com/your-username/kerag)
[![Python](https://img.shields.io/badge/python-3.8%2B-blue)](https://python.org)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

A Python implementation of the KERAG (Knowledge-Enhanced Retrieval-Augmented Generation) pipeline for advanced question answering using knowledge graphs.

## Overview

KERAG is a novel KG-based RAG pipeline that enhances QA coverage by retrieving a broader subgraph likely to contain relevant information. The implementation follows the methodology described in the paper "KERAG: Knowledge-Enhanced Retrieval-Augmented Generation for Advanced Question Answering" by Sun et al.

## Key Features

- **Entity-level Retrieval**: Retrieves entire entity neighborhoods rather than specific triples
- **Multi-hop Neighborhood Expansion**: Supports configurable hop limits for comprehensive knowledge retrieval
- **LLM-based Filtering**: Uses language models for relevance scoring and semantic filtering
- **Chain-of-Thought Reasoning**: Implements CoT-based answer generation for complex questions
- **Multiple Knowledge Base Support**: Works with both SPARQL-based and API-based knowledge bases
- **Async/Await Support**: Full asynchronous implementation for better performance
- **Extensible Architecture**: Supports multiple LLM providers and knowledge base types

## Architecture

The KERAG pipeline consists of four main phases:

### 1. Planning Phase

- **Entity Identification**: Identifies the main topic entity from the question
- **Domain Detection**: Determines the domain (movie, music, sports, finance, book, person, organization, location)
- **Retrieval Plan Generation**: Creates retrieval plans with relations, max_hops, and filters

### 2. Retrieval Phase

- **Neighborhood Expansion**: Expands entity neighborhoods with multi-hop exploration
- **Knowledge Base Integration**: Supports both SPARQL and API-based knowledge bases
- **Entity-level Retrieval**: Retrieves comprehensive entity information

### 3. Filtering Phase

- **Relevance Scoring**: Uses LLMs to compute relevance scores for entities and triples
- **Semantic Filtering**: Removes irrelevant content based on question semantics
- **Knowledge Overloading Management**: Limits retrieved content to prevent context window overflow

### 4. Summarization Phase

- **Chain-of-Thought Reasoning**: Implements CoT-based answer generation
- **Confidence Scoring**: Computes confidence scores for generated answers
- **Complex Question Handling**: Supports aggregation and reasoning questions

## Installation

### Prerequisites

- Python 3.8+
- pip or conda

### Dependencies

Install the required dependencies:

```bash
pip install -r requirements.txt
```

### Environment Setup

Create a `.env` file with your API keys:

```bash
# OpenAI
OPENAI_API_KEY=your_openai_api_key

# Anthropic
ANTHROPIC_API_KEY=your_anthropic_api_key

# Google
GOOGLE_API_KEY=your_google_api_key

# xAI
XAI_API_KEY=your_xai_api_key

# General API key
API_KEY=your_general_api_key
```

## Usage

### Basic Usage

```python
from workflows.kerag_workflow import KERAGWorkflow
from knowledge_bases.sparql_kb import SPARQLKnowledgeBase
from core.types import RetrievalConfig

# Initialize knowledge base
kb = SPARQLKnowledgeBase("http://your-sparql-endpoint.com/sparql")

# Configure retrieval
config = RetrievalConfig(
    max_hops=3,
    max_entities=1000,
    max_triples=500,
    timeout=30,
    confidence_threshold=0.5
)

# Initialize workflow
workflow = KERAGWorkflow(
    knowledge_base=kb,
    retrieval_config=config,
    llm_model="gpt-4",
    api_key="your_api_key"
)

# Process a question
async def ask_question(question: str):
    result = await workflow.process_question(question)
    return result

# Example usage
question = "Which books written by J.K. Rowling are related to magic?"
answer = await ask_question(question)
print(f"Answer: {answer['answer']}")
print(f"Confidence: {answer['confidence']}")
```

### API-based Knowledge Base

```python
from knowledge_bases.api_kb import APIKnowledgeBase

# Initialize API knowledge base
kb = APIKnowledgeBase("https://your-api-endpoint.com", "your_api_key")

# Use with workflow
workflow = KERAGWorkflow(
    knowledge_base=kb,
    retrieval_config=config,
    llm_model="gpt-4",
    api_key="your_api_key"
)
```

### Advanced Configuration

```python
from core.types import (
    RetrievalConfig,
    FilteringConfig,
    SummarizationConfig
)

# Configure retrieval
retrieval_config = RetrievalConfig(
    max_hops=3,
    max_entities=1000,
    max_triples=500,
    relation_filters=["author", "wrote"],
    include_attributes=True,
    timeout=30,
    confidence_threshold=0.5
)

# Configure filtering
filtering_config = FilteringConfig(
    relevance_threshold=0.7,
    max_triples=100,
    use_llm_filtering=True
)

# Configure summarization
summarization_config = SummarizationConfig(
    use_cot=True,
    max_tokens=1000,
    temperature=0.1,
    strategy="cot"
)
```

## Testing

### Run All Tests

```bash
python -m pytest
```

### Run Specific Test Categories

```bash
# Core functionality tests
python -m pytest tests/test_chains.py tests/test_knowledge_bases.py tests/test_workflows.py

# Workflow tests only
python -m pytest tests/test_workflows.py

# Chain tests only
python -m pytest tests/test_chains.py
```

### Test Coverage

```bash
# Run tests with coverage
python -m pytest --cov=. --cov-report=html --cov-report=term

# Generate coverage report
python -m pytest --cov=. --cov-report=html
```

## Project Structure

```
kerag/
├── agents/                    # Agent implementations
├── chains/                    # LangChain LCEL chains
│   ├── planning.py           # Planning chain
│   ├── retrieval.py          # Retrieval chain
│   ├── filtering.py          # Filtering chain
│   └── summarization.py      # Summarization chain
├── config/                    # Configuration management
│   └── settings.py           # Settings and prompts
├── core/                      # Core interfaces and types
│   ├── interfaces.py         # Interface definitions
│   ├── types.py              # Data models
│   └── exceptions.py         # Custom exceptions
├── knowledge_bases/           # Knowledge base implementations
│   ├── sparql_kb.py          # SPARQL knowledge base
│   └── api_kb.py             # API knowledge base
├── workflows/                 # Workflow orchestration
│   └── kerag_workflow.py     # Main KERAG workflow
├── tests/                     # Test suite
│   ├── test_chains.py        # Chain tests
│   ├── test_knowledge_bases.py # Knowledge base tests
│   ├── test_workflows.py     # Workflow tests
│   └── test_*.py             # Additional test files
├── requirements.txt          # Dependencies
├── .env.example             # Environment variables template
└── README.md                # This file
```

## Configuration

### Environment Variables

| Variable            | Description       | Required |
| ------------------- | ----------------- | -------- |
| `OPENAI_API_KEY`    | OpenAI API key    | Optional |
| `ANTHROPIC_API_KEY` | Anthropic API key | Optional |
| `GOOGLE_API_KEY`    | Google API key    | Optional |
| `XAI_API_KEY`       | xAI API key       | Optional |
| `API_KEY`           | General API key   | Optional |

### LLM Models Supported

- **OpenAI**: gpt-4, gpt-3.5-turbo
- **Anthropic**: claude-3-opus, claude-3-sonnet, claude-3-haiku
- **Google**: gemini-1.5-pro, gemini-1.5-flash
- **xAI**: grok-1, grok-2

### Knowledge Base Types

- **SPARQL**: For SPARQL-based knowledge graphs
- **API**: For REST API-based knowledge bases

## Examples

### Simple Question Answering

```python
# Simple question
question = "What is the capital of France?"
result = await workflow.process_question(question)
print(result['answer'])  # "The capital of France is Paris."
```

### Complex Multi-hop Questions

```python
# Complex question requiring multi-hop reasoning
question = "Which books written by J.K. Rowling have been adapted into movies?"
result = await workflow.process_question(question)
print(result['answer'])
```

### Domain-specific Questions

```python
# Movie domain question
question = "Which movies did Tom Hanks star in during the 1990s?"
result = await workflow.process_question(question)
print(result['answer'])
```

## Performance

### Test Results

- **Core functionality**: 42/48 tests passing (87.5% success rate)
- **Workflow tests**: 15/15 tests passing (100% success rate)
- **Chain tests**: 18/18 tests passing (100% success rate)
- **Knowledge base tests**: 9/14 tests passing (with some mocking issues in extended tests)

### Test Coverage

Current test coverage is **52%** overall with the following breakdown:

- **Core types and interfaces**: 100% coverage
- **Workflow orchestration**: 91% coverage
- **Knowledge base implementations**: 48-66% coverage
- **Chain implementations**: 53-68% coverage

The implementation follows the KERAG methodology and supports:

- Multi-hop neighborhood expansion
- Entity-level retrieval
- LLM-based filtering and reasoning
- Chain-of-Thought answer generation

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Run the test suite
6. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this implementation in your research, please cite the original paper:

```bibtex
@article{sun2025kerag,
  title={KERAG: Knowledge-Enhanced Retrieval-Augmented Generation for Advanced Question Answering},
  author={Sun, Yushi and Sun, Kai and Xu, Yifan Ethan and Yang, Xiao and Dong, Xin Luna and Tang, Nan and Chen, Lei},
  journal={arXiv preprint arXiv:2509.04716},
  year={2025}
}
```

**Reference**: [KERAG: Knowledge-Enhanced Retrieval-Augmented Generation for Advanced Question Answering](https://arxiv.org/abs/2509.04716) - arXiv:2509.04716

## Acknowledgments

- Original KERAG paper authors
- LangChain community
- LangGraph developers
- Knowledge graph research community
