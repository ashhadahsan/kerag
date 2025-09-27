"""
Configuration settings for KERAG implementation.
"""

import os
from typing import Dict, Any, Optional
from pydantic import Field
from pydantic_settings import BaseSettings
from core.types import KnowledgeBaseType, DomainType


class Settings(BaseSettings):
    """Global settings for KERAG."""

    # LLM Configuration
    llm_model: str = Field(default="gpt-4", env="KERAG_LLM_MODEL")
    openai_api_key: Optional[str] = Field(default=None, env="OPENAI_API_KEY")
    anthropic_api_key: Optional[str] = Field(default=None, env="ANTHROPIC_API_KEY")
    google_api_key: Optional[str] = Field(default=None, env="GOOGLE_API_KEY")
    xai_api_key: Optional[str] = Field(default=None, env="XAI_API_KEY")

    # Knowledge Base Configuration
    default_kb_type: KnowledgeBaseType = Field(default=KnowledgeBaseType.SPARQL)
    sparql_endpoint: Optional[str] = Field(default=None, env="SPARQL_ENDPOINT")
    api_base_url: Optional[str] = Field(default=None, env="API_BASE_URL")
    api_key: Optional[str] = Field(default=None, env="API_KEY")

    # Retrieval Configuration
    max_hops: int = Field(default=3, env="KERAG_MAX_HOPS")
    max_entities: int = Field(default=1000, env="KERAG_MAX_ENTITIES")
    relevance_threshold: float = Field(default=0.5, env="KERAG_RELEVANCE_THRESHOLD")

    # Performance Configuration
    timeout: int = Field(default=30, env="KERAG_TIMEOUT")
    max_retries: int = Field(default=3, env="KERAG_MAX_RETRIES")
    batch_size: int = Field(default=10, env="KERAG_BATCH_SIZE")

    # Debug Configuration
    debug: bool = Field(default=False, env="KERAG_DEBUG")
    log_level: str = Field(default="INFO", env="KERAG_LOG_LEVEL")

    class Config:
        env_file = ".env"
        case_sensitive = False


# Global settings instance
settings = Settings()


class KnowledgeBaseConfig:
    """Configuration for different knowledge base types."""

    SPARQL_CONFIG = {
        "default_endpoint": "http://dbpedia.org/sparql",
        "timeout": 30,
        "default_graph": None,
        "format": "json",
        "user_agent": "KERAG/1.0",
    }

    API_CONFIG = {
        "timeout": 30,
        "retry_attempts": 3,
        "rate_limit": 100,  # requests per minute
        "headers": {"Content-Type": "application/json", "User-Agent": "KERAG/1.0"},
    }


class DomainConfig:
    """Configuration for different domains."""

    DOMAIN_SCHEMAS = {
        DomainType.MOVIE: {
            "common_relations": [
                "actor",
                "director",
                "genre",
                "release_date",
                "box_office",
            ],
            "entity_types": ["Movie", "Person", "Genre", "Studio"],
            "attributes": ["title", "year", "rating", "duration"],
        },
        DomainType.MUSIC: {
            "common_relations": ["artist", "album", "genre", "release_date", "label"],
            "entity_types": ["Song", "Artist", "Album", "Genre", "Label"],
            "attributes": ["title", "year", "duration", "track_number"],
        },
        DomainType.SPORTS: {
            "common_relations": ["player", "team", "league", "position", "statistics"],
            "entity_types": ["Player", "Team", "League", "Game", "Season"],
            "attributes": ["name", "position", "statistics", "achievements"],
        },
        DomainType.FINANCE: {
            "common_relations": ["company", "stock", "revenue", "market_cap", "sector"],
            "entity_types": ["Company", "Stock", "Sector", "FinancialReport"],
            "attributes": ["symbol", "price", "market_cap", "revenue"],
        },
        DomainType.BOOK: {
            "common_relations": [
                "author",
                "publisher",
                "genre",
                "publication_date",
                "series",
            ],
            "entity_types": ["Book", "Author", "Publisher", "Genre", "Series"],
            "attributes": ["title", "isbn", "pages", "language", "edition"],
        },
    }


class LLMConfig:
    """Configuration for different LLM providers."""

    OPENAI_CONFIG = {
        "model": "gpt-4",
        "temperature": 0.1,
        "max_tokens": 4000,
        "top_p": 1.0,
        "frequency_penalty": 0.0,
        "presence_penalty": 0.0,
    }

    ANTHROPIC_CONFIG = {
        "model": "claude-3-opus-20240229",
        "temperature": 0.1,
        "max_tokens": 4000,
        "top_p": 1.0,
    }

    COHERE_CONFIG = {
        "model": "command",
        "temperature": 0.1,
        "max_tokens": 4000,
        "k": 0,
        "p": 0.75,
    }

    GOOGLE_CONFIG = {
        "model": "gemini-1.5-pro",
        "temperature": 0.1,
        "max_tokens": 4000,
        "top_p": 0.95,
        "top_k": 40,
    }

    XAI_CONFIG = {
        "model": "grok-beta",
        "temperature": 0.1,
        "max_tokens": 4000,
        "top_p": 1.0,
    }


class PromptConfig:
    """Configuration for prompt templates."""

    PLANNING_PROMPTS = {
        "entity_identification": """
        Given the following question, identify the main topic entity and its type.
        
        Question: {question}
        
        Please provide:
        1. Entity name
        2. Entity type
        3. Confidence score (0-1)
        
        Format your response as JSON:
        {{
            "entity_name": "...",
            "entity_type": "...",
            "confidence": 0.95
        }}
        """,
        "domain_detection": """
        Given the following question and topic entity, determine the domain.
        
        Question: {question}
        Topic Entity: {topic_entity}
        
        Choose from: movie, music, sports, finance, book, person, organization, location, general
        
        Format your response as JSON:
        {{
            "domain": "...",
            "confidence": 0.95,
            "reasoning": "..."
        }}
        """,
        "retrieval_planning": """
        Based on the question, topic entity, and domain, create a retrieval plan.
        
        Question: {question}
        Topic Entity: {topic_entity}
        Domain: {domain}
        
        Provide:
        1. Relevant relations to explore
        2. Maximum number of hops
        3. Filters to apply
        
        Format your response as JSON:
        {{
            "relations": ["relation1", "relation2"],
            "max_hops": 2,
            "filters": {{"type": "specific"}},
            "reasoning": "..."
        }}
        """,
    }

    FILTERING_PROMPTS = {
        "relevance_scoring": """
        Given the question and retrieved knowledge, score the relevance of each entity and triple.
        
        Question: {question}
        Retrieved Knowledge: {knowledge}
        
        For each entity and triple, provide a relevance score (0-1) where:
        - 1.0 = directly relevant to answering the question
        - 0.5 = somewhat relevant
        - 0.0 = not relevant
        
        Format your response as JSON:
        {{
            "entity_scores": {{"entity_id": 0.95, ...}},
            "triple_scores": {{"triple_id": 0.85, ...}},
            "reasoning": "..."
        }}
        """
    }

    SUMMARIZATION_PROMPTS = {
        "cot_reasoning": """
        Answer the following question using Chain-of-Thought reasoning based on the provided knowledge.
        
        Question: {question}
        
        Knowledge:
        {knowledge}
        
        Please:
        1. Identify the key information needed to answer the question
        2. Trace through the logical steps to reach the answer
        3. Provide a clear and concise final answer
        
        Format your response as:
        Reasoning: [Your step-by-step reasoning]
        Answer: [Your final answer]
        """,
        "confidence_scoring": """
        Given the question, answer, and supporting knowledge, assess the confidence in the answer.
        
        Question: {question}
        Answer: {answer}
        Supporting Knowledge: {knowledge}
        
        Consider:
        1. How well the knowledge supports the answer
        2. Any gaps or uncertainties
        3. The reliability of the information sources
        
        Provide a confidence score (0-1) and reasoning.
        
        Format your response as JSON:
        {{
            "confidence": 0.95,
            "reasoning": "...",
            "uncertainties": ["..."],
            "supporting_evidence": ["..."]
        }}
        """,
    }
