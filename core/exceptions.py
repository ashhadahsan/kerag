"""
Custom exceptions for KERAG implementation.
"""


class KERAGException(Exception):
    """Base exception for KERAG errors."""

    pass


class KnowledgeBaseError(KERAGException):
    """Exception raised for knowledge base related errors."""

    pass


class SPARQLError(KnowledgeBaseError):
    """Exception raised for SPARQL query errors."""

    pass


class APIError(KnowledgeBaseError):
    """Exception raised for API-related errors."""

    pass


class EntityNotFoundError(KnowledgeBaseError):
    """Exception raised when an entity is not found."""

    pass


class PlanningError(KERAGException):
    """Exception raised during the planning phase."""

    pass


class RetrievalError(KERAGException):
    """Exception raised during the retrieval phase."""

    pass


class FilteringError(KERAGException):
    """Exception raised during the filtering phase."""

    pass


class SummarizationError(KERAGException):
    """Exception raised during the summarization phase."""

    pass


class ConfigurationError(KERAGException):
    """Exception raised for configuration related errors."""

    pass


class ValidationError(KERAGException):
    """Exception raised for input validation errors."""

    pass


class TimeoutError(KERAGException):
    """Exception raised when operations timeout."""

    pass
