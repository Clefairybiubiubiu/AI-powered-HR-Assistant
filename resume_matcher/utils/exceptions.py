"""
Custom exceptions for Resume-JD Matcher.
Provides specific exception types for better error handling.
"""


class ResumeMatcherError(Exception):
    """Base exception for all resume matcher errors."""
    pass


class DocumentProcessingError(ResumeMatcherError):
    """Exception raised when document processing fails."""
    pass


class FileValidationError(ResumeMatcherError):
    """Exception raised when file validation fails."""
    pass


class PathValidationError(ResumeMatcherError):
    """Exception raised when path validation fails."""
    pass


class EmbeddingError(ResumeMatcherError):
    """Exception raised when embedding generation fails."""
    pass


class SimilarityComputationError(ResumeMatcherError):
    """Exception raised when similarity computation fails."""
    pass

