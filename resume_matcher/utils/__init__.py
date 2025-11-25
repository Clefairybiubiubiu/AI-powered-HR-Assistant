"""Utility modules for resume-JD matcher."""
from .document_processor import DocumentProcessor
from .exceptions import (
    DocumentProcessingError,
    EmbeddingError,
    FileValidationError,
    PathValidationError,
    ResumeMatcherError,
    SimilarityComputationError,
)
from .path_validation import (
    check_file_size,
    validate_directory,
    validate_file_for_processing,
    validate_file_path,
    validate_path,
)

# LLM client (optional)
try:
    from .llm_client import LLMClient, get_llm_client, is_llm_available
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False

# Skills database (optional)
try:
    from .skills_database import (
        extract_skills_from_text,
        categorize_skills,
        ALL_TECHNICAL_SKILLS,
        ALL_SKILLS,
        SKILLS_DICT,
        SKILLS_BY_CATEGORY
    )
    SKILLS_DATABASE_AVAILABLE = True
except ImportError:
    SKILLS_DATABASE_AVAILABLE = False

# Build __all__ list
__all__ = [
    'DocumentProcessor',
    'ResumeMatcherError',
    'DocumentProcessingError',
    'FileValidationError',
    'PathValidationError',
    'EmbeddingError',
    'SimilarityComputationError',
    'validate_path',
    'validate_directory',
    'validate_file_path',
    'validate_file_for_processing',
    'check_file_size',
]

if LLM_AVAILABLE:
    __all__.extend(['LLMClient', 'get_llm_client', 'is_llm_available'])

if SKILLS_DATABASE_AVAILABLE:
    __all__.extend([
        'extract_skills_from_text',
        'categorize_skills',
        'ALL_TECHNICAL_SKILLS',
        'ALL_SKILLS',
        'SKILLS_DICT',
        'SKILLS_BY_CATEGORY'
    ])

