"""
Configuration management for Resume-JD Matcher.
Centralizes all configuration constants and settings.
"""
import os
from pathlib import Path
from dataclasses import dataclass
from typing import Optional


@dataclass
class AppConfig:
    """Application configuration with defaults and environment variable support."""
    
    # Paths
    data_dir: Path = Path(os.getenv("RESUME_DATA_DIR", "./data"))
    log_file: Path = Path(os.getenv("LOG_FILE", "resume_matcher.log"))
    
    # File Processing
    max_file_size_mb: int = int(os.getenv("MAX_FILE_SIZE_MB", "10"))
    max_file_size: int = int(os.getenv("MAX_FILE_SIZE_MB", "10")) * 1024 * 1024  # Convert to bytes
    supported_formats: tuple = ('.txt', '.pdf', '.docx')
    
    # Text Processing
    max_name_length: int = 50
    min_text_length: int = 10
    max_text_length: int = 4000
    min_section_length: int = 10
    
    # Similarity Thresholds
    similarity_threshold_high: float = 0.7
    similarity_threshold_medium: float = 0.4
    skill_similarity_threshold: float = 0.8
    
    # TF-IDF Parameters
    tfidf_max_features: int = 1000
    tfidf_ngram_range: tuple = (1, 2)
    tfidf_min_df: int = 1
    tfidf_max_df: float = 0.8
    
    # Embedding Parameters
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_dim: int = 384  # all-MiniLM-L6-v2 dimension
    embedding_max_length: int = 4000
    embedding_batch_size: int = 32
    
    # Section Weights (defaults)
    default_education_weight: float = 0.1
    default_skills_weight: float = 0.4
    default_experience_weight: float = 0.4
    default_summary_weight: float = 0.2
    
    # Cache Settings
    embedding_cache_size: int = 1000
    text_cache_size: int = 128
    
    # Logging
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    
    @classmethod
    def from_env(cls) -> "AppConfig":
        """Create config from environment variables."""
        return cls(
            data_dir=Path(os.getenv("RESUME_DATA_DIR", "./data")),
            log_file=Path(os.getenv("LOG_FILE", "resume_matcher.log")),
            max_file_size_mb=int(os.getenv("MAX_FILE_SIZE_MB", "10")),
            log_level=os.getenv("LOG_LEVEL", "INFO"),
        )
    
    def validate(self) -> None:
        """Validate configuration values."""
        if self.max_file_size_mb <= 0:
            raise ValueError("max_file_size_mb must be positive")
        if not (0 <= self.similarity_threshold_high <= 1):
            raise ValueError("similarity_threshold_high must be between 0 and 1")
        if not (0 <= self.similarity_threshold_medium <= 1):
            raise ValueError("similarity_threshold_medium must be between 0 and 1")
        if self.embedding_batch_size <= 0:
            raise ValueError("embedding_batch_size must be positive")


# Global config instance
config = AppConfig.from_env()

