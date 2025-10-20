"""
Configuration settings for HR Assistant
"""
import os
from typing import Optional
from pydantic_settings import BaseSettings  # pyright: ignore[reportMissingImports]


class Settings(BaseSettings):
    # API Settings
    api_title: str = "HR Assistant API"
    api_version: str = "1.0.0"
    api_description: str = "AI-powered HR Candidate Screening Assistant"
    
    # Server Settings
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = True
    
    # Database Settings (for future use)
    database_url: Optional[str] = None
    
    # ML Model Settings
    sentence_transformer_model: str = "all-MiniLM-L6-v2"
    spacy_model: str = "en_core_web_sm"
    
    # File Upload Settings
    max_file_size: int = 10 * 1024 * 1024  # 10MB
    allowed_file_types: list = [".pdf", ".docx", ".doc", ".txt"]
    
    # CORS Settings
    cors_origins: list = ["http://localhost:8501", "http://127.0.0.1:8501"]
    
    # Logging
    log_level: str = "INFO"
    
    class Config:
        env_file = ".env"
        case_sensitive = False


# Global settings instance
settings = Settings()
