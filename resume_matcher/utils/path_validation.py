"""
Path validation utilities for security and safety.
Prevents path traversal attacks and validates file operations.
"""
from pathlib import Path
from typing import Union

from ..config import config
from .exceptions import PathValidationError, FileValidationError


def validate_path(path: Union[str, Path], must_exist: bool = True) -> Path:
    """
    Validate and normalize file path.
    
    Args:
        path: Path to validate
        must_exist: Whether the path must exist
    
    Returns:
        Resolved Path object
    
    Raises:
        PathValidationError: If path is invalid
    """
    try:
        path = Path(path).resolve()
    except (OSError, ValueError) as e:
        raise PathValidationError(f"Invalid path: {path}") from e
    
    # Prevent path traversal
    if '..' in str(path):
        raise PathValidationError(f"Path traversal detected: {path}")
    
    # Check if path exists
    if must_exist and not path.exists():
        raise PathValidationError(f"Path does not exist: {path}")
    
    return path


def validate_directory(path: Union[str, Path]) -> Path:
    """
    Validate that path is a directory.
    
    Args:
        path: Path to validate
    
    Returns:
        Resolved Path object
    
    Raises:
        PathValidationError: If path is not a directory
    """
    path = validate_path(path, must_exist=True)
    
    if not path.is_dir():
        raise PathValidationError(f"Path is not a directory: {path}")
    
    return path


def validate_file_path(file_path: Path, allowed_dir: Path) -> Path:
    """
    Validate that file is within allowed directory.
    
    Args:
        file_path: File path to validate
        allowed_dir: Directory that file must be within
    
    Returns:
        Resolved Path object
    
    Raises:
        PathValidationError: If file is outside allowed directory
    """
    resolved_file = validate_path(file_path, must_exist=False)
    resolved_dir = validate_directory(allowed_dir)
    
    try:
        resolved_file.relative_to(resolved_dir)
    except ValueError:
        raise PathValidationError(
            f"File {file_path} is outside allowed directory {allowed_dir}"
        )
    
    return resolved_file


def check_file_size(file_path: Path) -> None:
    """
    Check if file size is within limits.
    
    Args:
        file_path: File to check
    
    Raises:
        FileValidationError: If file exceeds maximum size
    """
    if not file_path.exists():
        return
    
    size = file_path.stat().st_size
    if size > config.max_file_size:
        raise FileValidationError(
            f"File {file_path} exceeds maximum size of {config.max_file_size / (1024*1024):.1f} MB "
            f"(actual: {size / (1024*1024):.1f} MB)"
        )


def validate_file_for_processing(file_path: Path, allowed_dir: Path) -> Path:
    """
    Complete validation for file processing.
    
    Args:
        file_path: File to validate
        allowed_dir: Directory that file must be within
    
    Returns:
        Resolved Path object
    
    Raises:
        PathValidationError: If path validation fails
        FileValidationError: If file validation fails
    """
    # Validate path
    validated_path = validate_file_path(file_path, allowed_dir)
    
    # Check file size
    check_file_size(validated_path)
    
    # Check file extension
    if validated_path.suffix.lower() not in config.supported_formats:
        raise FileValidationError(
            f"Unsupported file format: {validated_path.suffix}. "
            f"Supported formats: {', '.join(config.supported_formats)}"
        )
    
    return validated_path

