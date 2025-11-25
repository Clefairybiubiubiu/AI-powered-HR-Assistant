# Resume Matcher - Refactored Package

This is the refactored, modular version of the Resume-JD Matcher.

## Quick Start

```python
from resume_matcher.config import config
from resume_matcher.logging_config import setup_logging
from resume_matcher.matchers import ResumeJDMatcher

# Set up logging
setup_logging(level="INFO")

# Create matcher
matcher = ResumeJDMatcher("/path/to/data")
matcher.load_documents()
similarity_matrix = matcher.compute_similarity()
```

## Package Structure

```
resume_matcher/
├── config.py              # Configuration management
├── logging_config.py      # Logging setup
├── matchers/
│   ├── base_matcher.py   # Base class with shared methods
│   └── tfidf_matcher.py  # TF-IDF implementation
└── utils/
    ├── document_processor.py  # Document processing
    ├── exceptions.py          # Custom exceptions
    ├── path_validation.py     # Security utilities
    └── embedding_cache.py     # Improved caching
```

## Key Improvements

1. **Modular Structure**: Code organized into logical modules
2. **Proper Logging**: Replaced all print statements with logging
3. **Error Handling**: Comprehensive error handling with custom exceptions
4. **Security**: Path validation and file size limits
5. **Configuration**: Centralized configuration with environment variable support
6. **Code Reuse**: BaseMatcher eliminates code duplication

## Configuration

Configuration can be set via environment variables or code:

```python
import os
os.environ["RESUME_DATA_DIR"] = "/path/to/data"
os.environ["LOG_LEVEL"] = "DEBUG"
os.environ["MAX_FILE_SIZE_MB"] = "20"
```

Or in code:

```python
from resume_matcher.config import config
config.data_dir = Path("/path/to/data")
config.log_level = "DEBUG"
```

## Logging

Logging is automatically configured on import. To customize:

```python
from resume_matcher.logging_config import setup_logging

setup_logging(
    level="DEBUG",
    log_file=Path("custom.log"),
    format_string="%(levelname)s - %(message)s"
)
```

## Error Handling

All errors use custom exceptions:

```python
from resume_matcher.utils.exceptions import (
    DocumentProcessingError,
    PathValidationError,
    FileValidationError
)

try:
    matcher.load_documents()
except PathValidationError as e:
    print(f"Invalid path: {e}")
except DocumentProcessingError as e:
    print(f"Document processing failed: {e}")
```

## Security Features

- Path validation prevents directory traversal
- File size limits prevent memory issues
- File format validation ensures only supported formats

```python
from resume_matcher.utils.path_validation import (
    validate_directory,
    validate_file_for_processing
)

# Validate directory
safe_dir = validate_directory("/path/to/data")

# Validate file
safe_file = validate_file_for_processing(
    Path("resume.pdf"),
    allowed_dir=safe_dir
)
```

## Migration from Old Code

### Before:
```python
from resume_jd_matcher import ResumeJDMatcher
matcher = ResumeJDMatcher("/path/to/data")
```

### After:
```python
from resume_matcher.matchers import ResumeJDMatcher
matcher = ResumeJDMatcher("/path/to/data")
```

The API is mostly compatible, but now with better error handling and logging.

