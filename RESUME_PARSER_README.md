# ResumeParser Class Documentation

A sophisticated resume parsing class that uses spaCy for advanced NLP processing and keyword extraction.

## Features

- **spaCy Integration**: Uses `en_core_web_sm` model for tokenization and lemmatization
- **Stopwords Removal**: Automatically removes common stopwords and punctuation
- **Keyword Extraction**: Extracts skills, education, and experience keywords
- **Named Entity Recognition**: Identifies persons, organizations, dates, and locations
- **Text Statistics**: Provides comprehensive text analysis
- **Fallback Support**: Works even without spaCy (basic keyword matching)

## Installation

### Prerequisites

```bash
pip install spacy nltk
python -m spacy download en_core_web_sm
```

### Dependencies

- `spacy`: Advanced NLP processing
- `nltk`: Stopwords and tokenization
- `re`: Regular expressions for text processing

## Usage

### Basic Usage

```python
from backend.models.document_parser import ResumeParser

# Initialize parser
parser = ResumeParser()

# Parse resume text
resume_text = "Your resume text here..."
result = parser.parse_resume(resume_text)

# Access results
skills = result['skills']
education = result['education']
experience = result['experience']
```

### Advanced Usage

```python
# Get text statistics
stats = parser.get_text_statistics(resume_text)
print(f"Word count: {stats['word_count']}")
print(f"Unique words: {stats['unique_words']}")

# Extract named entities
entities = parser.extract_named_entities(resume_text)
for entity, label in entities:
    print(f"{entity} ({label})")

# Get keyword frequencies
freq = parser.get_keyword_frequency(resume_text)
for keyword, count in freq.items():
    print(f"{keyword}: {count}")
```

## API Reference

### ResumeParser Class

#### `__init__(model_name="en_core_web_sm")`

Initialize the parser with a spaCy model.

**Parameters:**

- `model_name` (str): spaCy model name (default: "en_core_web_sm")

#### `parse_resume(text: str) -> Dict[str, List[str]]`

Parse resume text and extract keywords.

**Parameters:**

- `text` (str): Raw resume text

**Returns:**

- `Dict[str, List[str]]`: Dictionary with keys 'skills', 'education', 'experience'

**Example:**

```python
result = parser.parse_resume(resume_text)
skills = result['skills']  # ['python', 'django', 'aws', ...]
education = result['education']  # ['bachelor of science', 'master', ...]
experience = result['experience']  # ['senior engineer', '5 years', ...]
```

#### `get_text_statistics(text: str) -> Dict[str, Any]`

Get comprehensive text statistics.

**Returns:**

- `word_count`: Number of words
- `sentence_count`: Number of sentences
- `character_count`: Number of characters
- `average_word_length`: Average word length
- `unique_words`: Number of unique words
- `pos_tags`: Part-of-speech tag distribution
- `named_entities`: Number of named entities

#### `extract_named_entities(text: str) -> List[Tuple[str, str]]`

Extract named entities from text.

**Returns:**

- `List[Tuple[str, str]]`: List of (entity, label) tuples

**Entity Labels:**

- `PERSON`: Person names
- `ORG`: Organizations
- `GPE`: Geopolitical entities
- `DATE`: Dates
- `CARDINAL`: Numbers

#### `get_keyword_frequency(text: str) -> Dict[str, int]`

Get frequency of technical keywords.

**Returns:**

- `Dict[str, int]`: Keyword frequency dictionary

## Skill Categories

The parser recognizes skills in these categories:

### Programming Languages

- Python, Java, JavaScript, TypeScript, C++, C#, PHP, Ruby, Go, Rust, Swift, Kotlin, Scala, R, MATLAB, SQL, HTML, CSS

### Frameworks & Libraries

- React, Angular, Vue, Node.js, Django, Flask, Spring, Express, Laravel, Rails, FastAPI

### Databases

- MySQL, PostgreSQL, MongoDB, Redis, Elasticsearch, Cassandra, DynamoDB, Oracle, SQLite, MariaDB, Neo4j, Firebase

### Cloud Platforms

- AWS, Azure, GCP, Docker, Kubernetes, Terraform, Jenkins, GitLab, GitHub Actions, CI/CD, Microservices, Serverless, Lambda, EC2, S3

### Data Science

- pandas, numpy, scikit-learn, TensorFlow, PyTorch, Keras, Jupyter, matplotlib, seaborn, plotly, Tableau, Power BI, Spark, Hadoop

### Tools

- Git, GitHub, GitLab, Jira, Confluence, Slack, Teams, Zoom, Figma, Sketch, Adobe, Photoshop, Illustrator, VSCode, PyCharm

## Education Keywords

### Degree Types

- Bachelor, Master, PhD, Doctorate, Associate, Diploma, Certificate

### Fields of Study

- Computer Science, Engineering, Mathematics, Statistics, Data Science, Business, Economics, Physics, Chemistry, Biology

### Institution Types

- University, College, Institute, School

## Experience Keywords

### Position Types

- Engineer, Developer, Analyst, Manager, Director, Lead, Senior, Junior

### Company Indicators

- Inc, Corp, LLC, Ltd, Company, Technologies, Solutions

### Time Indicators

- Years, Months, Experience, Worked, Employed

## Example Output

```python
{
    'skills': [
        'python', 'django', 'aws', 'docker', 'kubernetes',
        'postgresql', 'mongodb', 'react', 'javascript', 'git'
    ],
    'education': [
        'bachelor of science in computer science',
        'master of science in software engineering',
        'university of california',
        'stanford university'
    ],
    'experience': [
        'senior software engineer',
        '5 years of experience',
        'techcorp inc',
        'leading development teams'
    ]
}
```

## Text Statistics Example

```python
{
    'word_count': 240,
    'sentence_count': 5,
    'character_count': 1250,
    'average_word_length': 5.2,
    'unique_words': 111,
    'pos_tags': Counter({'PROPN': 98, 'PUNCT': 52, 'NOUN': 37, ...}),
    'named_entities': 41
}
```

## Named Entities Example

```python
[
    ('John Smith', 'PERSON'),
    ('TechCorp Inc.', 'ORG'),
    ('University of California', 'ORG'),
    ('2020', 'DATE'),
    ('AWS', 'ORG')
]
```

## Error Handling

The parser includes robust error handling:

- **spaCy Model Missing**: Falls back to basic keyword matching
- **Empty Text**: Returns empty results gracefully
- **Invalid Input**: Handles malformed text appropriately

## Performance

- **Typical Processing Time**: 0.5-2 seconds per resume
- **Memory Usage**: ~200MB (includes spaCy model)
- **Text Length**: Handles resumes up to 10,000 words efficiently

## Customization

### Adding New Skills

```python
# Extend the technical_skills dictionary
parser.technical_skills['new_category'] = ['skill1', 'skill2', 'skill3']
```

### Modifying Education Keywords

```python
# Add new education keywords
parser.education_keywords['degrees'].append('certificate')
parser.education_keywords['fields'].append('artificial intelligence')
```

### Adjusting Experience Keywords

```python
# Add new position types
parser.experience_keywords['positions'].append('architect')
parser.experience_keywords['companies'].append('startup')
```

## Testing

Run the test suite:

```bash
python test_resume_parser.py
```

Run the example:

```bash
python resume_parser_example.py
```

## Troubleshooting

### Common Issues

1. **spaCy Model Not Found**:

   ```bash
   python -m spacy download en_core_web_sm
   ```

2. **NLTK Data Missing**:

   ```bash
   python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
   ```

3. **Memory Issues**: Ensure sufficient RAM for the spaCy model

4. **Import Errors**: Check that all dependencies are installed

### Performance Tips

- Use shorter text inputs for faster processing
- Consider caching results for repeated parsing
- Monitor memory usage with large documents
- Use batch processing for multiple resumes

## License

This ResumeParser is part of the HR Assistant project and follows the same licensing terms.
