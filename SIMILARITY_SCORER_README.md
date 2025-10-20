# SimilarityScorer Class Documentation

A comprehensive similarity scoring system that integrates ResumeParser, SkillExtractor, and SentenceTransformer for advanced candidate-job matching.

## Features

- **Integrated Components**: Combines ResumeParser, SkillExtractor, and SentenceTransformer
- **Weighted Scoring Formula**: `score = α * skill_match + β * experience_alignment + γ * education_match`
- **Default Weights**: α=0.4, β=0.4, γ=0.2
- **Semantic Similarity**: Uses SentenceTransformer for advanced text comparison
- **Batch Processing**: Score multiple candidates against one job
- **Detailed Analysis**: Comprehensive breakdown of all scoring components

## Installation

### Prerequisites

```bash
pip install spacy sentence-transformers scikit-learn numpy nltk
python -m spacy download en_core_web_sm
```

### Dependencies

- `spacy`: Advanced NLP processing
- `sentence-transformers`: Semantic similarity
- `scikit-learn`: Cosine similarity calculations
- `numpy`: Numerical operations
- `nltk`: Stopwords and tokenization

## Usage

### Basic Usage

```python
from backend.models.similarity_scorer import SimilarityScorer

# Initialize scorer
scorer = SimilarityScorer()

# Compute fit score
result = scorer.compute_fit_score(resume_text, job_description)

# Access results
overall_score = result['overall_score']
skill_match = result['skill_match']
experience_alignment = result['experience_alignment']
education_match = result['education_match']
```

### Advanced Usage

```python
# Get detailed analysis
detailed_result = scorer.get_detailed_analysis(resume_text, job_description)

# Batch scoring
candidates = [
    {'name': 'John Doe', 'resume': '...'},
    {'name': 'Jane Smith', 'resume': '...'}
]
results = scorer.batch_score_candidates(candidates, job_description)
```

## API Reference

### SimilarityScorer Class

#### `__init__(sentence_transformer_model="all-MiniLM-L6-v2", spacy_model="en_core_web_sm")`

Initialize the similarity scorer.

**Parameters:**

- `sentence_transformer_model`: SentenceTransformer model name
- `spacy_model`: spaCy model name

#### `compute_fit_score(resume_text: str, job_description: str, job_requirements: Optional[str] = None) -> Dict[str, Any]`

Compute comprehensive fit score between resume and job description.

**Parameters:**

- `resume_text`: Raw resume text
- `job_description`: Job description text
- `job_requirements`: Optional separate requirements text

**Returns:**

- `overall_score`: Overall weighted score (0.0 - 1.0)
- `skill_match`: Technical skills matching score (0.0 - 1.0)
- `experience_alignment`: Experience level alignment (0.0 - 1.0)
- `education_match`: Education requirement match (0.0 - 1.0)
- `semantic_similarity`: Overall semantic similarity (0.0 - 1.0)
- `weights`: Scoring weights used
- `resume_analysis`: Detailed resume analysis
- `job_analysis`: Detailed job analysis
- `matching_details`: Matched, missing, and extra skills

#### `get_detailed_analysis(resume_text: str, job_description: str, job_requirements: Optional[str] = None) -> Dict[str, Any]`

Get comprehensive analysis including text statistics and named entities.

**Returns:**

- All results from `compute_fit_score`
- `resume_statistics`: Text statistics (word count, sentence count, etc.)
- `resume_entities`: Named entities found in resume
- `resume_keyword_frequency`: Keyword frequency analysis

#### `batch_score_candidates(candidates: List[Dict[str, str]], job_description: str, job_requirements: Optional[str] = None) -> List[Dict[str, Any]]`

Score multiple candidates against a job description.

**Parameters:**

- `candidates`: List of candidate dictionaries with 'name' and 'resume' keys
- `job_description`: Job description text
- `job_requirements`: Optional separate requirements text

**Returns:**

- List of scoring results sorted by overall score (descending)

## Scoring Formula

The similarity score is calculated using a weighted formula:

```
overall_score = α * skill_match + β * experience_alignment + γ * education_match
```

Where:

- **α = 0.4** (skill_match weight)
- **β = 0.4** (experience_alignment weight)
- **γ = 0.2** (education_match weight)

### Score Components

1. **Skill Match (0.0 - 1.0)**

   - Exact skill matches between resume and job requirements
   - Semantic similarity for related skills (threshold: 0.7)
   - Covers: programming languages, frameworks, databases, cloud platforms, tools

2. **Experience Alignment (0.0 - 1.0)**

   - Matches experience levels: entry, junior, mid, senior, lead, principal
   - Candidate can be higher level than required (no penalty)
   - Penalty for being underqualified

3. **Education Match (0.0 - 1.0)**

   - Matches education requirements: high school, bachelor, master, phd
   - Candidate can have higher education than required
   - Partial credit for lower education levels

4. **Semantic Similarity (0.0 - 1.0)**
   - Overall text similarity using SentenceTransformer embeddings
   - Provides additional context for the analysis

## Example Output

### Basic Fit Score

```python
{
    'overall_score': 0.854,
    'skill_match': 0.635,
    'experience_alignment': 1.000,
    'education_match': 1.000,
    'semantic_similarity': 0.651,
    'weights': {
        'alpha': 0.4,
        'beta': 0.4,
        'gamma': 0.2
    },
    'resume_analysis': {
        'skills_found': ['python', 'django', 'aws', 'docker', ...],
        'education_found': ['bachelor of science', 'master', ...],
        'experience_found': ['senior engineer', '5 years', ...],
        'skills_with_confidence': [...]
    },
    'job_analysis': {
        'skills_required': ['python', 'django', 'postgresql', ...],
        'skills_categories': {...}
    },
    'matching_details': {
        'matched_skills': ['python', 'django', 'aws'],
        'missing_skills': ['postgresql', 'kubernetes'],
        'extra_skills': ['react', 'javascript']
    }
}
```

### Batch Scoring Results

```python
[
    {
        'candidate_name': 'Alice Johnson',
        'overall_score': 0.908,
        'skill_match': 0.769,
        'experience_alignment': 1.000,
        'education_match': 1.000,
        # ... other fields
    },
    {
        'candidate_name': 'Bob Smith',
        'overall_score': 0.612,
        'skill_match': 0.231,
        'experience_alignment': 0.800,
        'education_match': 1.000,
        # ... other fields
    }
]
```

## Experience Level Mapping

The scorer recognizes these experience levels:

- **Entry**: entry, junior, 0-2 years, 1-3 years, fresh graduate
- **Mid**: mid, intermediate, 3-5 years, 4-6 years, 2-4 years
- **Senior**: senior, lead, principal, architect, 5+ years, 6+ years, 7+ years

## Education Level Mapping

The scorer recognizes these education levels:

- **High School**: high school, diploma, certificate
- **Associate**: associate, aa, as
- **Bachelor**: bachelor, bs, ba, bsc, undergraduate, college
- **Master**: master, ms, ma, msc, mba, graduate
- **PhD**: phd, doctorate, ph.d, doctoral

## Skill Categories

The scorer recognizes skills in these categories:

### Programming Languages

Python, Java, JavaScript, TypeScript, C++, C#, PHP, Ruby, Go, Rust, Swift, Kotlin, Scala, R, MATLAB, SQL, HTML, CSS

### Frameworks & Libraries

React, Angular, Vue, Node.js, Django, Flask, Spring, Express, Laravel, Rails, FastAPI

### Databases

MySQL, PostgreSQL, MongoDB, Redis, Elasticsearch, Cassandra, DynamoDB, Oracle, SQLite, MariaDB, Neo4j, Firebase

### Cloud Platforms

AWS, Azure, GCP, Docker, Kubernetes, Terraform, Jenkins, GitLab, GitHub Actions, CI/CD, Microservices, Serverless, Lambda, EC2, S3

### Data Science

pandas, numpy, scikit-learn, TensorFlow, PyTorch, Keras, Jupyter, matplotlib, seaborn, plotly, Tableau, Power BI, Spark, Hadoop

### Tools

Git, GitHub, GitLab, Jira, Confluence, Slack, Teams, Zoom, Figma, Sketch, Adobe, Photoshop, Illustrator, VSCode, PyCharm

## Customization

### Adjusting Weights

```python
scorer = SimilarityScorer()
scorer.alpha = 0.5  # Increase skill match weight
scorer.beta = 0.3   # Decrease experience weight
scorer.gamma = 0.2  # Keep education weight
```

### Adding New Skills

Extend the skill databases in the SkillExtractor class:

```python
# In SkillExtractor.__init__()
self.technical_skills['new_category'] = ['skill1', 'skill2', 'skill3']
```

### Custom Experience Levels

```python
scorer.experience_levels['expert'] = 4
scorer.experience_levels['consultant'] = 5
```

## Performance

- **Typical Processing Time**: 2-5 seconds per resume
- **Memory Usage**: ~300MB (includes all models)
- **Concurrent Requests**: Supports multiple simultaneous requests
- **Model Loading**: First request may take longer due to model initialization

## Error Handling

The scorer includes robust error handling:

- **Model Loading Errors**: Graceful fallback to basic processing
- **Empty Input**: Returns neutral scores
- **Malformed Text**: Handles parsing errors appropriately
- **Batch Processing**: Individual candidate errors don't affect others

## Testing

Run the test suite:

```bash
python test_similarity_scorer.py
```

Run the example:

```bash
python similarity_scorer_example.py
```

## Troubleshooting

### Common Issues

1. **spaCy Model Missing**:

   ```bash
   python -m spacy download en_core_web_sm
   ```

2. **SentenceTransformer Model Download**:
   The model will be downloaded automatically on first use.

3. **Memory Issues**: Ensure sufficient RAM for all models (~300MB)

4. **Import Errors**: Check that all dependencies are installed

### Performance Tips

- Use shorter text inputs for faster processing
- Consider caching results for repeated analysis
- Use batch processing for multiple candidates
- Monitor memory usage with large documents

## Integration with HR Assistant

The SimilarityScorer integrates seamlessly with the HR Assistant project:

- **ResumeParser**: Extracts structured information from resumes
- **SkillExtractor**: Identifies technical and soft skills
- **SentenceTransformer**: Provides semantic similarity
- **FastAPI Backend**: RESTful API endpoints
- **Streamlit Frontend**: User-friendly interface

## License

This SimilarityScorer is part of the HR Assistant project and follows the same licensing terms.
