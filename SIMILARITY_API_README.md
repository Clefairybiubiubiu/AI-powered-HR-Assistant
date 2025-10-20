# Resume-Job Similarity API

A focused FastAPI application that calculates similarity scores between resumes and job descriptions using Sentence-BERT embeddings and weighted scoring.

## Features

- **POST /similarity**: Calculate similarity between resume and job description
- **Sentence-BERT Embeddings**: Uses 'all-MiniLM-L6-v2' model for semantic similarity
- **Weighted Scoring Formula**: `score = α * skill_match + β * experience_alignment + γ * education_match`
- **Default Weights**: α=0.4, β=0.4, γ=0.2
- **Comprehensive Analysis**: Skills matching, experience alignment, education matching

## Quick Start

### 1. Install Dependencies

```bash
pip install -r similarity_requirements.txt
```

### 2. Start the API Server

```bash
python run_similarity_app.py
# OR
python similarity_app.py
# OR
uvicorn similarity_app:app --reload --host 0.0.0.0 --port 8000
```

### 3. Access the API

- **API Server**: http://localhost:8000
- **Interactive Docs**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health

## API Usage

### POST /similarity

**Request Body:**

```json
{
  "resume": "John Smith\nSenior Software Engineer\n...",
  "job_desc": "We are looking for a Senior Python Developer..."
}
```

**Response:**

```json
{
  "similarity_score": 0.85,
  "skill_match": 0.9,
  "experience_alignment": 0.8,
  "education_match": 0.75,
  "details": {
    "semantic_similarity": 0.82,
    "resume_skills": ["python", "django", "aws", "docker"],
    "job_skills": ["python", "django", "postgresql", "kubernetes"],
    "matched_skills": ["python", "django"],
    "missing_skills": ["postgresql", "kubernetes"],
    "weights": {
      "alpha": 0.4,
      "beta": 0.4,
      "gamma": 0.2
    }
  }
}
```

## Scoring Formula

The similarity score is calculated using a weighted formula:

```
similarity_score = α * skill_match + β * experience_alignment + γ * education_match
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

   - Matches experience levels: entry, mid, senior
   - Candidate can be higher level than required (no penalty)
   - Penalty for being underqualified

3. **Education Match (0.0 - 1.0)**

   - Matches education requirements: high school, bachelor, master, phd
   - Candidate can have higher education than required
   - Partial credit for lower education levels

4. **Semantic Similarity (0.0 - 1.0)**
   - Overall text similarity using Sentence-BERT embeddings
   - Provides additional context for the analysis

## Testing

### Test the API

```bash
python test_similarity.py
```

### Manual Testing with curl

```bash
curl -X POST "http://localhost:8000/similarity" \
     -H "Content-Type: application/json" \
     -d '{
       "resume": "John Smith\nSenior Software Engineer\nPython, Django, AWS experience...",
       "job_desc": "We need a Senior Python Developer with Django and AWS experience..."
     }'
```

### Python Client Example

```python
import requests

response = requests.post(
    "http://localhost:8000/similarity",
    json={
        "resume": "Your resume text here...",
        "job_desc": "Job description here..."
    }
)

result = response.json()
print(f"Similarity Score: {result['similarity_score']}")
```

## Technical Details

### Dependencies

- **FastAPI**: Web framework
- **Sentence-Transformers**: Embeddings and similarity
- **scikit-learn**: Cosine similarity calculations
- **NumPy**: Numerical operations
- **Pydantic**: Data validation

### Model Information

- **Embedding Model**: `all-MiniLM-L6-v2`
- **Model Size**: ~80MB (downloaded automatically)
- **Performance**: Fast inference, good quality embeddings
- **Languages**: Optimized for English text

### Skill Categories

The API recognizes skills in these categories:

- **Programming**: Python, Java, JavaScript, TypeScript, etc.
- **Databases**: MySQL, PostgreSQL, MongoDB, Redis, etc.
- **Cloud**: AWS, Azure, GCP, Docker, Kubernetes, etc.
- **Data Science**: pandas, numpy, scikit-learn, TensorFlow, etc.

## Configuration

### Adjusting Weights

Modify the weights in `similarity_app.py`:

```python
ALPHA = 0.4  # skill_match weight
BETA = 0.4   # experience_alignment weight
GAMMA = 0.2  # education_match weight
```

### Adding New Skills

Extend the `technical_skills` dictionary in the `SimilarityCalculator` class:

```python
self.technical_skills = {
    'new_category': ['skill1', 'skill2', 'skill3'],
    # ... existing categories
}
```

## Performance

- **Typical Response Time**: 1-3 seconds
- **Memory Usage**: ~200MB (includes model)
- **Concurrent Requests**: Supports multiple simultaneous requests
- **Model Loading**: First request may take longer due to model initialization

## Error Handling

The API handles various error conditions:

- **400 Bad Request**: Missing or invalid input
- **500 Internal Server Error**: Processing errors
- **Connection Errors**: Model loading issues

## Example Use Cases

1. **Resume Screening**: Automatically score candidate resumes against job postings
2. **Job Matching**: Find the best candidates for a position
3. **Skill Gap Analysis**: Identify missing skills in candidate profiles
4. **Recruitment Analytics**: Analyze hiring patterns and requirements

## Troubleshooting

### Common Issues

1. **Model Download**: First run downloads the Sentence-BERT model (~80MB)
2. **Memory Issues**: Ensure sufficient RAM for the model
3. **Port Conflicts**: Change port if 8000 is in use
4. **Import Errors**: Install all dependencies from requirements.txt

### Performance Tips

- Use shorter text inputs for faster processing
- Consider caching results for repeated requests
- Monitor memory usage with large documents
- Use batch processing for multiple comparisons

## API Documentation

Once the server is running, visit:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

These provide interactive documentation where you can test the API directly in your browser.
