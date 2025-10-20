# AI-Powered HR Candidate Screening Assistant

An intelligent system for automated resume screening and candidate-job fit analysis using Generative AI and semantic similarity.

## Features

- **Document Parsing**: Extract text from PDF and DOCX resumes
- **Skill Extraction**: NLP pipeline for identifying skills and qualifications
- **Semantic Similarity**: Sentence-BERT for embedding and similarity scoring
- **FastAPI Backend**: RESTful API for processing and scoring
- **Streamlit Frontend**: User-friendly interface for recruiters
- **Modular Design**: Clean separation of concerns

## Project Structure

```
hr-assistant/
├── backend/                     # FastAPI Backend
│   ├── __init__.py
│   ├── main.py                  # FastAPI application entry point
│   ├── api/                     # API routes and endpoints
│   │   ├── __init__.py
│   │   ├── candidates.py        # Candidate-related endpoints
│   │   ├── jobs.py              # Job-related endpoints
│   │   └── scoring.py           # Scoring endpoints
│   ├── models/                  # Data models
│   │   ├── __init__.py
│   │   ├── candidate.py         # Candidate data models
│   │   └── job.py               # Job description models
│   ├── services/                # Business logic services
│   │   ├── __init__.py
│   │   ├── document_parser.py   # Document parsing service
│   │   ├── skill_extractor.py  # Skill extraction service
│   │   └── similarity_scorer.py # Similarity scoring service
│   ├── core/                    # Core backend functionality
│   │   └── __init__.py
│   ├── middleware/              # Backend middleware
│   │   └── __init__.py
│   └── utils/                   # Backend utilities
│       └── __init__.py
├── frontend/                    # Streamlit Frontend
│   ├── app.py                   # Main Streamlit application
│   ├── pages/                   # Streamlit pages
│   │   └── __init__.py
│   ├── components/              # Reusable components
│   │   └── __init__.py
│   ├── assets/                  # Frontend assets
│   │   └── __init__.py
│   └── utils/                   # Frontend utilities
│       └── __init__.py
├── shared/                      # Shared utilities
│   ├── __init__.py
│   ├── config/                  # Shared configuration
│   │   ├── __init__.py
│   │   └── settings.py          # Application settings
│   ├── models/                  # Shared data models
│   │   └── __init__.py
│   ├── utils/                   # Shared utilities
│   │   └── __init__.py
│   ├── embeddings/              # Embedding utilities
│   │   └── __init__.py
│   ├── parsers/                 # Document parsing utilities
│   │   └── __init__.py
│   └── scorers/                 # Scoring utilities
│       └── __init__.py
├── data/                        # Sample data
│   ├── sample_jobs.json         # Sample job descriptions
│   └── sample_resumes/          # Sample resume files
│       ├── sample_resume_1.txt
│       └── sample_resume_2.txt
├── tests/                       # Test suite
│   ├── __init__.py
│   ├── unit/                    # Unit tests
│   │   └── __init__.py
│   └── integration/             # Integration tests
│       └── __init__.py
├── docs/                        # Documentation
├── config/                      # Configuration files
├── logs/                        # Log files
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

## Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd hr-assistant
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download required models**
   ```bash
   python -m spacy download en_core_web_sm
   ```

## Usage

### Starting the Application

1. **Start the FastAPI backend** (Terminal 1):
   ```bash
   cd backend
   uvicorn main:app --reload --host 0.0.0.0 --port 8000
   ```

2. **Start the Streamlit frontend** (Terminal 2):
   ```bash
   cd frontend
   streamlit run app.py
   ```

3. **Access the application**:
   - Frontend: http://localhost:8501
   - Backend API: http://localhost:8000
   - API Documentation: http://localhost:8000/docs

### Using the System

1. **Upload Resume**: Use the Streamlit interface to upload PDF/DOCX resume files
2. **Analyze Job**: Enter job description and requirements
3. **Calculate Fit**: Get detailed scoring and recommendations
4. **View Results**: See comprehensive match analysis

## API Endpoints

### Candidate Endpoints
- `POST /api/v1/candidates/parse` - Parse uploaded resume
- `GET /api/v1/candidates/` - Get all candidates
- `GET /api/v1/candidates/{candidate_id}` - Get specific candidate

### Job Endpoints
- `POST /api/v1/jobs/analyze` - Analyze job description
- `GET /api/v1/jobs/` - Get all jobs
- `GET /api/v1/jobs/{job_id}` - Get specific job

### Scoring Endpoints
- `POST /api/v1/scoring/calculate-fit` - Calculate fit score
- `POST /api/v1/scoring/batch-scoring` - Batch scoring
- `GET /api/v1/scoring/scores/{job_id}` - Get scores for job

### Health Check
- `GET /health` - API health status

## Key Features

### Document Parsing
- Supports PDF, DOCX, DOC, and TXT formats
- Extracts structured information (name, email, phone, experience)
- Handles various resume formats and layouts

### Skill Extraction
- Uses spaCy NLP for advanced text processing
- Identifies technical and soft skills
- Provides confidence scores for each skill
- Categorizes skills by type (programming, databases, cloud, etc.)

### Similarity Scoring
- Sentence-BERT embeddings for semantic similarity
- Multiple scoring dimensions:
  - Text similarity
  - Skills matching
  - Experience level matching
  - Education matching
- Weighted overall score calculation

### Frontend Interface
- Clean, intuitive Streamlit interface
- File upload with drag-and-drop support
- Real-time processing feedback
- Comprehensive results visualization

## Development

### Project Structure Benefits
- **Modular Design**: Clean separation between backend, frontend, and shared utilities
- **Scalable Architecture**: Easy to add new features and services
- **Testable Code**: Separate test suites for unit and integration testing
- **Configuration Management**: Centralized settings and environment management

### Adding New Features
1. **New API Endpoints**: Add to `backend/api/`
2. **New Services**: Add to `backend/services/`
3. **New Models**: Add to `backend/models/` or `shared/models/`
4. **Frontend Pages**: Add to `frontend/pages/`
5. **Tests**: Add to `tests/unit/` or `tests/integration/`

## Configuration

The application uses environment variables for configuration. Key settings:

- `API_TITLE`: API title
- `DEBUG`: Debug mode
- `HOST`: Server host
- `PORT`: Server port
- `SENTENCE_TRANSFORMER_MODEL`: ML model name
- `SPACY_MODEL`: spaCy model name
- `MAX_FILE_SIZE`: Maximum file upload size
- `ALLOWED_FILE_TYPES`: Allowed file extensions

## Troubleshooting

### Common Issues

1. **spaCy model not found**:
   ```bash
   python -m spacy download en_core_web_sm
   ```

2. **Sentence-BERT model download**:
   The model will be downloaded automatically on first use.

3. **Port conflicts**:
   Change ports in the startup commands if 8000 or 8501 are in use.

4. **File upload issues**:
   Check file size limits and supported formats.

### Performance Tips

- Use GPU acceleration for Sentence-BERT if available
- Consider caching embeddings for repeated analysis
- Use batch processing for multiple candidates
- Monitor memory usage with large files

