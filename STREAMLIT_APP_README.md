# HR Assistant Streamlit Frontend

A user-friendly Streamlit web application that provides an intuitive interface for resume-job similarity analysis using AI-powered matching.

## Features

- **File Upload Support**: Upload PDF, DOCX, DOC, and TXT files directly
- **Dual Input Methods**: Choose between file upload or text paste
- **Automatic Text Extraction**: Parse and extract text from uploaded documents
- **Real-time Analysis**: Instant similarity scoring with progress indicators
- **Comprehensive Results**: Detailed breakdown of skill match, experience alignment, and education match
- **Progress Visualization**: Progress bars and metrics for easy understanding
- **API Integration**: Seamless connection to FastAPI backend
- **Error Handling**: Robust error handling and user feedback

## Installation

### Prerequisites

```bash
pip install streamlit requests PyPDF2 python-docx
```

### Dependencies

- `streamlit`: Web application framework
- `requests`: HTTP client for API communication
- `PyPDF2`: PDF file parsing
- `python-docx`: DOCX file parsing

## Usage

### Quick Start

1. **Start the FastAPI Backend** (Terminal 1):

   ```bash
   python run_similarity_app.py
   # OR
   python similarity_app.py
   ```

2. **Start the Streamlit Frontend** (Terminal 2):

   ```bash
   python run_streamlit_app.py
   # OR
   cd frontend && streamlit run app.py
   ```

3. **Access the Application**:
   - Open http://localhost:8501 in your browser
   - Choose input method: "Upload Files" or "Paste Text"
   - **Option A - Upload Files**: Upload PDF/DOCX files for resume and job description
   - **Option B - Paste Text**: Paste text directly in the text areas
   - Click "Analyze Compatibility"

### Complete Demo

Run the complete system demo:

```bash
python demo_hr_assistant.py
```

This will:

- Start both backend and frontend automatically
- Test the API connection
- Open the web interface in your browser

## Interface Overview

### Main Components

1. **Header Section**

   - Title and description
   - API connection status
   - Instructions sidebar

2. **Input Method Selection**

   - **Upload Files**: Upload PDF, DOCX, DOC, TXT files
   - **Paste Text**: Direct text input in text areas

3. **Input Section**

   - **File Upload Mode**:
     - Resume file uploader (PDF, DOCX, DOC, TXT)
     - Job description file uploader (PDF, DOCX, DOC, TXT)
     - Automatic text extraction and preview
   - **Text Paste Mode**:
     - Resume text area for direct input
     - Job description text area for direct input
   - **Analyze Button**: Trigger similarity analysis

4. **Results Section**
   - **Score Metrics**: Overall score, skill match, experience, education
   - **Progress Bars**: Visual representation of scores
   - **Detailed Analysis**: Comprehensive breakdown with tabs

### Results Display

#### Score Metrics

- **Overall Score**: Weighted combination of all factors
- **Skill Match**: Technical skills alignment
- **Experience Alignment**: Experience level matching
- **Education Match**: Education requirement alignment

#### Progress Visualization

- **Overall Similarity**: Main progress bar
- **Component Scores**: Individual progress bars for each factor
- **Percentage Display**: Precise score values

#### Detailed Analysis Tabs

1. **🔍 Skills Analysis**

   - Skills found in resume
   - Skills required for job
   - Skill categorization

2. **📋 Matching Details**

   - Matched skills (✅)
   - Missing skills (❌)
   - Extra skills (➕)

3. **⚖️ Scoring Weights**

   - Formula explanation
   - Weight values (α, β, γ)
   - Calculation breakdown

4. **📊 Additional Info**
   - Semantic similarity
   - Additional metadata

## API Integration

### Endpoints Used

- **Health Check**: `GET /health`
- **Similarity Analysis**: `POST /similarity`

### Request Format

```json
{
  "resume": "Resume text here...",
  "job_desc": "Job description here..."
}
```

### Response Format

```json
{
  "similarity_score": 0.854,
  "skill_match": 0.635,
  "experience_alignment": 1.000,
  "education_match": 1.000,
  "semantic_similarity": 0.651,
  "details": {
    "resume_skills": [...],
    "job_skills": [...],
    "matched_skills": [...],
    "missing_skills": [...],
    "weights": {...}
  }
}
```

## Error Handling

### Connection Issues

- **API Not Available**: Clear error message with troubleshooting steps
- **Connection Timeout**: Graceful handling with retry suggestions
- **Server Error**: Detailed error reporting

### Input Validation

- **Empty Resume**: Warning message
- **Empty Job Description**: Warning message
- **Invalid Input**: Error handling with suggestions

### User Feedback

- **Progress Indicators**: Real-time status updates
- **Success Messages**: Confirmation of successful analysis
- **Error Messages**: Clear, actionable error descriptions

## Customization

### Styling

The app uses Streamlit's built-in styling with custom configurations:

- **Page Config**: Wide layout, custom title and icon
- **Color Scheme**: Default Streamlit theme
- **Responsive Design**: Adapts to different screen sizes

### Configuration

Modify the API endpoint in `app.py`:

```python
API_BASE_URL = "http://localhost:8000"
SIMILARITY_ENDPOINT = f"{API_BASE_URL}/similarity"
```

### Adding Features

Extend the app by modifying `frontend/app.py`:

- Add new input fields
- Create additional analysis tabs
- Implement new visualization components

## Performance

### Response Times

- **Typical Analysis**: 2-5 seconds
- **Progress Updates**: Real-time feedback
- **Error Handling**: Immediate user notification

### Optimization Tips

- Use shorter text inputs for faster processing
- Monitor API response times
- Consider caching for repeated analyses

## Troubleshooting

### Common Issues

1. **"Cannot connect to API server"**

   - Ensure FastAPI backend is running on port 8000
   - Check firewall settings
   - Verify API endpoint configuration

2. **"Request timed out"**

   - Check API server status
   - Reduce input text length
   - Monitor server resources

3. **"Module not found" errors**

   - Install required dependencies: `pip install streamlit requests`
   - Check Python environment
   - Verify import paths

4. **Port conflicts**
   - Change ports in startup scripts
   - Check for other services using ports 8000/8501
   - Use different ports if needed

### Debug Mode

Run Streamlit in debug mode:

```bash
streamlit run frontend/app.py --logger.level debug
```

### Logs

Check Streamlit logs for detailed error information:

- Console output
- Browser developer tools
- Streamlit error messages

## Development

### File Structure

```
frontend/
├── app.py                 # Main Streamlit application
└── __init__.py           # Package initialization
```

### Key Functions

- `check_api_health()`: Verify API connection
- `send_similarity_request()`: Send analysis request
- `display_score_results()`: Show similarity scores
- `display_detailed_analysis()`: Show comprehensive results

### Testing

Run integration tests:

```bash
python test_streamlit_integration.py
```

Test individual components:

```bash
python -c "import sys; sys.path.append('frontend'); import app; print('✅ App imports successfully')"
```

## Integration with HR Assistant

The Streamlit frontend integrates seamlessly with the complete HR Assistant system:

- **FastAPI Backend**: RESTful API for similarity scoring
- **ResumeParser**: Advanced resume text processing
- **SkillExtractor**: Technical skill identification
- **SimilarityScorer**: Comprehensive matching algorithm

## License

This Streamlit frontend is part of the HR Assistant project and follows the same licensing terms.

## Support

For issues and questions:

1. Check the troubleshooting section above
2. Review the API documentation at http://localhost:8000/docs
3. Test the backend independently
4. Verify all dependencies are installed correctly
