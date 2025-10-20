# Data Normalization System

## 🎯 Overview

The HR Assistant now includes intelligent data normalization that automatically handles inconsistent, partial, or messy input data for both job descriptions and candidate information. This ensures the backend always receives structured, consistent data regardless of input format.

## ✨ Key Features

### **Automatic Field Mapping**

- Maps inconsistent field names to standardized schema
- Handles variations like `job_title` → `title`, `job_description` → `description`
- Extracts data from long text descriptions using NLP heuristics

### **Intelligent Data Extraction**

- **Job Descriptions**: Extracts title, description, requirements, location, salary
- **Candidate Data**: Extracts name, email, phone, skills, experience summary
- Uses regex patterns and keyword detection for robust extraction

### **Flexible Input Handling**

- Accepts any JSON structure with any field names
- Processes long text descriptions automatically
- Handles partial data gracefully
- Robust error handling for edge cases

## 🏗️ Architecture

### **Job Description Normalization**

```
Input: Any JSON with job data
↓
backend/utils/jd_formatter.py
↓
Output: Standardized schema
{
  "title": "string",
  "description": "string",
  "requirements": "string",
  "location": "string",
  "salary_range": "string"
}
```

### **Candidate Data Normalization**

```
Input: Parsed text + any JSON with candidate data
↓
backend/utils/candidate_formatter.py
↓
Output: Standardized schema
{
  "name": "string",
  "email": "string",
  "phone": "string",
  "skills": ["string", "string"],
  "experience_summary": "string"
}
```

## 🔧 Implementation

### **1. Job Description Formatter (`backend/utils/jd_formatter.py`)**

#### **Core Function**

```python
def normalize_jd_input(raw_input: Dict[str, Any]) -> Dict[str, str]:
    """Normalize job description input into standardized schema"""
```

#### **Extraction Methods**

- **Title**: First line detection, "Job Title:" patterns, title case matching
- **Description**: Section detection ("About the job", "Job Summary")
- **Requirements**: Bullet point extraction, "Requirements:" sections
- **Location**: Pattern matching for city/state, "Remote" keywords
- **Salary**: Regex patterns for salary ranges, currency symbols

#### **Example Usage**

```python
from backend.utils.jd_formatter import normalize_jd_input

# Input: Inconsistent field names
raw_input = {
    "job_title": "Python Developer",
    "job_description": "Develop Python applications",
    "must_have": "Python, Django, 3+ years",
    "city": "San Francisco, CA",
    "compensation": "$100,000 - $130,000"
}

# Output: Standardized schema
normalized = normalize_jd_input(raw_input)
# {
#   "title": "Python Developer",
#   "description": "Develop Python applications",
#   "requirements": "Python, Django, 3+ years",
#   "location": "San Francisco, CA",
#   "salary_range": "$100,000 - $130,000"
# }
```

### **2. Candidate Formatter (`backend/utils/candidate_formatter.py`)**

#### **Core Function**

```python
def normalize_candidate_input(parsed_text: str, raw_input: Dict[str, Any] = None) -> Dict[str, Any]:
    """Normalize candidate/resume input into standardized schema"""
```

#### **Extraction Methods**

- **Name**: First line detection, "Name:" patterns, title case matching
- **Email**: Regex pattern matching for email addresses
- **Phone**: Multiple phone number format detection
- **Skills**: Technical skills database matching, section extraction
- **Experience**: Section detection, job title/company pattern matching

#### **Example Usage**

```python
from backend.utils.candidate_formatter import normalize_candidate_input

# Input: Parsed resume text
resume_text = """
John Smith
Senior Software Engineer
john.smith@email.com
(555) 123-4567

Skills: Python, JavaScript, React, AWS
Experience: 5+ years in software development
"""

# Output: Standardized schema
normalized = normalize_candidate_input(resume_text)
# {
#   "name": "John Smith",
#   "email": "john.smith@email.com",
#   "phone": "(555) 123-4567",
#   "skills": ["Python", "JavaScript", "React", "AWS"],
#   "experience_summary": "5+ years in software development"
# }
```

## 🚀 API Integration

### **Updated Endpoints**

#### **Job Description Analysis**

- **Original**: `POST /api/v1/jobs/analyze` (strict schema)
- **New**: `POST /api/v1/jobs/analyze-flexible` (flexible input)

```python
# Flexible endpoint usage
@router.post("/analyze-flexible", response_model=JobAnalysis)
async def analyze_job_flexible(job_request: FlexibleJobRequest):
    # Normalize the input data
    normalized_data = normalize_jd_input(job_request.data)

    # Process with normalized data
    skills = skill_extractor.extract_skills(normalized_data['description'])
    # ... rest of analysis
```

#### **Candidate Parsing**

- **Original**: `POST /api/v1/candidates/parse` (file upload)
- **New**: `POST /api/v1/candidates/parse-flexible` (flexible input)

```python
# Flexible endpoint usage
@router.post("/parse-flexible", response_model=Candidate)
async def parse_resume_flexible(data: Dict[str, Any]):
    # Extract text content
    text_content = extract_text_from_data(data)

    # Normalize candidate data
    normalized_data = normalize_candidate_input(text_content, data)

    # Create candidate with normalized data
    candidate = Candidate(**normalized_data)
```

## 🧪 Testing

### **Test Script**

```bash
python test_normalization.py
```

### **Test Coverage**

- ✅ Long text input processing
- ✅ Inconsistent field name mapping
- ✅ Partial data handling
- ✅ Already normalized data passthrough
- ✅ Edge case handling (empty, null, special characters)
- ✅ API integration testing

### **Test Results**

```
🎉 All normalization tests completed successfully!
✅ Job description normalization: WORKING
✅ Candidate data normalization: WORKING
✅ API integration: READY
✅ Edge case handling: ROBUST
```

## 📊 Supported Input Formats

### **Job Description Inputs**

```json
// Long text description
{
  "data": {
    "raw_text": "Senior Python Developer\n\nAbout the Role:\n..."
  }
}

// Inconsistent field names
{
  "data": {
    "job_title": "Data Scientist",
    "job_description": "Analyze data...",
    "must_have": "Python, SQL, ML",
    "city": "New York, NY",
    "compensation": "$100,000 - $130,000"
  }
}

// Partial data
{
  "data": {
    "title": "Software Engineer",
    "description": "Develop applications...",
    "requirements": "Python, JavaScript, 2+ years"
  }
}
```

### **Candidate Inputs**

```json
// Parsed resume text
{
  "text": "John Smith\nSenior Engineer\njohn@email.com\n..."
}

// Inconsistent field names
{
  "candidate_name": "Jane Doe",
  "email_address": "jane@example.com",
  "mobile_number": "555-123-4567",
  "technical_skills": ["Python", "SQL"],
  "work_history": "5+ years experience"
}

// Partial data
{
  "name": "Bob Johnson",
  "email": "bob@company.com",
  "skills": ["Java", "Spring"]
}
```

## 🔍 Extraction Heuristics

### **Job Description Extraction**

- **Title**: First line, "Job Title:" patterns, title case detection
- **Description**: "About the job", "Job Summary" sections
- **Requirements**: Bullet points, "Requirements:" sections
- **Location**: City/state patterns, "Remote" keywords
- **Salary**: Currency symbols, number ranges

### **Candidate Extraction**

- **Name**: First line, "Name:" patterns, title case
- **Email**: Standard email regex patterns
- **Phone**: Multiple phone format patterns
- **Skills**: Technical skills database matching
- **Experience**: Job title/company pattern detection

## 🛡️ Error Handling

### **Robust Error Handling**

- ✅ Empty input handling
- ✅ Null value processing
- ✅ Type conversion (string, numeric, boolean)
- ✅ Special character handling
- ✅ Very long text processing
- ✅ Malformed data recovery

### **Fallback Strategies**

- **Missing fields**: Empty string defaults
- **Invalid data**: Type conversion attempts
- **Extraction failures**: Pattern fallbacks
- **Edge cases**: Graceful degradation

## 🎯 Benefits

### **For Developers**

- **Consistent API**: Always receive structured data
- **Reduced validation**: Automatic data cleaning
- **Flexible input**: Accept any JSON structure
- **Error resilience**: Handle malformed data gracefully

### **For Users**

- **Easy integration**: No strict schema requirements
- **Automatic processing**: Smart data extraction
- **Error tolerance**: Handle messy input data
- **Seamless experience**: Transparent normalization

## 🚀 Usage Examples

### **1. Job Description Analysis**

```python
# Send any job data format
job_data = {
    "job_title": "Python Developer",
    "job_description": "Develop Python apps...",
    "must_have": "Python, Django, 3+ years",
    "city": "San Francisco, CA"
}

# API automatically normalizes and processes
response = requests.post("/api/v1/jobs/analyze-flexible",
                        json={"data": job_data})
```

### **2. Candidate Parsing**

```python
# Send any candidate data format
candidate_data = {
    "candidate_name": "Alice Smith",
    "email_address": "alice@email.com",
    "technical_skills": ["Python", "JavaScript"],
    "work_history": "5+ years experience"
}

# API automatically normalizes and processes
response = requests.post("/api/v1/candidates/parse-flexible",
                        json=candidate_data)
```

## 🔧 Configuration

### **Customization Options**

- **Skill databases**: Modify technical skills lists
- **Pattern matching**: Adjust regex patterns
- **Field mapping**: Add new field name mappings
- **Extraction rules**: Customize extraction heuristics

### **Performance Optimization**

- **Caching**: Cache normalized results
- **Batch processing**: Process multiple items
- **Async processing**: Non-blocking normalization
- **Memory management**: Efficient text processing

## 📈 Future Enhancements

### **Planned Features**

- **Machine Learning**: AI-powered field extraction
- **Language Support**: Multi-language processing
- **Custom Schemas**: User-defined normalization rules
- **Real-time Processing**: Live data normalization
- **Analytics**: Normalization success metrics

## 🎉 Success!

The data normalization system ensures that the HR Assistant can handle any input format gracefully, providing a robust and user-friendly experience for both job description analysis and candidate data processing.

**Key Achievements:**

- ✅ Automatic field mapping and extraction
- ✅ Handles inconsistent field names
- ✅ Extracts data from long text descriptions
- ✅ Robust error handling for edge cases
- ✅ Seamless API integration
- ✅ Comprehensive testing coverage
