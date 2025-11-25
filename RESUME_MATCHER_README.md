# Resume-JD Matching Dashboard

A comprehensive Streamlit application that matches resumes with job descriptions using both traditional TF-IDF and advanced semantic matching with Sentence-BERT.

## 🎯 **Dual Matching Modes**

### 📊 **TF-IDF Mode** (Traditional)

- Uses scikit-learn's TfidfVectorizer for text analysis
- Cosine similarity computation
- Fast and reliable for keyword-based matching

### 🧠 **Semantic Mode** (Advanced)

- **Sentence-BERT Integration**: Uses `all-MiniLM-L6-v2` model for semantic understanding
- **Section-based Analysis**: Separate matching for Education, Skills, and Experience
- **Weighted Scoring**: Customizable weights (default: Education 20%, Skills 50%, Experience 30%)
- **Importance Detection**: Automatically detects "Required", "Preferred", and "Nice-to-have" requirements
- **Natural Language Explanations**: AI-generated explanations for each match

## Features

- **Multi-format Support**: Handles TXT, PDF, and DOCX files
- **Smart Name Extraction**: Automatically extracts candidate names from resume content
- **Intelligent File Naming**: Renames candidates to "Name-resume" format
- **Interactive Dashboard**:
  - 🔥 **Heatmap**: Visual similarity matrix with color coding
  - 📈 **Top Matches**: Ranked results with match percentages
  - 📋 **Detailed Scores**: Complete similarity matrix with statistics
  - 🎯 **Section Breakdown**: Detailed analysis of each section's similarity (Semantic mode only)
  - 👤 **Resume Details**: Click to view candidate information, skills, experience
  - 💼 **Job Requirements**: Click to view job details and top candidates
- **Export Functionality**: Download results as CSV with explanations
- **Real-time Processing**: Load and analyze documents on-the-fly
- **Auto-refresh**: Detects file changes and automatically reloads
- **Dynamic Weight Controls**: Real-time adjustment of section weights (Semantic mode)
- **Caching**: Optimized performance with embedding caching

## 🚀 First-Time Setup Guide

Follow this checklist when you clone or download the project for the very first time.

1. **Install prerequisites**

   - Python 3.9+ (`python3 --version`)
   - pip (`python3 -m ensurepip --upgrade`)
   - Optional but recommended: `virtualenv` (`pip install virtualenv`)

2. **Create a virtual environment**

   ```bash
   cd /path/to/Hr\ Assistant
   python3 -m venv .venv
   source .venv/bin/activate  # Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**

   ```bash
   pip install --upgrade pip
   pip install -r resume_matcher_requirements.txt
   ```

   > For the Robert assistant / Gemini features you also need `google-generativeai`.  
   > If it is not already in the requirements file: `pip install google-generativeai`.

4. **Prepare runtime assets**

   - Create `data/` (or your preferred directory) with resumes/JDs.
   - Ensure `.env` remains empty or only contains non-secret defaults (the Gemini key is entered in the UI).

5. **Run the app**

   ```bash
   streamlit run resume_jd_matcher.py
   ```

   - The sidebar contains the “AI Enhancement” section → paste your Google Gemini API key when you need AI features (Robert, match explanations, etc.). The key is stored only in `st.session_state` during that session.

6. **First launch sanity checks**
   - Click **Load Documents** and verify the heatmap populates.
   - Open “Candidate Profiles” to confirm extraction works.
   - Test “Ask Robert” so you know the Gemini integration is configured.

## Installation

1. Install required packages:

```bash
pip install -r resume_matcher_requirements.txt
```

2. Run the application:

```bash
streamlit run resume_jd_matcher.py
```

## Usage

1. **Prepare your data**: Place resumes and job descriptions in a directory

   - Resume files should contain "candidate" in the filename
   - Job description files should start with "JD"
   - Supported formats: TXT, PDF, DOCX

2. **Configure the app**:

   - Set the data directory path in the sidebar
   - Choose the number of top matches to display
   - Click "Load Documents" to process files

3. **Analyze results**:
   - View the similarity heatmap
   - Check top matches for each job description
   - Examine detailed similarity scores

## File Structure

```
resume_jd_matcher.py          # Main Streamlit application
resume_matcher_requirements.txt # Required packages
RESUME_MATCHER_README.md      # Documentation
```

## Example Data Structure

```
/Users/junfeibai/Desktop/5560/test/
├── Candidate 1.docx          → Candidate-1
├── Candidate 2.pdf           → Candidate-2
├── Candidate 3.txt           → Candidate-3
├── Junfei Bai - CV.pdf       → Candidate-4
├── JD1.docx                  → JD1
├── JD2.docx                  → JD2
└── JD3.docx                  → JD3
```

## 🧠 **Semantic Matching Features**

### **Section-based Analysis**

- **Education**: Matches academic qualifications and degrees
- **Skills**: Matches technical skills with importance weighting
- **Experience**: Matches work experience and career progression

### **Importance Detection**

- **Required** (1.0x): "Python (Required)", "Must have SQL"
- **Preferred** (0.75x): "R (Preferred)", "Plus: Machine Learning"
- **Nice-to-have** (0.5x): "Excel (Nice-to-have)", "Optional: Docker"

### **Weighted Scoring**

- **Education Weight**: 20% (default)
- **Skills Weight**: 50% (default)
- **Experience Weight**: 30% (default)
- **Dynamic Adjustment**: Real-time weight changes via sliders

### **Natural Language Explanations**

- AI-generated explanations for each candidate-JD match
- Highlights strongest alignment areas
- Provides context for similarity scores

### **Export Functionality**

- Download results as CSV with explanations
- Includes section scores and match percentages
- Compatible with Excel and other analysis tools

**File Naming Rules:**

- **Job Descriptions**: Files starting with "JD" (case-insensitive)
- **Candidate Resumes**: ALL other files (automatically renamed to Candidate-1, Candidate-2, etc.)
- **Original filenames**: Preserved for reference in the app

## Similarity Score Interpretation

- **🟢 Green (0.7+)**: Excellent match
- **🟡 Yellow (0.4-0.7)**: Good match
- **🔴 Red (<0.4)**: Poor match

## Technical Details

- **Text Processing**: Removes special characters, converts to lowercase, filters stop words
- **TF-IDF**: Uses 1-2 gram features with max 1000 features
- **Similarity**: Cosine similarity between TF-IDF vectors
- **Visualization**: Plotly-based interactive heatmaps

## Test Results

The application successfully processes:

- ✅ 3 resumes (TXT, PDF, DOCX formats)
- ✅ 3 job descriptions (DOCX format)
- ✅ Computes 3x3 similarity matrix
- ✅ Generates interactive visualizations
- ✅ Provides ranked match results

## Troubleshooting

- **File format issues**: Ensure files are not corrupted
- **Encoding problems**: The app handles UTF-8 and Latin-1 encodings
- **Memory issues**: Large files may require more RAM
- **Port conflicts**: Change the port if 8502 is occupied

## Future Enhancements

- Support for more file formats (RTF, ODT)
- Advanced text preprocessing (lemmatization, stemming)
- Machine learning-based matching
- Export results to CSV/Excel
- Batch processing capabilities

## ✅ Pre-Publish Safety Checklist

Before pushing the repository to GitHub:

1. **Confirm no sensitive files are tracked**

   ```bash
   git status
   git ls-files '*.pdf' '*.docx'
   ```

   `.gitignore` already excludes `.env`, `.env.*`, and common resume formats, but double-check.

2. **Verify no API keys in code**

   ```bash
   rg -n "API_KEY" -g'*.*'
   rg -n "gemini" resume_jd_matcher.py resume_matcher -g'*.*'
   ```

   All keys must be provided via the Streamlit sidebar at runtime.

3. **Remove local caches / artifacts**

   ```bash
   rm -rf .streamlit/
   rm -rf __pycache__ .venv
   ```

4. **Optional:** run tests / linting (if configured) and capture the output in the commit message.

Following this guide ensures new contributors can spin up the project quickly and that the repository stays clean and safe to publish.
