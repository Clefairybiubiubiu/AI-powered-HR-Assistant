# Resume-JD Matching Dashboard

A comprehensive Streamlit application that matches resumes with job descriptions using TF-IDF and cosine similarity.

## Features

- **Multi-format Support**: Handles TXT, PDF, and DOCX files
- **Smart Name Extraction**: Automatically extracts candidate names from resume content
- **Intelligent File Naming**: Renames candidates to "Name-resume" format
- **TF-IDF Analysis**: Uses scikit-learn's TfidfVectorizer for text analysis
- **Cosine Similarity**: Computes similarity scores between resumes and job descriptions
- **Interactive Dashboard**:
  - 🔥 **Heatmap**: Visual similarity matrix with color coding
  - 📈 **Top Matches**: Ranked results with match percentages
  - 📋 **Detailed Scores**: Complete similarity matrix with statistics
  - 👤 **Resume Details**: Click to view candidate information, skills, experience
  - 💼 **Job Requirements**: Click to view job details and top candidates
- **Real-time Processing**: Load and analyze documents on-the-fly
- **Auto-refresh**: Detects file changes and automatically reloads

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
test_resume_matcher.py        # Test script
resume_matcher_requirements.txt # Required packages
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
