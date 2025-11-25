# How to Run the HR Assistant on Streamlit

## Quick Start

### Step 1: Navigate to Project Directory

```bash
cd "/Users/junfeibai/Desktop/5560/Hr Assistant"
```

### Step 2: Run Streamlit

```bash
streamlit run resume_jd_matcher.py
```

That's it! The app will automatically open in your browser.

## Complete Setup (First Time Only)

### 1. Install Dependencies

```bash
pip install -r resume_matcher_requirements.txt
```

### 2. Install Google Gemini (for AI enhancements)

```bash
pip install google-generativeai
```

### 3. Set API Key

Provide your own Google Gemini API key:

```bash
export GOOGLE_API_KEY="YOUR_API_KEY_HERE"
```

### 4. Run the Application

```bash
streamlit run resume_jd_matcher.py
```

## What You'll See

1. **Browser opens automatically** at `http://localhost:8501`
2. **Sidebar** with configuration options:
   - AI Enhancement toggle
   - Data directory path
   - Matching mode selection
   - Weight controls (for Semantic mode)
3. **Main dashboard** with:
   - Similarity heatmaps
   - Top matches
   - Detailed scores
   - Resume and JD details

## Using the Application

### Step 1: Configure Settings

1. **Set Data Directory**: Enter the path to your folder containing resumes and job descriptions
   - Example: `/Users/junfeibai/Desktop/5560/test`
   - Resumes: Any files (PDF, DOCX, TXT)
   - Job Descriptions: Files starting with "JD" (case-insensitive)

2. **Choose Matching Mode**:
   - **Semantic Mode**: Advanced AI-powered matching with section breakdown
   - **Improved Similarity Mode**: Enhanced preprocessing and scaling

3. **Enable AI Enhancements** (Optional):
   - Check the box to use Google Gemini for better explanations
   - You'll see "✅ AI Enhancement Active (Google Gemini)" if configured

### Step 2: Load Documents

1. Click **"🔄 Load Documents"** button in the sidebar
2. Wait for processing (shows spinner)
3. You'll see document counts in the main area

### Step 3: View Results

The app automatically shows:
- **Similarity Heatmap**: Visual matrix of all matches
- **Top Matches**: Ranked candidates for each job
- **Detailed Scores**: Complete similarity matrix
- **Section Breakdown**: Individual section scores (Semantic mode)
- **Resume Details**: Click to view candidate information
- **Job Details**: Click to view job requirements and top candidates

## Troubleshooting

### Port Already in Use

If port 8501 is busy, use a different port:

```bash
streamlit run resume_jd_matcher.py --server.port 8502
```

### API Key Not Working

Check if it's set:

```bash
echo $GOOGLE_API_KEY
```

If empty, set it again:

```bash
export GOOGLE_API_KEY="YOUR_API_KEY_HERE"
```

### Dependencies Missing

Install all requirements:

```bash
pip install -r resume_matcher_requirements.txt
pip install google-generativeai
```

### File Not Found

Make sure you're in the correct directory:

```bash
pwd
# Should show: /Users/junfeibai/Desktop/5560/Hr Assistant
```

## Command Summary

```bash
# Navigate to project
cd "/Users/junfeibai/Desktop/5560/Hr Assistant"

# Set API key (if needed)
export GOOGLE_API_KEY="YOUR_API_KEY_HERE"

# Run the app
streamlit run resume_jd_matcher.py
```

## Features Available

✅ **Multi-format Support**: PDF, DOCX, TXT files
✅ **AI-Powered Matching**: Semantic understanding with Sentence-BERT
✅ **Interactive Visualizations**: Heatmaps, charts, detailed breakdowns
✅ **AI Explanations**: Google Gemini-powered match explanations
✅ **Professional Summaries**: AI-generated candidate summaries
✅ **Real-time Weight Adjustment**: Customize matching weights
✅ **Export to CSV**: Download results with explanations

Enjoy using your HR Assistant! 🎯

