# 🎯 AI-Powered HR Assistant

A comprehensive Streamlit application that intelligently matches resumes with job descriptions using both traditional TF-IDF and advanced semantic matching with Sentence-BERT. Enhanced with AI-powered features including the Robert assistant, O*NET skill taxonomy integration, and intelligent resume parsing.

![Python](https://img.shields.io/badge/python-3.9+-blue.svg)
![Streamlit](https://img.shields.io/badge/streamlit-1.0+-red.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## ✨ Features

### 🧠 **Dual Matching Modes**

- **📊 TF-IDF Mode**: Fast keyword-based matching using scikit-learn's TfidfVectorizer
- **🧠 Semantic Mode**: Advanced semantic understanding with Sentence-BERT embeddings
  - Section-based analysis (Education, Skills, Experience, Summary)
  - Customizable weighted scoring
  - AI-generated match explanations
  - O*NET skill taxonomy integration (beta)

### 🤖 **AI-Powered Enhancements**

- **Robert Assistant**: Conversational AI helper for resume analysis and matching insights
- **Smart Resume Parsing**: Enhanced extraction using Google Gemini API
- **Professional Summaries**: AI-generated candidate summaries
- **Skill Taxonomy**: O*NET-based skill expansion for richer matching

### 📊 **Interactive Dashboard**

- 🔥 **Similarity Heatmap**: Visual matrix with color-coded scores
- 📈 **Top Matches**: Ranked candidate-job pairs
- 📋 **Detailed Analysis**: Complete similarity breakdowns
- 🎯 **Component Analysis**: Section-by-section matching scores
- 👤 **Candidate Profiles**: Detailed resume information
- 💼 **Job Requirements**: JD analysis and top candidates

### 🛠️ **Technical Features**

- Multi-format support (PDF, DOCX, TXT)
- Smart name extraction from resumes
- Real-time document processing
- Auto-refresh on file changes
- Embedding caching for performance
- Export results to CSV

## 🚀 Quick Start

### Prerequisites

- Python 3.9 or higher
- pip package manager

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/Clefairybiubiubiu/AI-powered-HR-Assistant.git
   cd AI-powered-HR-Assistant
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install --upgrade pip
   pip install -r resume_matcher_requirements.txt
   ```

4. **Run the application**
   ```bash
   streamlit run resume_jd_matcher.py
   ```

   Or use the provided script:
   ```bash
   ./start_streamlit.sh
   ```

5. **Access the dashboard**
   - Open your browser to `http://localhost:8501`
   - Upload resumes and job descriptions via the sidebar
   - Click "Load Documents" to start matching

## 🔑 API Key Setup (Optional)

For AI-powered features (Robert assistant, enhanced parsing, summaries):

1. Get a free Google Gemini API key from [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Enter the key in the sidebar under "🤖 AI Enhancement"
3. Enable "Enable AI-Powered Enhancements" checkbox
4. The key is stored only in session state (not saved to disk)

## 📁 Project Structure

```
AI-powered-HR-Assistant/
├── resume_jd_matcher.py          # Main Streamlit application
├── enhanced_resume_parser.py      # Enhanced resume parsing utilities
├── resume_matcher/                # Core package
│   ├── matchers/                  # Matching algorithms
│   ├── utils/                      # Utilities (LLM client, document processor, etc.)
│   ├── skill_taxonomy_onet.py     # O*NET skill taxonomy (beta)
│   └── config.py                  # Configuration management
├── tests/                          # Unit tests
├── cache/                          # Caching directory
│   └── onet/                       # O*NET expansion cache
├── resume_matcher_requirements.txt # Python dependencies
└── README.md                       # This file
```

## 💡 Usage Examples

### Basic Matching

1. **Upload Files**: Use the sidebar to upload resume and JD files
2. **Select Mode**: Choose between Semantic or Improved Similarity mode
3. **Load Documents**: Click "🔄 Load Documents"
4. **View Results**: Explore the heatmap, top matches, and detailed analysis

### Using Robert Assistant

1. Enable AI enhancements and enter your Gemini API key
2. Select a candidate and/or job description (optional)
3. Ask questions like:
   - "Why is this candidate a good fit for JD1?"
   - "What skills are missing from this resume?"
   - "Summarize the candidate's experience"

### O*NET Skill Expansion (Beta)

1. Enable "Enable O*NET Smart Skill Expansion" in the sidebar
2. The system will expand skills using O*NET taxonomy
3. Place O*NET CSV files in `data/onet/` for full functionality

## 🎛️ Configuration

### Matching Weights (Semantic Mode)

Adjust section weights in the sidebar:
- **Education**: Default 10%
- **Skills**: Default 40%
- **Experience**: Default 40%
- **Summary**: Default 10%

Weights automatically normalize to sum to 100%.

### File Naming Conventions

- **Job Descriptions**: Files starting with "JD" (case-insensitive)
- **Resumes**: All other files are treated as candidate resumes
- Supported formats: `.pdf`, `.docx`, `.doc`, `.txt`

## 🧪 Testing

Run the test suite:

```bash
pytest tests/
```

## 📊 Similarity Score Interpretation

- **🟢 Green (0.7+)**: Excellent match
- **🟡 Yellow (0.4-0.7)**: Good match
- **🔴 Red (<0.4)**: Poor match

## 🔧 Troubleshooting

### Common Issues

**"SentenceTransformer not available"**
```bash
pip install sentence-transformers
```

**"Google Gemini API not configured"**
- Enter your API key in the sidebar
- Ensure `google-generativeai` is installed: `pip install google-generativeai`

**"No matches found"**
- Verify files are uploaded correctly
- Check that at least one resume and one JD are loaded
- Ensure file formats are supported (PDF, DOCX, TXT)

**Low similarity scores**
- Try enabling AI enhancements for better parsing
- Check that resume sections are being extracted (view Candidate Profiles tab)
- Adjust matching weights in Semantic mode

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- [Sentence-Transformers](https://www.sbert.net/) for semantic embeddings
- [Streamlit](https://streamlit.io/) for the web framework
- [Google Gemini](https://ai.google.dev/) for AI enhancements
- [O*NET](https://www.onetcenter.org/) for skill taxonomy data

## 📧 Contact

For questions or issues, please open an issue on GitHub.

---

**Made with ❤️ for HR professionals**

