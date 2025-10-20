"""
Resume-JD Matching Dashboard
A Streamlit app that matches resumes with job descriptions using TF-IDF and cosine similarity.
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import re
import time
from pathlib import Path
from typing import List, Dict, Tuple
import plotly.express as px
import plotly.graph_objects as go
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import PyPDF2
import docx
from docx import Document
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Resume-JD Matcher",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

class DocumentProcessor:
    """Handles document parsing for different file formats."""
    
    @staticmethod
    def extract_text_from_txt(file_path: str) -> str:
        """Extract text from TXT file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read()
        except UnicodeDecodeError:
            with open(file_path, 'r', encoding='latin-1') as file:
                return file.read()
    
    @staticmethod
    def extract_text_from_pdf(file_path: str) -> str:
        """Extract text from PDF file."""
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
                return text
        except Exception as e:
            st.error(f"Error reading PDF {file_path}: {str(e)}")
            return ""
    
    @staticmethod
    def extract_text_from_docx(file_path: str) -> str:
        """Extract text from DOCX file."""
        try:
            doc = Document(file_path)
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            return text
        except Exception as e:
            st.error(f"Error reading DOCX {file_path}: {str(e)}")
            return ""
    
    @classmethod
    def extract_text(cls, file_path: str) -> str:
        """Extract text from any supported file format."""
        file_ext = Path(file_path).suffix.lower()
        
        if file_ext == '.txt':
            return cls.extract_text_from_txt(file_path)
        elif file_ext == '.pdf':
            return cls.extract_text_from_pdf(file_path)
        elif file_ext == '.docx':
            return cls.extract_text_from_docx(file_path)
        else:
            st.warning(f"Unsupported file format: {file_ext}")
            return ""

class ResumeJDMatcher:
    """Main class for matching resumes with job descriptions."""
    
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self.processor = DocumentProcessor()
        self.resumes = {}
        self.job_descriptions = {}
        self.similarity_matrix = None
        self.candidate_names = []
        self.jd_names = []
    
    def load_documents(self):
        """Load all resumes and job descriptions from the directory."""
        if not self.data_dir.exists():
            st.error(f"Directory {self.data_dir} does not exist!")
            return
        
        # Load job descriptions first (files starting with "JD")
        jd_files = [f for f in self.data_dir.glob("*") if f.is_file() and f.name.lower().startswith("jd")]
        for file_path in jd_files:
            name = file_path.stem
            text = self.processor.extract_text(str(file_path))
            if text.strip():
                self.job_descriptions[name] = text
                self.jd_names.append(name)
        
        # Load ALL other files as candidate resumes (anything that doesn't start with "JD")
        all_files = [f for f in self.data_dir.glob("*") if f.is_file()]
        candidate_files = [f for f in all_files if not f.name.lower().startswith("jd")]
        
        # Sort files by name for consistent ordering
        candidate_files.sort(key=lambda x: x.name.lower())
        
        for i, file_path in enumerate(candidate_files, 1):
            text = self.processor.extract_text(str(file_path))
            if text.strip():
                # Extract candidate name from resume content
                candidate_name = self.extract_candidate_name(text)
                # Create standardized name: Name-resume
                standardized_name = f"{candidate_name}-resume"
                
                self.resumes[standardized_name] = text
                self.candidate_names.append(standardized_name)
                
                # Store original filename and extracted name for reference
                if not hasattr(self, 'original_filenames'):
                    self.original_filenames = {}
                if not hasattr(self, 'extracted_names'):
                    self.extracted_names = {}
                
                self.original_filenames[standardized_name] = file_path.name
                self.extracted_names[standardized_name] = candidate_name
        
        st.success(f"Loaded {len(self.resumes)} resumes and {len(self.job_descriptions)} job descriptions")
        
        # Show mapping of standardized names to original filenames
        if hasattr(self, 'original_filenames') and self.original_filenames:
            st.info("📋 **File Mapping:**")
            for std_name, orig_name in self.original_filenames.items():
                st.write(f"• {std_name} ← {orig_name}")
    
    def extract_candidate_name(self, text: str) -> str:
        """Extract candidate name from resume text."""
        lines = text.split('\n')
        
        # Look for name patterns in the first few lines
        for i, line in enumerate(lines[:10]):  # Check first 10 lines
            line = line.strip()
            if not line:
                continue
            
            # Pattern 1: "name: Gary" or "Name: Gary" format
            name_match = re.search(r'(?:name|Name)\s*:\s*([A-Za-z\s]+)', line)
            if name_match:
                name = name_match.group(1).strip()
                # Clean up the name (remove extra spaces, symbols)
                name = re.sub(r'[^\w\s]', '', name).strip()
                if name and len(name.split()) >= 1:
                    return name
            
            # Pattern 2: "Name (Title)" format
            match = re.match(r'^([A-Za-z\s]+)\s*\([^)]+\)', line)
            if match:
                name = match.group(1).strip()
                # Clean up symbols and special characters
                name = re.sub(r'[^\w\s]', '', name).strip()
                if len(name.split()) >= 1:  # At least first name
                    return name
            
            # Pattern 3: Just name on first line (clean version)
            if i == 0 and len(line.split()) >= 1:
                # Remove common prefixes and suffixes
                clean_line = re.sub(r'^(name|Name|contact|Contact)\s*:?\s*', '', line, flags=re.IGNORECASE)
                clean_line = re.sub(r'[^\w\s]', '', clean_line).strip()
                
                # Check if it looks like a name (no special characters, reasonable length)
                if (clean_line and 
                    re.match(r'^[A-Za-z\s]+$', clean_line) and 
                    2 <= len(clean_line) <= 50 and
                    not any(word.lower() in ['contact', 'email', 'phone', 'address', 'summary', 'objective'] for word in clean_line.split())):
                    return clean_line
            
            # Pattern 4: Name followed by title on next line
            if i < len(lines) - 1:
                next_line = lines[i + 1].strip()
                clean_line = re.sub(r'[^\w\s]', '', line).strip()
                if (len(clean_line.split()) >= 1 and 
                    not any(word.lower() in ['contact', 'email', 'phone', 'address', 'summary', 'objective'] for word in clean_line.split()) and
                    any(word.lower() in ['engineer', 'developer', 'scientist', 'analyst', 'manager', 'specialist'] for word in next_line.split())):
                    return clean_line
        
        # Fallback: return first meaningful line (cleaned)
        for line in lines[:5]:
            line = line.strip()
            if line:
                # Clean the line
                clean_line = re.sub(r'^(name|Name|contact|Contact)\s*:?\s*', '', line, flags=re.IGNORECASE)
                clean_line = re.sub(r'[^\w\s]', '', clean_line).strip()
                if clean_line and len(clean_line.split()) >= 1:
                    return clean_line
        
        return "Unknown Candidate"
    
    def preprocess_text(self, text: str) -> str:
        """Clean and preprocess text for better matching."""
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters but keep spaces
        text = re.sub(r'[^\w\s]', ' ', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove common stop words (basic list)
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should'}
        words = text.split()
        words = [word for word in words if word not in stop_words and len(word) > 2]
        
        return ' '.join(words)
    
    def compute_similarity(self):
        """Compute TF-IDF and cosine similarity between resumes and job descriptions."""
        if not self.resumes or not self.job_descriptions:
            st.error("No documents loaded!")
            return
        
        # Prepare documents
        all_docs = []
        doc_labels = []
        
        # Add resumes
        for name, text in self.resumes.items():
            processed_text = self.preprocess_text(text)
            all_docs.append(processed_text)
            doc_labels.append(f"Resume: {name}")
        
        # Add job descriptions
        for name, text in self.job_descriptions.items():
            processed_text = self.preprocess_text(text)
            all_docs.append(processed_text)
            doc_labels.append(f"JD: {name}")
        
        # Compute TF-IDF
        vectorizer = TfidfVectorizer(
            max_features=1000,
            ngram_range=(1, 2),
            min_df=1,
            max_df=0.8
        )
        
        tfidf_matrix = vectorizer.fit_transform(all_docs)
        
        # Compute cosine similarity
        similarity_matrix = cosine_similarity(tfidf_matrix)
        
        # Extract resume-JD similarities
        num_resumes = len(self.resumes)
        num_jds = len(self.job_descriptions)
        
        self.similarity_matrix = similarity_matrix[:num_resumes, num_resumes:]
        
        return self.similarity_matrix
    
    def get_top_matches(self, top_k: int = 3) -> pd.DataFrame:
        """Get top matches for each job description."""
        if self.similarity_matrix is None:
            return pd.DataFrame()
        
        results = []
        
        for jd_idx, jd_name in enumerate(self.jd_names):
            similarities = self.similarity_matrix[:, jd_idx]
            top_indices = np.argsort(similarities)[::-1][:top_k]
            
            for rank, resume_idx in enumerate(top_indices):
                results.append({
                    'Job Description': jd_name,
                    'Rank': rank + 1,
                    'Candidate': self.candidate_names[resume_idx],
                    'Similarity Score': similarities[resume_idx],
                    'Match Percentage': f"{similarities[resume_idx] * 100:.1f}%"
                })
        
        return pd.DataFrame(results)
    
    def get_directory_info(self) -> Dict:
        """Get information about files in the directory."""
        if not self.data_dir.exists():
            return {"resume_files": [], "jd_files": [], "total_files": 0}
        
        resume_files = []
        jd_files = []
        
        for file_path in self.data_dir.glob("*"):
            if file_path.is_file():
                if file_path.name.lower().startswith("jd"):
                    jd_files.append({
                        "name": file_path.name,
                        "size": file_path.stat().st_size,
                        "modified": file_path.stat().st_mtime
                    })
                else:
                    # All non-JD files are treated as candidate resumes
                    resume_files.append({
                        "name": file_path.name,
                        "size": file_path.stat().st_size,
                        "modified": file_path.stat().st_mtime
                    })
        
        return {
            "resume_files": resume_files,
            "jd_files": jd_files,
            "total_files": len(resume_files) + len(jd_files)
        }
    
    def extract_resume_summary(self, resume_text: str) -> Dict:
        """Extract key information from resume."""
        lines = resume_text.split('\n')
        summary = {
            'name': self.extract_candidate_name(resume_text),
            'email': '',
            'phone': '',
            'location': '',
            'skills': [],
            'experience': [],
            'education': [],
            'summary': ''
        }
        
        # Extract contact information
        for line in lines:
            line_lower = line.lower()
            if '@' in line and 'email' in line_lower:
                summary['email'] = line.strip()
            elif 'phone' in line_lower or re.search(r'\(\d{3}\)', line):
                summary['phone'] = line.strip()
            elif any(city in line_lower for city in ['san francisco', 'new york', 'seattle', 'chicago', 'boston', 'austin']):
                summary['location'] = line.strip()
        
        # Extract skills (look for skills sections)
        in_skills_section = False
        for line in lines:
            line_lower = line.lower()
            if 'skill' in line_lower and ('technical' in line_lower or 'core' in line_lower):
                in_skills_section = True
                continue
            elif in_skills_section and line.strip():
                if any(word in line_lower for word in ['experience', 'education', 'work', 'project']):
                    in_skills_section = False
                    break
                # Extract skills from this line
                skills = re.findall(r'\b[A-Za-z][A-Za-z0-9\s]+\b', line)
                summary['skills'].extend([skill.strip() for skill in skills if len(skill.strip()) > 2])
        
        # Extract experience (look for job titles and companies)
        for line in lines:
            if any(word in line.lower() for word in ['engineer', 'developer', 'scientist', 'analyst', 'manager']) and '|' in line:
                summary['experience'].append(line.strip())
        
        # Extract education
        for line in lines:
            if any(word in line.lower() for word in ['university', 'college', 'bachelor', 'master', 'phd', 'degree']):
                summary['education'].append(line.strip())
        
        # Extract summary/objective
        in_summary = False
        for line in lines:
            line_lower = line.lower()
            if any(word in line_lower for word in ['summary', 'objective', 'profile', 'about']):
                in_summary = True
                continue
            elif in_summary and line.strip():
                if any(word in line_lower for word in ['experience', 'education', 'skills', 'contact']):
                    break
                summary['summary'] += line.strip() + ' '
        
        return summary
    
    def extract_jd_requirements(self, jd_text: str) -> Dict:
        """Extract requirements from job description."""
        lines = jd_text.split('\n')
        requirements = {
            'title': '',
            'company': '',
            'location': '',
            'salary': '',
            'requirements': [],
            'responsibilities': [],
            'skills_required': []
        }
        
        # Extract title and company
        for line in lines[:5]:
            if any(word in line.lower() for word in ['engineer', 'developer', 'scientist', 'analyst', 'manager']):
                requirements['title'] = line.strip()
                break
        
        # Extract company
        for line in lines:
            if 'company:' in line.lower() or 'at ' in line.lower():
                requirements['company'] = line.strip()
                break
        
        # Extract requirements
        in_requirements = False
        for line in lines:
            line_lower = line.lower()
            if any(word in line_lower for word in ['requirement', 'qualification', 'must have', 'should have']):
                in_requirements = True
                continue
            elif in_requirements and line.strip():
                if any(word in line_lower for word in ['responsibilit', 'benefit', 'compensation', 'salary']):
                    in_requirements = False
                    break
                if line.strip().startswith('-') or line.strip().startswith('•'):
                    requirements['requirements'].append(line.strip())
        
        # Extract skills
        for line in lines:
            if any(skill in line.lower() for skill in ['python', 'java', 'sql', 'javascript', 'react', 'aws', 'docker']):
                requirements['skills_required'].append(line.strip())
        
        return requirements

def create_heatmap(similarity_matrix: np.ndarray, candidate_names: List[str], jd_names: List[str]):
    """Create an interactive heatmap of similarity scores."""
    df = pd.DataFrame(
        similarity_matrix,
        index=candidate_names,
        columns=jd_names
    )
    
    # Round to 3 decimal places for display
    df_display = df.round(3)
    
    fig = px.imshow(
        df_display,
        text_auto=True,
        aspect="auto",
        color_continuous_scale="RdYlBu_r",
        title="Resume-JD Similarity Heatmap"
    )
    
    fig.update_layout(
        xaxis_title="Job Descriptions",
        yaxis_title="Candidates",
        font=dict(size=12)
    )
    
    return fig, df

def main():
    """Main Streamlit application."""
    
    # Header
    st.title("🎯 Resume-JD Matching Dashboard")
    st.markdown("**Match candidates with job descriptions using TF-IDF and cosine similarity**")
    
    # Sidebar
    st.sidebar.header("Configuration")
    data_dir = st.sidebar.text_input(
        "Data Directory Path",
        value="/Users/junfeibai/Desktop/5560/test",
        help="Path to directory containing resumes and job descriptions"
    )
    
    top_k = st.sidebar.slider(
        "Top Matches to Display",
        min_value=1,
        max_value=10,
        value=3,
        help="Number of top matches to show for each job description"
    )
    
    # Auto-refresh option
    auto_refresh = st.sidebar.checkbox(
        "🔄 Auto-refresh on file changes",
        value=False,
        help="Automatically reload documents when files change"
    )
    
    # Manual refresh button
    if st.sidebar.button("🔄 Refresh Documents", help="Manually reload all documents"):
        if 'matcher' in st.session_state:
            del st.session_state.matcher
        if 'similarity_matrix' in st.session_state:
            del st.session_state.similarity_matrix
        st.rerun()
    
    # File change detection
    if auto_refresh and 'matcher' in st.session_state:
        current_matcher = st.session_state.matcher
        current_info = current_matcher.get_directory_info()
        
        # Store directory info in session state for comparison
        if 'last_directory_info' not in st.session_state:
            st.session_state.last_directory_info = current_info
        
        # Check if files have changed
        if current_info != st.session_state.last_directory_info:
            st.info("🔄 Files have changed! Reloading documents...")
            del st.session_state.matcher
            if 'similarity_matrix' in st.session_state:
                del st.session_state.similarity_matrix
            st.session_state.last_directory_info = current_info
            st.rerun()
    
    # Initialize matcher
    if st.sidebar.button("🔄 Load Documents") or 'matcher' not in st.session_state:
        with st.spinner("Loading documents..."):
            matcher = ResumeJDMatcher(data_dir)
            matcher.load_documents()
            
            if matcher.resumes and matcher.job_descriptions:
                st.session_state.matcher = matcher
                st.session_state.last_directory_info = matcher.get_directory_info()
                st.success("✅ Documents loaded successfully!")
            else:
                st.error("❌ No documents found or failed to load!")
                return
    
    if 'matcher' not in st.session_state:
        st.info("👆 Please click 'Load Documents' to start")
        return
    
    matcher = st.session_state.matcher
    
    # File monitoring section
    st.subheader("📁 Directory Monitoring")
    
    # Show current directory status
    dir_info = matcher.get_directory_info()
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Resume Files", len(dir_info["resume_files"]))
    
    with col2:
        st.metric("Job Description Files", len(dir_info["jd_files"]))
    
    with col3:
        st.metric("Total Files", dir_info["total_files"])
    
    # Show file details
    if dir_info["resume_files"] or dir_info["jd_files"]:
        with st.expander("📋 File Details", expanded=False):
            if dir_info["resume_files"]:
                st.write("**Resume Files:**")
                for file_info in dir_info["resume_files"]:
                    st.write(f"• {file_info['name']} ({file_info['size']} bytes)")
            
            if dir_info["jd_files"]:
                st.write("**Job Description Files:**")
                for file_info in dir_info["jd_files"]:
                    st.write(f"• {file_info['name']} ({file_info['size']} bytes)")
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📄 Loaded Documents")
        
        st.write("**Resumes:**")
        for name in matcher.candidate_names:
            st.write(f"• {name}")
        
        st.write("**Job Descriptions:**")
        for name in matcher.jd_names:
            st.write(f"• {name}")
    
    with col2:
        st.subheader("⚙️ Processing Options")
        
        if st.button("🔍 Compute Similarity", type="primary"):
            with st.spinner("Computing TF-IDF and cosine similarity..."):
                similarity_matrix = matcher.compute_similarity()
                
                if similarity_matrix is not None:
                    st.session_state.similarity_matrix = similarity_matrix
                    st.success("✅ Similarity computation completed!")
                else:
                    st.error("❌ Failed to compute similarity!")
    
    # Results section
    if 'similarity_matrix' in st.session_state and st.session_state.similarity_matrix is not None:
        st.markdown("---")
        st.subheader("📊 Results Dashboard")
        
        similarity_matrix = st.session_state.similarity_matrix
        
        # Create tabs for different views
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["🔥 Heatmap", "📈 Top Matches", "📋 Detailed Scores", "👤 Resume Details", "💼 Job Requirements"])
        
        with tab1:
            st.subheader("Similarity Heatmap")
            fig, df = create_heatmap(similarity_matrix, matcher.candidate_names, matcher.jd_names)
            st.plotly_chart(fig, use_container_width=True)
            
            # Display raw similarity matrix
            st.subheader("Raw Similarity Scores")
            st.dataframe(df, use_container_width=True)
        
        with tab2:
            st.subheader(f"Top {top_k} Matches per Job Description")
            top_matches = matcher.get_top_matches(top_k)
            
            if not top_matches.empty:
                # Display as cards
                for jd in top_matches['Job Description'].unique():
                    st.write(f"**{jd}**")
                    jd_matches = top_matches[top_matches['Job Description'] == jd]
                    
                    for _, match in jd_matches.iterrows():
                        score = match['Similarity Score']
                        color = "🟢" if score > 0.7 else "🟡" if score > 0.4 else "🔴"
                        
                        st.write(f"{color} **{match['Candidate']}** - {match['Match Percentage']}")
                    
                    st.write("---")
            else:
                st.warning("No matches found!")
        
        with tab3:
            st.subheader("Detailed Analysis")
            
            # Summary statistics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Average Similarity", f"{similarity_matrix.mean():.3f}")
            
            with col2:
                st.metric("Highest Match", f"{similarity_matrix.max():.3f}")
            
            with col3:
                st.metric("Lowest Match", f"{similarity_matrix.min():.3f}")
            
            # Detailed table
            st.subheader("All Similarity Scores")
            detailed_df = pd.DataFrame(
                similarity_matrix,
                index=matcher.candidate_names,
                columns=matcher.jd_names
            )
            
            # Add color coding
            def color_similarity(val):
                if val > 0.7:
                    return 'background-color: #d4edda'  # Green
                elif val > 0.4:
                    return 'background-color: #fff3cd'  # Yellow
                else:
                    return 'background-color: #f8d7da'  # Red
            
            styled_df = detailed_df.style.applymap(color_similarity)
            st.dataframe(styled_df, use_container_width=True)
        
        with tab4:
            st.subheader("👤 Resume Details Dashboard")
            
            # Candidate selection
            selected_candidate = st.selectbox(
                "Select a candidate to view details:",
                matcher.candidate_names,
                key="resume_selector"
            )
            
            if selected_candidate and selected_candidate in matcher.resumes:
                resume_text = matcher.resumes[selected_candidate]
                summary = matcher.extract_resume_summary(resume_text)
                
                # Display resume summary
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("### 📋 Contact Information")
                    st.write(f"**Name:** {summary['name']}")
                    if summary['email']:
                        st.write(f"**Email:** {summary['email']}")
                    if summary['phone']:
                        st.write(f"**Phone:** {summary['phone']}")
                    if summary['location']:
                        st.write(f"**Location:** {summary['location']}")
                
                with col2:
                    st.markdown("### 🎯 Professional Summary")
                    if summary['summary']:
                        st.write(summary['summary'])
                    else:
                        st.write("No summary available")
                
                # Skills section
                if summary['skills']:
                    st.markdown("### 🛠️ Skills")
                    skills_text = ", ".join(summary['skills'][:10])  # Show first 10 skills
                    st.write(skills_text)
                    if len(summary['skills']) > 10:
                        st.write(f"... and {len(summary['skills']) - 10} more skills")
                
                # Experience section
                if summary['experience']:
                    st.markdown("### 💼 Experience")
                    for exp in summary['experience'][:5]:  # Show first 5 experiences
                        st.write(f"• {exp}")
                
                # Education section
                if summary['education']:
                    st.markdown("### 🎓 Education")
                    for edu in summary['education'][:3]:  # Show first 3 education entries
                        st.write(f"• {edu}")
                
                # Similarity scores for this candidate
                st.markdown("### 📊 Match Scores")
                candidate_idx = matcher.candidate_names.index(selected_candidate)
                candidate_scores = similarity_matrix[candidate_idx]
                
                scores_df = pd.DataFrame({
                    'Job Description': matcher.jd_names,
                    'Similarity Score': candidate_scores,
                    'Match Percentage': [f"{score * 100:.1f}%" for score in candidate_scores]
                }).sort_values('Similarity Score', ascending=False)
                
                st.dataframe(scores_df, use_container_width=True)
        
        with tab5:
            st.subheader("💼 Job Requirements Dashboard")
            
            # Job description selection
            selected_jd = st.selectbox(
                "Select a job description to view requirements:",
                matcher.jd_names,
                key="jd_selector"
            )
            
            if selected_jd and selected_jd in matcher.job_descriptions:
                jd_text = matcher.job_descriptions[selected_jd]
                requirements = matcher.extract_jd_requirements(jd_text)
                
                # Display job requirements
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("### 📋 Job Information")
                    if requirements['title']:
                        st.write(f"**Title:** {requirements['title']}")
                    if requirements['company']:
                        st.write(f"**Company:** {requirements['company']}")
                    if requirements['location']:
                        st.write(f"**Location:** {requirements['location']}")
                    if requirements['salary']:
                        st.write(f"**Salary:** {requirements['salary']}")
                
                with col2:
                    st.markdown("### 🎯 Job Overview")
                    st.write("Full job description preview:")
                    st.text_area("", jd_text[:500] + "..." if len(jd_text) > 500 else jd_text, height=200)
                
                # Requirements section
                if requirements['requirements']:
                    st.markdown("### ✅ Requirements")
                    for req in requirements['requirements'][:10]:  # Show first 10 requirements
                        st.write(f"• {req}")
                
                # Skills required
                if requirements['skills_required']:
                    st.markdown("### 🛠️ Required Skills")
                    skills_text = ", ".join(requirements['skills_required'][:15])  # Show first 15 skills
                    st.write(skills_text)
                
                # Top candidates for this job
                st.markdown("### 🏆 Top Candidates for This Job")
                jd_idx = matcher.jd_names.index(selected_jd)
                jd_scores = similarity_matrix[:, jd_idx]
                
                # Get top 5 candidates
                top_indices = np.argsort(jd_scores)[::-1][:5]
                
                top_candidates_df = pd.DataFrame({
                    'Rank': range(1, len(top_indices) + 1),
                    'Candidate': [matcher.candidate_names[i] for i in top_indices],
                    'Similarity Score': jd_scores[top_indices],
                    'Match Percentage': [f"{score * 100:.1f}%" for score in jd_scores[top_indices]]
                })
                
                st.dataframe(top_candidates_df, use_container_width=True)
    
    # Footer
    st.markdown("---")
    st.markdown("**💡 Tips:**")
    st.markdown("""
    - **Green (0.7+)**: Excellent match
    - **Yellow (0.4-0.7)**: Good match  
    - **Red (<0.4)**: Poor match
    - Higher scores indicate better alignment between candidate skills and job requirements
    """)

if __name__ == "__main__":
    main()
