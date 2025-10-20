#!/usr/bin/env python3
"""
Enhanced Resume-JD Matcher using pyresparser for structured extraction.
Organizes content into specific sections and performs TF-IDF and semantic matching.
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import re
import time
from pathlib import Path
from typing import List, Dict, Tuple, Any
import plotly.express as px
import plotly.graph_objects as go
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')

# Pyresparser imports
try:
    from pyresparser import ResumeParser
    PYRESPARSER_AVAILABLE = True
except ImportError:
    PYRESPARSER_AVAILABLE = False
    st.error("❌ pyresparser not available. Install with: pip install pyresparser")

# Semantic matching imports
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

@st.cache_resource
def load_sentence_transformer():
    """Load and cache the SentenceTransformer model."""
    if not SENTENCE_TRANSFORMERS_AVAILABLE:
        return None
    try:
        model = SentenceTransformer('all-MiniLM-L6-v2')
        return model
    except Exception as e:
        st.error(f"Failed to load SentenceTransformer: {e}")
        return None

class PyresparserDocumentProcessor:
    """Document processor using pyresparser for structured extraction."""
    
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self.resumes = {}
        self.job_descriptions = {}
        self.resume_sections = {}
        self.jd_sections = {}
        self.candidate_names = []
        self.jd_names = []
        
    def load_documents(self):
        """Load and parse all documents using pyresparser."""
        if not PYRESPARSER_AVAILABLE:
            st.error("❌ pyresparser not available. Please install: pip install pyresparser")
            return
        
        # Load resumes
        resume_files = [f for f in self.data_dir.glob("*") if f.is_file() and not f.name.startswith("JD")]
        jd_files = [f for f in self.data_dir.glob("*") if f.is_file() and f.name.startswith("JD")]
        
        st.info(f"Found {len(resume_files)} resume files and {len(jd_files)} JD files")
        
        # Process resumes
        for file_path in resume_files:
            try:
                st.write(f"Processing resume: {file_path.name}")
                parsed_data = self.parse_resume_with_pyresparser(file_path)
                if parsed_data:
                    # Extract candidate name
                    candidate_name = self.extract_candidate_name(parsed_data, file_path.name)
                    self.candidate_names.append(candidate_name)
                    
                    # Organize into sections
                    sections = self.organize_resume_sections(parsed_data)
                    self.resume_sections[candidate_name] = sections
                    self.resumes[candidate_name] = self.create_resume_text(sections)
                    
            except Exception as e:
                st.error(f"Error processing {file_path.name}: {e}")
        
        # Process job descriptions
        for file_path in jd_files:
            try:
                st.write(f"Processing JD: {file_path.name}")
                parsed_data = self.parse_jd_with_pyresparser(file_path)
                if parsed_data:
                    jd_name = file_path.stem
                    self.jd_names.append(jd_name)
                    
                    # Organize into sections
                    sections = self.organize_jd_sections(parsed_data)
                    self.jd_sections[jd_name] = sections
                    self.job_descriptions[jd_name] = self.create_jd_text(sections)
                    
            except Exception as e:
                st.error(f"Error processing {file_path.name}: {e}")
    
    def parse_resume_with_pyresparser(self, file_path: Path) -> Dict[str, Any]:
        """Parse resume using pyresparser."""
        try:
            parser = ResumeParser(str(file_path))
            parsed_data = parser.get_extracted_data()
            return parsed_data
        except Exception as e:
            st.error(f"Pyresparser failed for {file_path.name}: {e}")
            return None
    
    def parse_jd_with_pyresparser(self, file_path: Path) -> Dict[str, Any]:
        """Parse job description using pyresparser."""
        try:
            # For JDs, we'll use pyresparser but adapt the structure
            parser = ResumeParser(str(file_path))
            parsed_data = parser.get_extracted_data()
            return parsed_data
        except Exception as e:
            st.error(f"Pyresparser failed for {file_path.name}: {e}")
            return None
    
    def extract_candidate_name(self, parsed_data: Dict[str, Any], filename: str) -> str:
        """Extract candidate name from parsed data."""
        name = parsed_data.get('name', '')
        if name:
            # Clean the name
            name = re.sub(r'[^\w\s]', '', name).strip()
            if name:
                return f"{name}-resume"
        
        # Fallback to filename
        return f"Candidate-{filename.split('.')[0]}"
    
    def organize_resume_sections(self, parsed_data: Dict[str, Any]) -> Dict[str, str]:
        """Organize resume data into specific sections."""
        sections = {
            'contact_information': '',
            'professional_summary': '',
            'education': '',
            'skills': '',
            'experience': '',
            'certifications': ''
        }
        
        # Contact Information
        contact_parts = []
        if parsed_data.get('name'):
            contact_parts.append(f"Name: {parsed_data['name']}")
        if parsed_data.get('email'):
            contact_parts.append(f"Email: {parsed_data['email']}")
        if parsed_data.get('mobile_number'):
            contact_parts.append(f"Phone: {parsed_data['mobile_number']}")
        if parsed_data.get('location'):
            contact_parts.append(f"Location: {parsed_data['location']}")
        sections['contact_information'] = ' | '.join(contact_parts)
        
        # Professional Summary
        if parsed_data.get('summary'):
            sections['professional_summary'] = parsed_data['summary']
        
        # Education
        if parsed_data.get('education'):
            if isinstance(parsed_data['education'], list):
                sections['education'] = ' | '.join(parsed_data['education'])
            else:
                sections['education'] = str(parsed_data['education'])
        
        # Skills
        if parsed_data.get('skills'):
            if isinstance(parsed_data['skills'], list):
                sections['skills'] = ' | '.join(parsed_data['skills'])
            else:
                sections['skills'] = str(parsed_data['skills'])
        
        # Experience
        if parsed_data.get('experience'):
            if isinstance(parsed_data['experience'], list):
                sections['experience'] = ' | '.join(parsed_data['experience'])
            else:
                sections['experience'] = str(parsed_data['experience'])
        
        # Certifications
        if parsed_data.get('certifications'):
            if isinstance(parsed_data['certifications'], list):
                sections['certifications'] = ' | '.join(parsed_data['certifications'])
            else:
                sections['certifications'] = str(parsed_data['certifications'])
        
        return sections
    
    def organize_jd_sections(self, parsed_data: Dict[str, Any]) -> Dict[str, str]:
        """Organize job description data into specific sections."""
        sections = {
            'job_information': '',
            'job_overview': '',
            'required_skills': '',
            'preferred_skills': '',
            'responsibilities': '',
            'requirements': ''
        }
        
        # Job Information
        job_info_parts = []
        if parsed_data.get('name'):
            job_info_parts.append(f"Position: {parsed_data['name']}")
        if parsed_data.get('location'):
            job_info_parts.append(f"Location: {parsed_data['location']}")
        sections['job_information'] = ' | '.join(job_info_parts)
        
        # Job Overview
        if parsed_data.get('summary'):
            sections['job_overview'] = parsed_data['summary']
        
        # Required Skills
        if parsed_data.get('skills'):
            if isinstance(parsed_data['skills'], list):
                sections['required_skills'] = ' | '.join(parsed_data['skills'])
            else:
                sections['required_skills'] = str(parsed_data['skills'])
        
        # Preferred Skills (extract from text)
        if parsed_data.get('experience'):
            if isinstance(parsed_data['experience'], list):
                sections['preferred_skills'] = ' | '.join(parsed_data['experience'])
            else:
                sections['preferred_skills'] = str(parsed_data['experience'])
        
        # Responsibilities
        if parsed_data.get('responsibilities'):
            if isinstance(parsed_data['responsibilities'], list):
                sections['responsibilities'] = ' | '.join(parsed_data['responsibilities'])
            else:
                sections['responsibilities'] = str(parsed_data['responsibilities'])
        
        # Requirements
        if parsed_data.get('requirements'):
            if isinstance(parsed_data['requirements'], list):
                sections['requirements'] = ' | '.join(parsed_data['requirements'])
            else:
                sections['requirements'] = str(parsed_data['requirements'])
        
        return sections
    
    def create_resume_text(self, sections: Dict[str, str]) -> str:
        """Create full resume text from sections."""
        text_parts = []
        for section_name, content in sections.items():
            if content:
                text_parts.append(f"{section_name.replace('_', ' ').title()}: {content}")
        return ' '.join(text_parts)
    
    def create_jd_text(self, sections: Dict[str, str]) -> str:
        """Create full JD text from sections."""
        text_parts = []
        for section_name, content in sections.items():
            if content:
                text_parts.append(f"{section_name.replace('_', ' ').title()}: {content}")
        return ' '.join(text_parts)

class PyresparserTFIDFMatcher:
    """TF-IDF matcher using pyresparser structured data."""
    
    def __init__(self, processor: PyresparserDocumentProcessor):
        self.processor = processor
        
    def compute_similarity(self) -> np.ndarray:
        """Compute TF-IDF similarity between resumes and JDs."""
        if not self.processor.resumes or not self.processor.job_descriptions:
            return np.array([])
        
        # Prepare documents
        resume_texts = list(self.processor.resumes.values())
        jd_texts = list(self.processor.job_descriptions.values())
        
        # TF-IDF vectorization
        vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        
        # Fit on all documents
        all_texts = resume_texts + jd_texts
        vectorizer.fit(all_texts)
        
        # Transform documents
        resume_vectors = vectorizer.transform(resume_texts)
        jd_vectors = vectorizer.transform(jd_texts)
        
        # Compute similarity
        similarity_matrix = cosine_similarity(resume_vectors, jd_vectors)
        
        return similarity_matrix

class PyresparserSemanticMatcher:
    """Semantic matcher using pyresparser structured data."""
    
    def __init__(self, processor: PyresparserDocumentProcessor):
        self.processor = processor
        self.model = load_sentence_transformer()
        self.embedding_cache = {}
        
    def compute_similarity(self) -> np.ndarray:
        """Compute semantic similarity between resumes and JDs."""
        if not self.processor.resumes or not self.processor.job_descriptions:
            return np.array([])
        
        if not self.model:
            st.error("❌ SentenceTransformer not available")
            return np.array([])
        
        # Prepare documents
        resume_texts = list(self.processor.resumes.values())
        jd_texts = list(self.processor.job_descriptions.values())
        
        # Get embeddings
        resume_embeddings = self._get_embeddings(resume_texts)
        jd_embeddings = self._get_embeddings(jd_texts)
        
        # Compute similarity
        similarity_matrix = cosine_similarity(resume_embeddings, jd_embeddings)
        
        return similarity_matrix
    
    def _get_embeddings(self, texts: List[str]) -> np.ndarray:
        """Get embeddings for texts with caching."""
        embeddings = []
        for text in texts:
            text_hash = hash(text)
            if text_hash in self.embedding_cache:
                embeddings.append(self.embedding_cache[text_hash])
            else:
                embedding = self.model.encode([text])[0]
                self.embedding_cache[text_hash] = embedding
                embeddings.append(embedding)
        return np.array(embeddings)

def main():
    """Main Streamlit application."""
    st.set_page_config(
        page_title="Pyresparser Resume-JD Matcher",
        page_icon="🎯",
        layout="wide"
    )
    
    st.title("🎯 Pyresparser Resume-JD Matcher")
    st.markdown("**Structured extraction and matching using pyresparser**")
    
    # Sidebar
    st.sidebar.header("⚙️ Configuration")
    
    # Data directory
    data_dir = st.sidebar.text_input(
        "📁 Data Directory",
        value="/Users/junfeibai/Desktop/5560/test",
        help="Directory containing resume and JD files"
    )
    
    # Matching mode
    matching_mode = st.sidebar.selectbox(
        "🔍 Matching Mode",
        ["TF-IDF Mode", "Semantic (Sentence-BERT) Mode"],
        help="Choose between TF-IDF and semantic matching"
    )
    
    # Pyresparser status
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🔧 Pyresparser Status")
    
    if PYRESPARSER_AVAILABLE:
        st.sidebar.success("✅ Pyresparser Available")
        st.sidebar.write("**Features:**")
        st.sidebar.write("• Structured resume parsing")
        st.sidebar.write("• Contact information extraction")
        st.sidebar.write("• Skills and experience detection")
        st.sidebar.write("• Education and certification parsing")
    else:
        st.sidebar.error("❌ Pyresparser Not Available")
        st.sidebar.write("**To enable:**")
        st.sidebar.write("```bash")
        st.sidebar.write("pip install pyresparser")
        st.sidebar.write("```")
    
    # Load documents
    if st.sidebar.button("🔄 Load Documents") or 'processor' not in st.session_state:
        if not PYRESPARSER_AVAILABLE:
            st.error("❌ pyresparser not available. Please install: pip install pyresparser")
            return
        
        with st.spinner("Loading documents with pyresparser..."):
            processor = PyresparserDocumentProcessor(data_dir)
            processor.load_documents()
            st.session_state.processor = processor
            
            # Initialize matcher based on mode
            if "Semantic" in matching_mode:
                if not SENTENCE_TRANSFORMERS_AVAILABLE:
                    st.error("❌ SentenceTransformer not available. Install: pip install sentence-transformers")
                    return
                matcher = PyresparserSemanticMatcher(processor)
            else:
                matcher = PyresparserTFIDFMatcher(processor)
            
            st.session_state.matcher = matcher
    
    if 'processor' not in st.session_state:
        st.info("👆 Click 'Load Documents' to start")
        return
    
    processor = st.session_state.processor
    matcher = st.session_state.matcher
    
    # Display document information
    st.markdown("### 📊 Document Overview")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("📄 Resumes", len(processor.resumes))
    with col2:
        st.metric("📋 Job Descriptions", len(processor.job_descriptions))
    with col3:
        st.metric("🔍 Mode", matching_mode)
    
    # Compute similarity
    if st.button("🎯 Compute Similarity"):
        with st.spinner("Computing similarity..."):
            similarity_matrix = matcher.compute_similarity()
            st.session_state.similarity_matrix = similarity_matrix
    
    if 'similarity_matrix' not in st.session_state:
        st.info("👆 Click 'Compute Similarity' to start matching")
        return
    
    similarity_matrix = st.session_state.similarity_matrix
    
    # Display results
    st.markdown("### 📈 Similarity Results")
    
    # Create results DataFrame
    results_data = []
    for i, resume_name in enumerate(processor.candidate_names):
        for j, jd_name in enumerate(processor.jd_names):
            results_data.append({
                'Resume': resume_name,
                'Job Description': jd_name,
                'Similarity': similarity_matrix[i][j]
            })
    
    results_df = pd.DataFrame(results_data)
    
    # Display heatmap
    st.markdown("#### 🔥 Similarity Heatmap")
    pivot_df = results_df.pivot(index='Resume', columns='Job Description', values='Similarity')
    
    fig = px.imshow(
        pivot_df.values,
        x=pivot_df.columns,
        y=pivot_df.index,
        color_continuous_scale='RdYlBu_r',
        aspect='auto',
        title="Resume-JD Similarity Matrix"
    )
    fig.update_layout(
        xaxis_title="Job Descriptions",
        yaxis_title="Resumes"
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Top matches
    st.markdown("#### 🏆 Top Matches")
    top_matches = results_df.nlargest(10, 'Similarity')
    st.dataframe(top_matches, use_container_width=True)
    
    # Detailed sections
    st.markdown("### 📋 Detailed Sections")
    
    # Resume sections
    st.markdown("#### 👤 Resume Sections")
    selected_resume = st.selectbox("Select Resume:", processor.candidate_names)
    if selected_resume in processor.resume_sections:
        sections = processor.resume_sections[selected_resume]
        for section_name, content in sections.items():
            if content:
                st.markdown(f"**{section_name.replace('_', ' ').title()}:**")
                st.write(content)
                st.markdown("---")
    
    # JD sections
    st.markdown("#### 📋 Job Description Sections")
    selected_jd = st.selectbox("Select Job Description:", processor.jd_names)
    if selected_jd in processor.jd_sections:
        sections = processor.jd_sections[selected_jd]
        for section_name, content in sections.items():
            if content:
                st.markdown(f"**{section_name.replace('_', ' ').title()}:**")
                st.write(content)
                st.markdown("---")

if __name__ == "__main__":
    main()
