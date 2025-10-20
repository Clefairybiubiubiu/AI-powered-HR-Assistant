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

# Semantic matching imports
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

# Enhanced resume parser imports
try:
    from enhanced_resume_parser import parse_resume, get_parser
    ENHANCED_PARSER_AVAILABLE = True
except ImportError:
    ENHANCED_PARSER_AVAILABLE = False

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

def extract_importance_level(text: str) -> float:
    """Extract importance level from text."""
    text_lower = text.lower()
    if "required" in text_lower or "must have" in text_lower:
        return 1.0
    elif "preferred" in text_lower or "plus" in text_lower:
        return 0.75
    elif "nice to have" in text_lower or "optional" in text_lower:
        return 0.5
    return 1.0

class ResumeSemanticMatcher:
    """Semantic matching using Sentence-BERT."""
    
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self.processor = DocumentProcessor()
        self.resumes = {}
        self.job_descriptions = {}
        self.candidate_names = []
        self.jd_names = []
        self.model = None
        self.embeddings_cache = {}
        
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
        
        st.success(f"Loaded {len(self.resumes)} resumes and {len(self.job_descriptions)} job descriptions")
    
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
    
    def extract_sections(self, text: str) -> Dict[str, str]:
        """Extract Education, Skills, and Experience sections from text."""
        sections = {
            'education': '',
            'skills': '',
            'experience': ''
        }
        
        # Clean and normalize the text
        text = self._clean_resume_text(text)
        
        # Split into lines and process
        lines = text.split('\n')
        current_section = None
        
        for i, line in enumerate(lines):
            line_lower = line.lower().strip()
            original_line = line.strip()
            
            # Skip empty lines
            if not original_line:
                continue
            
            # Detect section headers
            section_detected = self._detect_section_header(line_lower)
            if section_detected:
                current_section = section_detected
                continue
            
            # Skip contact information sections
            if self._is_contact_info(line_lower):
                current_section = None
                continue
            
            # Add content to current section
            if current_section and original_line:
                # Filter out unwanted content
                if self._should_include_content(current_section, line_lower, original_line):
                    sections[current_section] += original_line + " "
        
        return sections
    
    def extract_sections_enhanced(self, file_path: str) -> Dict[str, str]:
        """Extract sections using enhanced parser with better format detection."""
        if not ENHANCED_PARSER_AVAILABLE:
            # Fallback to original method
            text = self.processor.extract_text(file_path)
            return self.extract_sections(text)
        
        try:
            # Use enhanced parser
            parsed_data = parse_resume(file_path)
            
            # Use the improved section extraction method
            parser = get_parser()
            sections = parser._extract_sections_from_parsed_data(parsed_data)
            
            # Store parsed data for later use
            self._cached_parsed_data = parsed_data
            
            return sections
            
        except Exception as e:
            print(f"Enhanced parser failed: {e}")
            # Fallback to original method
            text = self.processor.extract_text(file_path)
            return self.extract_sections(text)
    
    def _clean_resume_text(self, text: str) -> str:
        """Clean and normalize resume text for better section detection."""
        # Handle case where text is all on one line (common with PDF extraction)
        if '\n' not in text or len(text.split('\n')) < 3:
            # Try to split on common separators
            text = text.replace('Experience', '\nExperience\n')
            text = text.replace('Skills', '\nSkills\n')
            text = text.replace('Education', '\nEducation\n')
            text = text.replace('•', '\n•')
            text = text.replace('o', '\no')
            text = text.replace('Profile', '\nProfile\n')
            text = text.replace('Contact', '\nContact\n')
        
        return text
    
    def _detect_section_header(self, line_lower: str) -> str:
        """Detect if a line is a section header."""
        # Education section
        if any(word in line_lower for word in ['education', 'academic', 'degree', 'university', 'college', 'qualification']):
            return 'education'
        
        # Skills section
        elif any(word in line_lower for word in ['skill', 'technical', 'competenc', 'expertise', 'proficien', 'technolog', 'tool']):
            return 'skills'
        
        # Experience section
        elif any(word in line_lower for word in ['experience', 'work', 'employment', 'career', 'professional', 'work history', 'employment history']):
            return 'experience'
        
        # Profile/Summary section (reset current section)
        elif any(word in line_lower for word in ['summary', 'objective', 'profile', 'about', 'overview']):
            return None
        
        return None
    
    def _is_contact_info(self, line_lower: str) -> bool:
        """Check if a line contains contact information."""
        contact_keywords = [
            'contact', 'email', 'phone', 'github', 'linkedin', 'website', 
            'address', 'location', 'seattle', 'wa', 'california', 'ca',
            'daniel park', 'data engineer', 'contact daniel'
        ]
        return any(keyword in line_lower for keyword in contact_keywords)
    
    def _should_include_content(self, section: str, line_lower: str, original_line: str) -> bool:
        """Determine if content should be included in a section."""
        # Skip contact information
        if self._is_contact_info(line_lower):
            return False
        
        # Skip very short lines that are likely headers or names
        if len(original_line.split()) <= 2 and not any(word in line_lower for word in ['engineer', 'analyst', 'scientist', 'developer', 'manager']):
            return False
        
        # For experience section, look for job-related content
        if section == 'experience':
            # Look for job titles, dates, or achievements
            job_indicators = [
                'engineer', 'analyst', 'scientist', 'developer', 'manager', 'director',
                'implemented', 'built', 'migrated', 'decreased', 'increased', 'created',
                'designed', 'developed', 'managed', 'led', 'improved', 'optimized'
            ]
            return any(indicator in line_lower for indicator in job_indicators)
        
        # For skills section, look for technical terms
        elif section == 'skills':
            skill_indicators = [
                'python', 'java', 'sql', 'javascript', 'react', 'aws', 'docker',
                'kubernetes', 'spark', 'kafka', 'airflow', 'snowflake', 'bigquery',
                'machine learning', 'data science', 'etl', 'streaming'
            ]
            return any(indicator in line_lower for indicator in skill_indicators)
        
        # For education section, look for degree-related content
        elif section == 'education':
            education_indicators = [
                'bachelor', 'master', 'phd', 'university', 'college', 'degree',
                'diploma', 'ms', 'bs', 'mba', 'computer science', 'data science'
            ]
            return any(indicator in line_lower for indicator in education_indicators)
        
        return True
    
    def extract_jd_requirements_with_importance(self, jd_text: str) -> Dict[str, List[Tuple[str, float]]]:
        """Extract requirements with importance levels from job description."""
        requirements = {
            'education': [],
            'skills': [],
            'experience': []
        }
        
        lines = jd_text.split('\n')
        current_section = None
        
        for line in lines:
            line_lower = line.lower().strip()
            
            # Detect section headers
            if any(word in line_lower for word in ['education', 'degree', 'qualification']):
                current_section = 'education'
                continue
            elif any(word in line_lower for word in ['skill', 'technical', 'requirement']):
                current_section = 'skills'
                continue
            elif any(word in line_lower for word in ['experience', 'work', 'employment']):
                current_section = 'experience'
                continue
            
            # Extract requirements with importance
            if current_section and line.strip():
                importance = extract_importance_level(line)
                requirements[current_section].append((line.strip(), importance))
        
        return requirements
    
    def compute_semantic_similarity(self, education_weight: float = 0.2, skills_weight: float = 0.5, experience_weight: float = 0.3):
        """Compute semantic similarity using Sentence-BERT."""
        if not self.model:
            self.model = load_sentence_transformer()
            if not self.model:
                st.error("SentenceTransformer not available. Please install sentence-transformers.")
                return None
        
        start_time = time.time()
        
        # Extract sections from all documents
        resume_sections = {}
        jd_requirements = {}
        
        for name, text in self.resumes.items():
            # Try enhanced parser first, fallback to original
            if ENHANCED_PARSER_AVAILABLE:
                # Find the original file path for enhanced parsing
                original_path = None
                if hasattr(self, 'original_filenames'):
                    original_path = self.original_filenames.get(name)
                
                if original_path:
                    sections = self.extract_sections_enhanced(str(self.data_dir / original_path))
                    # Store enhanced data for later use
                    if hasattr(self, '_cached_parsed_data'):
                        self._enhanced_data_cache = getattr(self, '_enhanced_data_cache', {})
                        self._enhanced_data_cache[name] = self._cached_parsed_data
                else:
                    sections = self.extract_sections(text)
            else:
                sections = self.extract_sections(text)
            
            resume_sections[name] = sections
            
            # Debug: Show extracted sections with enhanced parser info
            if st.checkbox(f"🔍 Debug sections for {name}", key=f"debug_{name}"):
                st.write(f"**Education:** {sections['education'][:200]}...")
                st.write(f"**Skills:** {sections['skills'][:200]}...")
                st.write(f"**Experience:** {sections['experience'][:200]}...")
                
                # Show enhanced parser confidence if available
                if hasattr(self, '_enhanced_data_cache') and name in self._enhanced_data_cache:
                    enhanced_data = self._enhanced_data_cache[name]
                    if enhanced_data.get('confidence_scores'):
                        st.write("**Enhanced Parser Confidence:**")
                        scores = enhanced_data['confidence_scores']
                        for field, score in scores.items():
                            if score > 0:
                                st.write(f"- {field.title()}: {score:.2f}")
        
        for name, text in self.job_descriptions.items():
            jd_requirements[name] = self.extract_jd_requirements_with_importance(text)
        
        # Compute similarities for each section
        similarities = {}
        
        for jd_name, jd_reqs in jd_requirements.items():
            similarities[jd_name] = {}
            
            for resume_name, resume_sections_data in resume_sections.items():
                section_scores = {}
                total_score = 0
                
                # Education similarity
                if resume_sections_data['education'] and jd_reqs['education']:
                    edu_similarity = self._compute_section_similarity(
                        resume_sections_data['education'], 
                        [req[0] for req in jd_reqs['education']]
                    )
                    section_scores['education'] = edu_similarity
                    total_score += edu_similarity * education_weight
                
                # Skills similarity with importance weighting
                if resume_sections_data['skills'] and jd_reqs['skills']:
                    skills_similarity = self._compute_weighted_section_similarity(
                        resume_sections_data['skills'], 
                        jd_reqs['skills']
                    )
                    section_scores['skills'] = skills_similarity
                    total_score += skills_similarity * skills_weight
                
                # Experience similarity
                if resume_sections_data['experience'] and jd_reqs['experience']:
                    exp_similarity = self._compute_section_similarity(
                        resume_sections_data['experience'], 
                        [req[0] for req in jd_reqs['experience']]
                    )
                    section_scores['experience'] = exp_similarity
                    total_score += exp_similarity * experience_weight
                
                similarities[jd_name][resume_name] = {
                    'total': total_score,
                    'sections': section_scores
                }
        
        # Convert to matrix format
        similarity_matrix = np.zeros((len(self.candidate_names), len(self.jd_names)))
        section_matrices = {
            'education': np.zeros((len(self.candidate_names), len(self.jd_names))),
            'skills': np.zeros((len(self.candidate_names), len(self.jd_names))),
            'experience': np.zeros((len(self.candidate_names), len(self.jd_names)))
        }
        
        for jd_idx, jd_name in enumerate(self.jd_names):
            for resume_idx, resume_name in enumerate(self.candidate_names):
                if jd_name in similarities and resume_name in similarities[jd_name]:
                    similarity_matrix[resume_idx, jd_idx] = similarities[jd_name][resume_name]['total']
                    
                    for section in ['education', 'skills', 'experience']:
                        if section in similarities[jd_name][resume_name]['sections']:
                            section_matrices[section][resume_idx, jd_idx] = similarities[jd_name][resume_name]['sections'][section]
        
        end_time = time.time()
        st.info(f"🧠 Semantic matching completed in {end_time - start_time:.2f} seconds")
        
        return similarity_matrix, section_matrices, similarities
    
    def _compute_section_similarity(self, resume_text: str, jd_requirements: List[str]) -> float:
        """Compute similarity between resume section and JD requirements."""
        if not resume_text.strip() or not jd_requirements:
            return 0.0
        
        # Combine JD requirements
        jd_text = " ".join(jd_requirements)
        
        # Get embeddings
        resume_embedding = self._get_embedding(resume_text)
        jd_embedding = self._get_embedding(jd_text)
        
        # Compute cosine similarity
        similarity = cosine_similarity([resume_embedding], [jd_embedding])[0][0]
        return float(similarity)
    
    def _compute_weighted_section_similarity(self, resume_text: str, jd_requirements: List[Tuple[str, float]]) -> float:
        """Compute weighted similarity for skills section."""
        if not resume_text.strip() or not jd_requirements:
            return 0.0
        
        resume_embedding = self._get_embedding(resume_text)
        
        # Compute weighted similarity
        total_weight = 0
        weighted_similarity = 0
        
        for req_text, importance in jd_requirements:
            req_embedding = self._get_embedding(req_text)
            similarity = cosine_similarity([resume_embedding], [req_embedding])[0][0]
            weighted_similarity += similarity * importance
            total_weight += importance
        
        return float(weighted_similarity / total_weight) if total_weight > 0 else 0.0
    
    def _get_embedding(self, text: str) -> np.ndarray:
        """Get embedding for text with caching."""
        cache_key = hash(text)
        if cache_key in self.embeddings_cache:
            return self.embeddings_cache[cache_key]
        
        embedding = self.model.encode(text)
        self.embeddings_cache[cache_key] = embedding
        return embedding
    
    def generate_explanation(self, resume_name: str, jd_name: str, similarities: Dict) -> str:
        """Generate natural language explanation for match."""
        if jd_name not in similarities or resume_name not in similarities[jd_name]:
            return "No match data available."
        
        match_data = similarities[jd_name][resume_name]
        sections = match_data['sections']
        
        # Find strongest sections
        strong_sections = []
        for section, score in sections.items():
            if score > 0.3:  # Threshold for "strong" match
                strong_sections.append(section)
        
        if not strong_sections:
            return f"Limited alignment between {resume_name} and {jd_name} requirements."
        
        if len(strong_sections) == 1:
            return f"Strong match in {strong_sections[0]} with {sections[strong_sections[0]]:.2f} similarity."
        else:
            return f"Strong alignment across {', '.join(strong_sections)} with average similarity of {np.mean([sections[s] for s in strong_sections]):.2f}."

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
    
    # Mode selection
    matching_mode = st.sidebar.selectbox(
        "🎯 Matching Mode",
        ["📊 TF-IDF Mode", "🧠 Semantic (Sentence-BERT) Mode"],
        help="Choose between traditional TF-IDF or advanced semantic matching"
    )
    
    # Show mode indicator
    if "Semantic" in matching_mode:
        st.sidebar.success("🧠 **Semantic Mode Active**")
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            st.sidebar.error("⚠️ SentenceTransformers not installed. Please run: `pip install sentence-transformers`")
    else:
        st.sidebar.info("📊 **TF-IDF Mode Active**")
    
    # Semantic mode controls
    if "Semantic" in matching_mode:
        st.sidebar.subheader("🎛️ Weight Controls")
        education_weight = st.sidebar.slider(
            "Education Weight",
            min_value=0.0,
            max_value=1.0,
            value=0.2,
            step=0.05,
            help="Weight for education section matching"
        )
        skills_weight = st.sidebar.slider(
            "Skills Weight",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.05,
            help="Weight for skills section matching"
        )
        experience_weight = st.sidebar.slider(
            "Experience Weight",
            min_value=0.0,
            max_value=1.0,
            value=0.3,
            step=0.05,
            help="Weight for experience section matching"
        )
        
        # Normalize weights
        total_weight = education_weight + skills_weight + experience_weight
        if total_weight > 0:
            education_weight /= total_weight
            skills_weight /= total_weight
            experience_weight /= total_weight
    
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
        if 'section_matrices' in st.session_state:
            del st.session_state.section_matrices
        if 'similarities' in st.session_state:
            del st.session_state.similarities
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
    
    # Enhanced Parser Status
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🔍 Enhanced Parser Status")
    
    if ENHANCED_PARSER_AVAILABLE:
        st.sidebar.success("✅ Enhanced Parser Available")
        st.sidebar.write("**Features:**")
        st.sidebar.write("• spaCy NER for entity recognition")
        st.sidebar.write("• pyresparser for structured extraction")
        st.sidebar.write("• Fallback regex patterns")
        st.sidebar.write("• Confidence scoring")
    else:
        st.sidebar.error("❌ Enhanced Parser Not Available")
        st.sidebar.write("**To enable:**")
        st.sidebar.write("```bash")
        st.sidebar.write("pip install spacy pyresparser nltk")
        st.sidebar.write("python -m spacy download en_core_web_sm")
        st.sidebar.write("```")
    
    # Initialize matcher based on mode
    if st.sidebar.button("🔄 Load Documents") or 'matcher' not in st.session_state:
        with st.spinner("Loading documents..."):
            if "Semantic" in matching_mode:
                matcher = ResumeSemanticMatcher(data_dir)
            else:
                matcher = ResumeJDMatcher(data_dir)
            
            matcher.load_documents()
            
            if matcher.resumes and matcher.job_descriptions:
                st.session_state.matcher = matcher
                st.session_state.matching_mode = matching_mode
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
            if "Semantic" in matching_mode:
                with st.spinner("Computing semantic similarity with Sentence-BERT..."):
                    result = matcher.compute_semantic_similarity(
                        education_weight=education_weight,
                        skills_weight=skills_weight,
                        experience_weight=experience_weight
                    )
                    
                    if result is not None:
                        similarity_matrix, section_matrices, similarities = result
                        st.session_state.similarity_matrix = similarity_matrix
                        st.session_state.section_matrices = section_matrices
                        st.session_state.similarities = similarities
                        st.success("✅ Semantic similarity computation completed!")
                    else:
                        st.error("❌ Failed to compute semantic similarity!")
            else:
                with st.spinner("Computing TF-IDF and cosine similarity..."):
                    similarity_matrix = matcher.compute_similarity()
                    
                    if similarity_matrix is not None:
                        st.session_state.similarity_matrix = similarity_matrix
                        st.success("✅ TF-IDF similarity computation completed!")
                    else:
                        st.error("❌ Failed to compute similarity!")
    
    # Results section
    if 'similarity_matrix' in st.session_state and st.session_state.similarity_matrix is not None:
        st.markdown("---")
        
        # Mode indicator
        current_mode = st.session_state.get('matching_mode', 'TF-IDF')
        if "Semantic" in current_mode:
            st.subheader("🧠 Semantic Results Dashboard")
        else:
            st.subheader("📊 TF-IDF Results Dashboard")
        
        similarity_matrix = st.session_state.similarity_matrix
        
        # Export functionality
        col1, col2 = st.columns([3, 1])
        with col2:
            if st.button("📥 Export Results", help="Download results as CSV"):
                # Create exportable dataframe
                export_data = []
                for jd_idx, jd_name in enumerate(matcher.jd_names):
                    for resume_idx, resume_name in enumerate(matcher.candidate_names):
                        row = {
                            'Job Description': jd_name,
                            'Candidate': resume_name,
                            'Total Similarity': similarity_matrix[resume_idx, jd_idx],
                            'Match Percentage': f"{similarity_matrix[resume_idx, jd_idx] * 100:.1f}%"
                        }
                        
                        # Add section scores for semantic mode
                        if "Semantic" in current_mode and 'section_matrices' in st.session_state:
                            section_matrices = st.session_state.section_matrices
                            for section in ['education', 'skills', 'experience']:
                                if section in section_matrices:
                                    row[f'{section.title()} Similarity'] = section_matrices[section][resume_idx, jd_idx]
                        
                        # Add explanation for semantic mode
                        if "Semantic" in current_mode and 'similarities' in st.session_state:
                            similarities = st.session_state.similarities
                            if jd_name in similarities and resume_name in similarities[jd_name]:
                                explanation = matcher.generate_explanation(resume_name, jd_name, similarities)
                                row['Explanation'] = explanation
                        
                        export_data.append(row)
                
                export_df = pd.DataFrame(export_data)
                csv = export_df.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name=f"resume_matching_results_{current_mode.lower().replace(' ', '_')}.csv",
                    mime="text/csv"
                )
        
        # Create tabs for different views
        if "Semantic" in current_mode:
            tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(["🔥 Heatmap", "📈 Top Matches", "📋 Detailed Scores", "🎯 Section Breakdown", "👤 Resume Details", "💼 Job Requirements", "🔍 Enhanced Parser"])
        else:
            tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["🔥 Heatmap", "📈 Top Matches", "📋 Detailed Scores", "👤 Resume Details", "💼 Job Requirements", "🔍 Enhanced Parser"])
        
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
        
        # Section breakdown tab (only for semantic mode)
        if "Semantic" in current_mode and 'section_matrices' in st.session_state:
            with tab4:
                st.subheader("🎯 Section Breakdown")
                
                section_matrices = st.session_state.section_matrices
                
                # Create section heatmaps
                for section_name, section_matrix in section_matrices.items():
                    if section_matrix.max() > 0:  # Only show sections with data
                        st.write(f"**{section_name.title()} Similarity**")
                        fig, _ = create_heatmap(section_matrix, matcher.candidate_names, matcher.jd_names)
                        fig.update_layout(title=f"{section_name.title()} Similarity Heatmap")
                        st.plotly_chart(fig, use_container_width=True)
                        st.write("---")
                
                # Detailed section scores table
                st.subheader("📊 Detailed Section Scores")
                section_data = []
                for jd_idx, jd_name in enumerate(matcher.jd_names):
                    for resume_idx, resume_name in enumerate(matcher.candidate_names):
                        row = {
                            'Job Description': jd_name,
                            'Candidate': resume_name,
                            'Total Score': similarity_matrix[resume_idx, jd_idx]
                        }
                        
                        for section_name, section_matrix in section_matrices.items():
                            row[f'{section_name.title()} Score'] = section_matrix[resume_idx, jd_idx]
                        
                        # Add explanation
                        if 'similarities' in st.session_state:
                            similarities = st.session_state.similarities
                            if jd_name in similarities and resume_name in similarities[jd_name]:
                                explanation = matcher.generate_explanation(resume_name, jd_name, similarities)
                                row['Explanation'] = explanation
                        
                        section_data.append(row)
                
                section_df = pd.DataFrame(section_data)
                st.dataframe(section_df, use_container_width=True)
        
        # Enhanced Parser tab
        if "Semantic" in current_mode:
            with tab7:
                st.subheader("🔍 Enhanced Parser Results")
                
                if ENHANCED_PARSER_AVAILABLE:
                    st.success("✅ Enhanced Parser Available")
                    
                    # Select candidate to view enhanced parsing
                    selected_candidate = st.selectbox(
                        "Select a candidate to view enhanced parsing:",
                        matcher.candidate_names,
                        key="enhanced_parser_selector"
                    )
                    
                    if selected_candidate and selected_candidate in matcher.resumes:
                        # Get original file path
                        original_path = None
                        if hasattr(matcher, 'original_filenames'):
                            original_path = matcher.original_filenames.get(selected_candidate)
                        
                        if original_path:
                            try:
                                # Parse with enhanced parser
                                parsed_data = parse_resume(str(matcher.data_dir / original_path))
                                
                                # Display parsed data
                                col1, col2 = st.columns(2)
                                
                                with col1:
                                    st.markdown("### 📋 Basic Information")
                                    st.write(f"**Name:** {parsed_data.get('name', 'N/A')}")
                                    st.write(f"**Email:** {parsed_data.get('email', 'N/A')}")
                                    st.write(f"**Phone:** {parsed_data.get('phone', 'N/A')}")
                                    
                                    st.markdown("### 🎓 Education")
                                    if parsed_data.get('education'):
                                        for edu in parsed_data['education']:
                                            st.write(f"• {edu}")
                                    else:
                                        st.write("No education data found")
                                
                                with col2:
                                    st.markdown("### 🛠️ Skills")
                                    if parsed_data.get('skills'):
                                        skills_text = ", ".join(parsed_data['skills'])
                                        st.write(skills_text)
                                    else:
                                        st.write("No skills data found")
                                    
                                    st.markdown("### 💼 Experience")
                                    if parsed_data.get('experience'):
                                        for exp in parsed_data['experience']:
                                            st.write(f"• {exp}")
                                    else:
                                        st.write("No experience data found")
                                
                                # Display entities
                                if parsed_data.get('entities'):
                                    st.markdown("### 🏷️ Extracted Entities")
                                    entities = parsed_data['entities']
                                    
                                    col3, col4 = st.columns(2)
                                    
                                    with col3:
                                        if entities.get('organizations'):
                                            st.write(f"**Organizations:** {', '.join(entities['organizations'])}")
                                        if entities.get('locations'):
                                            st.write(f"**Locations:** {', '.join(entities['locations'])}")
                                    
                                    with col4:
                                        if entities.get('persons'):
                                            st.write(f"**Persons:** {', '.join(entities['persons'])}")
                                        if entities.get('dates'):
                                            st.write(f"**Dates:** {', '.join(entities['dates'])}")
                                
                                # Display confidence scores
                                if parsed_data.get('confidence_scores'):
                                    st.markdown("### 📊 Confidence Scores")
                                    scores = parsed_data['confidence_scores']
                                    for field, score in scores.items():
                                        st.write(f"**{field.title()}:** {score:.2f}")
                                
                            except Exception as e:
                                st.error(f"Enhanced parsing failed: {e}")
                        else:
                            st.warning("Original file path not found for enhanced parsing")
                else:
                    st.error("❌ Enhanced Parser Not Available")
                    st.write("Please install required packages:")
                    st.code("pip install spacy pyresparser nltk")
                    st.write("Then download spaCy model:")
                    st.code("python -m spacy download en_core_web_sm")
        
        # Resume Details tab
        if "Semantic" in current_mode:
            with tab5:
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
        else:
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
                    
                    # Try to get enhanced parser data if available
                    enhanced_data = None
                    if ENHANCED_PARSER_AVAILABLE and hasattr(matcher, '_cached_parsed_data'):
                        enhanced_data = matcher._cached_parsed_data
                    
                    # Display resume summary
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("### 📋 Contact Information")
                        # Use enhanced parser data if available
                        if enhanced_data and enhanced_data.get('name'):
                            st.write(f"**Name:** {enhanced_data['name']}")
                        else:
                            st.write(f"**Name:** {summary['name']}")
                        
                        if enhanced_data and enhanced_data.get('email'):
                            st.write(f"**Email:** {enhanced_data['email']}")
                        elif summary['email']:
                            st.write(f"**Email:** {summary['email']}")
                        
                        if enhanced_data and enhanced_data.get('phone'):
                            st.write(f"**Phone:** {enhanced_data['phone']}")
                        elif summary['phone']:
                            st.write(f"**Phone:** {summary['phone']}")
                        
                        if summary['location']:
                            st.write(f"**Location:** {summary['location']}")
                        
                        # Show enhanced parser confidence scores
                        if enhanced_data and enhanced_data.get('confidence_scores'):
                            st.markdown("### 📊 Extraction Confidence")
                            scores = enhanced_data['confidence_scores']
                            for field, score in scores.items():
                                if score > 0:
                                    st.write(f"**{field.title()}:** {score:.2f}")
                    
                    with col2:
                        st.markdown("### 🎯 Professional Summary")
                        if summary['summary']:
                            st.write(summary['summary'])
                        else:
                            st.write("No summary available")
                        
                        # Show enhanced parser entities if available
                        if enhanced_data and enhanced_data.get('entities'):
                            st.markdown("### 🏷️ Extracted Entities")
                            entities = enhanced_data['entities']
                            if entities.get('organizations'):
                                st.write(f"**Organizations:** {', '.join(entities['organizations'])}")
                            if entities.get('locations'):
                                st.write(f"**Locations:** {', '.join(entities['locations'])}")
                    
                    # Skills section - use enhanced parser data if available
                    skills_to_show = []
                    if enhanced_data and enhanced_data.get('skills'):
                        skills_to_show = enhanced_data['skills']
                    elif summary['skills']:
                        skills_to_show = summary['skills']
                    
                    if skills_to_show:
                        st.markdown("### 🛠️ Skills")
                        if isinstance(skills_to_show, list):
                            skills_text = ", ".join(skills_to_show[:10])  # Show first 10 skills
                            st.write(skills_text)
                            if len(skills_to_show) > 10:
                                st.write(f"... and {len(skills_to_show) - 10} more skills")
                        else:
                            st.write(skills_to_show)
                    
                    # Experience section - use enhanced parser data if available
                    experience_to_show = []
                    if enhanced_data and enhanced_data.get('experience'):
                        experience_to_show = enhanced_data['experience']
                    elif summary['experience']:
                        experience_to_show = summary['experience']
                    
                    if experience_to_show:
                        st.markdown("### 💼 Experience")
                        if isinstance(experience_to_show, list):
                            for exp in experience_to_show[:5]:  # Show first 5 experiences
                                st.write(f"• {exp}")
                        else:
                            st.write(experience_to_show)
                    
                    # Education section - use enhanced parser data if available
                    education_to_show = []
                    if enhanced_data and enhanced_data.get('education'):
                        education_to_show = enhanced_data['education']
                    elif summary['education']:
                        education_to_show = summary['education']
                    
                    if education_to_show:
                        st.markdown("### 🎓 Education")
                        if isinstance(education_to_show, list):
                            for edu in education_to_show[:3]:  # Show first 3 education entries
                                st.write(f"• {edu}")
                        else:
                            st.write(education_to_show)
                    
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
        
        # Job Requirements tab
        if "Semantic" in current_mode:
            with tab6:
                st.subheader("💼 Job Requirements Dashboard")
                
                # Job description selection
                selected_jd = st.selectbox(
                    "Select a job description to view requirements:",
                    matcher.jd_names,
                    key="jd_selector"
                )
        else:
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
