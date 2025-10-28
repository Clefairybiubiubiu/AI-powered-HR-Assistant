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
    def normalize_text(text: str) -> str:
        """Normalize text for consistent parsing."""
        import re
        
        # 1. First, normalize line breaks and preserve structure
        text = re.sub(r'\r\n', '\n', text)  # Windows line endings
        text = re.sub(r'\r', '\n', text)     # Mac line endings
        
        # 2. Insert line breaks before section headers (before removing spaces)
        section_headers = [
            'PROFESSIONAL SUMMARY', 'PROFILE', 'SUMMARY', 'OBJECTIVE',
            'EDUCATION', 'EXPERIENCE', 'WORK EXPERIENCE', 'EMPLOYMENT',
            'SKILLS', 'TECHNICAL SKILLS', 'COMPETENCIES', 'EXPERTISE',
            'CERTIFICATIONS', 'PROJECTS', 'ACHIEVEMENTS', 'AWARDS',
            'PUBLICATIONS', 'LANGUAGES', 'INTERESTS', 'REFERENCES',
            'CONTACT', 'PERSONAL INFORMATION', 'CAREER OBJECTIVE',
            'PROFESSIONAL EXPERIENCE', 'ACADEMIC BACKGROUND'
        ]
        
        for header in section_headers:
            # Insert line break before headers (case insensitive)
            pattern = r'(?<!\n)(' + re.escape(header) + r')(?!\n)'
            text = re.sub(pattern, r'\n\1', text, flags=re.IGNORECASE)
        
        # Also handle mixed case headers
        mixed_case_headers = [
            'Professional Summary', 'Profile', 'Summary', 'Objective',
            'Education', 'Experience', 'Work Experience', 'Employment',
            'Skills', 'Technical Skills', 'Competencies', 'Expertise',
            'Certifications', 'Projects', 'Achievements', 'Awards',
            'Publications', 'Languages', 'Interests', 'References',
            'Contact', 'Personal Information', 'Career Objective',
            'Professional Experience', 'Academic Background'
        ]
        
        for header in mixed_case_headers:
            pattern = r'(?<!\n)(' + re.escape(header) + r')(?!\n)'
            text = re.sub(pattern, r'\n\1', text)
        
        # 3. Remove bullet symbols for consistent parsing
        bullet_symbols = ['•', '–', '○', '▪', '▫', '‣', '⁃', '◦', '‥', '…']
        for bullet in bullet_symbols:
            text = text.replace(bullet, '')
        
        # 4. Normalize spaces within lines (but preserve line breaks)
        lines = text.split('\n')
        normalized_lines = []
        for line in lines:
            # Replace multiple spaces/tabs with single space within each line
            normalized_line = re.sub(r'[ \t]+', ' ', line.strip())
            normalized_lines.append(normalized_line)
        
        text = '\n'.join(normalized_lines)
        
        # 5. Clean up multiple consecutive empty lines
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
        
        return text.strip()
    
    @staticmethod
    def detect_encoding(file_path: str) -> str:
        """Detect file encoding."""
        try:
            import chardet
            with open(file_path, 'rb') as file:
                raw_data = file.read()
                result = chardet.detect(raw_data)
                encoding = result['encoding']
                confidence = result['confidence']
                
                # If confidence is low, fall back to common encodings
                if confidence < 0.7:
                    return 'utf-8'
                
                return encoding if encoding else 'utf-8'
        except Exception:
            return 'utf-8'
    
    @staticmethod
    def extract_text_from_txt(file_path: str) -> str:
        """Extract text from TXT file with encoding detection and normalization."""
        try:
            # Detect encoding
            encoding = DocumentProcessor.detect_encoding(file_path)
            
            # Read file with detected encoding
            with open(file_path, 'r', encoding=encoding) as file:
                text = file.read()
            
            # Normalize text
            normalized_text = DocumentProcessor.normalize_text(text)
            
            print(f"DEBUG: TXT file {file_path} - Encoding: {encoding}, Length: {len(normalized_text)} chars")
            return normalized_text
            
        except UnicodeDecodeError:
            # Fallback to latin-1 if encoding detection fails
            try:
                with open(file_path, 'r', encoding='latin-1') as file:
                    text = file.read()
                normalized_text = DocumentProcessor.normalize_text(text)
                print(f"DEBUG: TXT file {file_path} - Fallback encoding: latin-1, Length: {len(normalized_text)} chars")
                return normalized_text
            except Exception as e:
                print(f"ERROR: Failed to read TXT file {file_path}: {e}")
                return ""
    
    @staticmethod
    def extract_text_from_pdf(file_path: str) -> str:
        """Extract text from PDF file with normalization."""
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
                
                # Normalize text
                normalized_text = DocumentProcessor.normalize_text(text)
                
                print(f"DEBUG: PDF file {file_path} - Length: {len(normalized_text)} chars")
                return normalized_text
                
        except Exception as e:
            print(f"ERROR: Failed to read PDF file {file_path}: {e}")
            return ""
    
    @staticmethod
    def extract_text_from_docx(file_path: str) -> str:
        """Extract text from DOCX file with normalization."""
        try:
            doc = Document(file_path)
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            
            # Normalize text
            normalized_text = DocumentProcessor.normalize_text(text)
            
            print(f"DEBUG: DOCX file {file_path} - Length: {len(normalized_text)} chars")
            return normalized_text
            
        except Exception as e:
            print(f"ERROR: Failed to read DOCX file {file_path}: {e}")
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
        
        # Clear existing data to prevent duplicates
        self.resumes.clear()
        self.job_descriptions.clear()
        self.candidate_names.clear()
        self.jd_names.clear()
        
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
                standardized_name = f"{candidate_name}-resume-{i}"
                
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
        
        # Extract summary/objective with enhanced detection
        in_summary = False
        summary_keywords = [
            'summary', 'objective', 'profile', 'about', 'overview', 
            'professional summary', 'professional overview', 'about me',
            'executive summary', 'career summary', 'personal summary',
            'professional profile', 'career objective', 'personal statement'
        ]
        
        for line in lines:
            line_lower = line.lower()
            # Check for summary section headers
            if any(keyword in line_lower for keyword in summary_keywords):
                in_summary = True
                continue
            elif in_summary and line.strip():
                # Stop if we hit another major section
                if any(word in line_lower for word in ['experience', 'education', 'skills', 'contact', 'work history', 'employment']):
                    break
                summary['summary'] += line.strip() + ' '
        
        # Add fallback: if summary is missing, concatenate experience + education
        if not summary['summary'].strip():
            fallback_parts = []
            if summary['experience']:
                fallback_parts.extend(summary['experience'][:2])  # First 2 experience entries
            if summary['education']:
                fallback_parts.extend(summary['education'][:1])   # First education entry
            if fallback_parts:
                summary['summary'] = ' '.join(fallback_parts)
        
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
        self.model = load_sentence_transformer()  # Load model during initialization
        self.embeddings_cache = {}
        self.similarity_matrix = None
        
    def load_documents(self):
        """Load all resumes and job descriptions from the directory."""
        if not self.data_dir.exists():
            st.error(f"Directory {self.data_dir} does not exist!")
            return
        
        # Clear existing data to prevent duplicates
        self.resumes.clear()
        self.job_descriptions.clear()
        self.candidate_names.clear()
        self.jd_names.clear()
        
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
                standardized_name = f"{candidate_name}-resume-{i}"
                
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
            'experience': '',
            'summary': ''
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
                print(f"DEBUG: Detected section '{section_detected}' from line: '{original_line}'")
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
        
        # Debug: Print parsed sections keys and content lengths
        print(f"DEBUG: Parsed sections keys: {list(sections.keys())}")
        for key, content in sections.items():
            print(f"DEBUG: {key}: {len(content)} characters")
            if content.strip():
                print(f"DEBUG: {key} preview: {content[:100]}...")
        
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
            
            # Also try the new parse_resume_sections method for better summary detection
            text = self.processor.extract_text(file_path)
            enhanced_sections = parser.parse_resume_sections(text, str(file_path))
            
            # Merge results, prioritizing enhanced_sections for summary
            if enhanced_sections.get('summary'):
                sections['summary'] = enhanced_sections['summary']
                print(f"DEBUG: Enhanced summary detection found: {enhanced_sections['summary'][:100]}...")
            
            # Store parsed data for later use
            self._cached_parsed_data = parsed_data
            
            # Add raw text for fallback purposes
            sections['raw_text'] = text
            
            return sections
            
        except Exception as e:
            print(f"Enhanced parser failed: {e}")
            # Fallback to original method
            text = self.processor.extract_text(file_path)
            sections = self.extract_sections(text)
            # Add raw text for fallback purposes
            sections['raw_text'] = text
            return sections
    
    def _clean_resume_text(self, text: str) -> str:
        """Clean and normalize resume text for better section detection."""
        # Handle case where text is all on one line (common with PDF extraction)
        if '\n' not in text or len(text.split('\n')) < 3:
            print("DEBUG: PDF text appears to be single line, applying enhanced splitting")
            
            # More comprehensive splitting for PDF text
            separators = [
                'Experience', 'Skills', 'Education', 'Profile', 'Contact',
                'Professional Summary', 'Work Experience', 'Technical Skills',
                'Academic Background', 'Career Objective', 'Summary',
                'Employment', 'Work History', 'Professional Experience',
                '•', 'o', '-', '–', '▪', '▫'
            ]
            
            for sep in separators:
                text = text.replace(sep, f'\n{sep}\n')
            
            # Also split on common patterns
            text = re.sub(r'([A-Z][a-z]+ [A-Z][a-z]+)', r'\n\1\n', text)  # Names like "Daniel Park"
            text = re.sub(r'(\d{4}–\d{4}|\d{4}-\d{4})', r'\n\1\n', text)  # Date ranges
            text = re.sub(r'([A-Z]{2,})', r'\n\1\n', text)  # All caps words
        
        return text
    
    def _detect_section_header(self, line_lower: str) -> str:
        """Detect if a line is a section header."""
        # Summary section with enhanced patterns
        summary_patterns = [
            'professional summary', 'profile', 'summary', 'summary of qualifications',
            'professional overview', 'about me', 'objective', 'career objective',
            'executive summary', 'personal summary', 'career summary',
            'professional profile', 'personal statement'
        ]
        if any(pattern in line_lower for pattern in summary_patterns):
            return 'summary'
        
        # Education section - more flexible patterns
        education_patterns = [
            'education', 'academic', 'degree', 'university', 'college', 'qualification',
            'bachelor', 'master', 'phd', 'diploma', 'certificate', 'graduated'
        ]
        if any(pattern in line_lower for pattern in education_patterns):
            return 'education'
        
        # Skills section - more flexible patterns
        skills_patterns = [
            'skill', 'technical', 'competenc', 'expertise', 'proficien', 'technolog', 'tool',
            'programming', 'language', 'framework', 'software', 'platform'
        ]
        if any(pattern in line_lower for pattern in skills_patterns):
            return 'skills'
        
        # Experience section - more flexible patterns
        experience_patterns = [
            'experience', 'work', 'employment', 'career', 'professional', 'work history', 
            'employment history', 'position', 'role', 'job', 'company', 'engineer', 'analyst',
            'manager', 'director', 'senior', 'junior', 'lead', 'principal'
        ]
        if any(pattern in line_lower for pattern in experience_patterns):
            return 'experience'
        
        return None
    
    def _is_contact_info(self, line_lower: str) -> bool:
        """Check if a line contains contact information."""
        contact_keywords = [
            'contact', 'email', 'phone', 'github', 'linkedin', 'website', 
            'address', 'location', '@', 'http', 'www', 'linkedin.com', 'github.com'
        ]
        
        # Only flag as contact info if it contains actual contact patterns
        has_contact_pattern = any(keyword in line_lower for keyword in contact_keywords)
        
        # Also check for email patterns
        has_email = '@' in line_lower and '.' in line_lower
        
        # Check for phone patterns
        has_phone = re.search(r'\+?\d{1,3}[-.\s]?\d{3,4}[-.\s]?\d{3,4}', line_lower)
        
        return has_contact_pattern or has_email or bool(has_phone)
    
    def _should_include_content(self, section: str, line_lower: str, original_line: str) -> bool:
        """Determine if content should be included in a section."""
        # Skip contact information
        if self._is_contact_info(line_lower):
            return False
        
        # Skip very short lines (less than 3 characters)
        if len(original_line.strip()) < 3:
            return False
        
        # Skip lines that are just section headers
        if line_lower in ['experience', 'skills', 'education', 'summary', 'profile', 'contact']:
            return False
        
        # For PDFs, be more lenient - include most content unless it's clearly contact info
        # This helps with PDFs where content might not match exact patterns
        
        # For experience section, be more flexible
        if section == 'experience':
            # Include if it has job-related content OR if it's substantial content (for PDFs)
            job_indicators = [
                'engineer', 'analyst', 'scientist', 'developer', 'manager', 'director',
                'implemented', 'built', 'migrated', 'decreased', 'increased', 'created',
                'designed', 'developed', 'managed', 'led', 'improved', 'optimized',
                'data', 'software', 'system', 'platform', 'pipeline', 'model'
            ]
            has_job_content = any(indicator in line_lower for indicator in job_indicators)
            is_substantial = len(original_line.strip()) > 20  # Substantial content for PDFs
            return has_job_content or is_substantial
        
        # For skills section, be more flexible
        elif section == 'skills':
            skill_indicators = [
                'python', 'java', 'sql', 'javascript', 'react', 'aws', 'docker',
                'kubernetes', 'spark', 'kafka', 'airflow', 'snowflake', 'bigquery',
                'machine learning', 'data science', 'etl', 'streaming', 'tool', 'technology'
            ]
            has_skill_content = any(indicator in line_lower for indicator in skill_indicators)
            is_substantial = len(original_line.strip()) > 15  # Substantial content for PDFs
            return has_skill_content or is_substantial
        
        # For education section, be more flexible
        elif section == 'education':
            education_indicators = [
                'bachelor', 'master', 'phd', 'university', 'college', 'degree',
                'diploma', 'ms', 'bs', 'mba', 'computer science', 'data science', 'graduated'
            ]
            has_education_content = any(indicator in line_lower for indicator in education_indicators)
            is_substantial = len(original_line.strip()) > 10  # Substantial content for PDFs
            return has_education_content or is_substantial
        
        # For summary section, include most content
        elif section == 'summary':
            return len(original_line.strip()) > 10
        
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
    
    def compute_semantic_similarity(self, education_weight: float = 0.1, skills_weight: float = 0.4, experience_weight: float = 0.4, summary_weight: float = 0.2):
        """Compute semantic similarity using improved weighted embedding algorithm."""
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
        
        # Compute improved similarities using weighted embedding approach
        similarities = {}
        
        for jd_name, jd_reqs in jd_requirements.items():
            similarities[jd_name] = {}
            
            # Combine JD requirements into single text
            jd_text_parts = []
            for req_type, reqs in jd_reqs.items():
                if reqs:
                    jd_text_parts.extend([req[0] for req in reqs])
            jd_text = " ".join(jd_text_parts)
            
            for resume_name, resume_sections_data in resume_sections.items():
                print(f"\nDEBUG: Computing semantic similarity for {resume_name} vs {jd_name}")
                
                # Generate professional summary from experience
                generated_summary = self.generate_professional_summary(resume_sections_data)
                print(f"DEBUG: Generated professional summary for {resume_name}: {generated_summary[:100]}...")
                
                # Update resume sections with generated summary
                resume_sections_data['summary'] = generated_summary
                
                # Compute weighted similarity using improved algorithm
                raw_sim, display_score = self.compute_weighted_similarity_with_custom_weights(
                    resume_sections_data, 
                    jd_text,
                    education_weight=education_weight,
                    skills_weight=skills_weight,
                    experience_weight=experience_weight,
                    summary_weight=summary_weight
                )
                
                # Get debug breakdown
                breakdown = self.debug_similarity_breakdown(resume_sections_data, jd_text)
                
                # Store detailed results
                similarities[jd_name][resume_name] = {
                    'raw_similarity': raw_sim,
                    'display_score': display_score,
                    'total': display_score / 100,  # Convert to 0-1 range for compatibility
                    'breakdown': breakdown
                }
                
                print(f"DEBUG: Final result - Raw: {raw_sim:.4f}, Display: {display_score}%")
        
        # Convert to matrix format using display scores
        similarity_matrix = np.zeros((len(self.candidate_names), len(self.jd_names)))
        section_matrices = {
            'education': np.zeros((len(self.candidate_names), len(self.jd_names))),
            'skills': np.zeros((len(self.candidate_names), len(self.jd_names))),
            'experience': np.zeros((len(self.candidate_names), len(self.jd_names))),
            'summary': np.zeros((len(self.candidate_names), len(self.jd_names)))
        }
        
        for jd_idx, jd_name in enumerate(self.jd_names):
            for resume_idx, resume_name in enumerate(self.candidate_names):
                if jd_name in similarities and resume_name in similarities[jd_name]:
                    # Use display score (0-100) converted to 0-1 range
                    similarity_matrix[resume_idx, jd_idx] = similarities[jd_name][resume_name]['display_score'] / 100
                    
                    # Store section breakdowns
                    breakdown = similarities[jd_name][resume_name]['breakdown']
                    for section in ['education', 'skills', 'experience', 'summary']:
                        if section in breakdown:
                            section_matrices[section][resume_idx, jd_idx] = breakdown[section]
        
        end_time = time.time()
        st.info(f"🧠 Semantic matching completed in {end_time - start_time:.2f} seconds")
        
        # Store similarity matrix for get_top_matches method
        self.similarity_matrix = similarity_matrix
        
        return similarity_matrix, section_matrices, similarities
    
    def compute_weighted_embedding_similarity(self, summary_weight: float = 0.5, skills_weight: float = 0.3, experience_weight: float = 0.2):
        """Compute similarity using weighted embedding averaging for high-level alignment."""
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
        
        for name, text in self.job_descriptions.items():
            jd_requirements[name] = self.extract_jd_requirements_with_importance(text)
        
        # Compute weighted embedding similarities
        similarities = {}
        
        for jd_name, jd_reqs in jd_requirements.items():
            similarities[jd_name] = {}
            
            for resume_name, resume_sections_data in resume_sections.items():
                # Create weighted embeddings for resume
                resume_embeddings = []
                resume_weights = []
                
                # Generate professional summary from experience
                generated_summary = self.generate_professional_summary(resume_sections_data)
                print(f"DEBUG: Generated professional summary for {resume_name}: {generated_summary[:100]}...")
                
                # Summary embedding (highest weight for high-level alignment)
                summary_text = generated_summary
                
                if summary_text:
                    summary_embedding = self._get_embedding(summary_text, 'summary', resume_sections_data)
                    resume_embeddings.append(summary_embedding)
                    resume_weights.append(summary_weight)
                    print(f"DEBUG: Added summary embedding for {resume_name} (weight: {summary_weight})")
                
                # Skills embedding (medium weight for technical alignment)
                skills_text = resume_sections_data.get('skills', '').strip()
                if skills_text:
                    skills_embedding = self._get_embedding(skills_text, 'skills', resume_sections_data)
                    resume_embeddings.append(skills_embedding)
                    resume_weights.append(skills_weight)
                    print(f"DEBUG: Added skills embedding for {resume_name} (weight: {skills_weight})")
                
                # Experience embedding (lower weight for detailed alignment)
                experience_text = resume_sections_data.get('experience', '').strip()
                if experience_text:
                    experience_embedding = self._get_embedding(experience_text, 'experience', resume_sections_data)
                    resume_embeddings.append(experience_embedding)
                    resume_weights.append(experience_weight)
                    print(f"DEBUG: Added experience embedding for {resume_name} (weight: {experience_weight})")
                
                if not resume_embeddings:
                    similarities[jd_name][resume_name] = {'total': 0.0, 'sections': {}}
                    continue
                
                # Normalize weights
                total_weight = sum(resume_weights)
                normalized_weights = [w / total_weight for w in resume_weights]
                
                # Compute weighted average embedding
                weighted_resume_embedding = np.average(resume_embeddings, axis=0, weights=normalized_weights)
                
                # Create weighted embeddings for JD
                jd_embeddings = []
                jd_weights = []
                
                # JD Summary embedding
                if jd_reqs.get('summary'):
                    jd_summary_text = " ".join([req[0] for req in jd_reqs['summary']])
                    jd_summary_embedding = self._get_embedding(jd_summary_text, 'summary_jd')
                    jd_embeddings.append(jd_summary_embedding)
                    jd_weights.append(summary_weight)
                
                # JD Skills embedding
                if jd_reqs.get('skills'):
                    jd_skills_text = " ".join([req[0] for req in jd_reqs['skills']])
                    jd_skills_embedding = self._get_embedding(jd_skills_text, 'skills_jd')
                    jd_embeddings.append(jd_skills_embedding)
                    jd_weights.append(skills_weight)
                
                # JD Experience embedding
                if jd_reqs.get('experience'):
                    jd_experience_text = " ".join([req[0] for req in jd_reqs['experience']])
                    jd_experience_embedding = self._get_embedding(jd_experience_text, 'experience_jd')
                    jd_embeddings.append(jd_experience_embedding)
                    jd_weights.append(experience_weight)
                
                if not jd_embeddings:
                    similarities[jd_name][resume_name] = {'total': 0.0, 'sections': {}}
                    continue
                
                # Normalize JD weights
                jd_total_weight = sum(jd_weights)
                jd_normalized_weights = [w / jd_total_weight for w in jd_weights]
                
                # Compute weighted average JD embedding
                weighted_jd_embedding = np.average(jd_embeddings, axis=0, weights=jd_normalized_weights)
                
                # Compute cosine similarity between weighted embeddings
                similarity = cosine_similarity([weighted_resume_embedding], [weighted_jd_embedding])[0][0]
                
                # Store results
                similarities[jd_name][resume_name] = {
                    'total': float(similarity),
                    'sections': {
                        'summary': summary_weight,
                        'skills': skills_weight,
                        'experience': experience_weight
                    }
                }
                
                print(f"DEBUG: Weighted embedding similarity for {resume_name} vs {jd_name}: {similarity:.4f}")
        
        # Convert to matrix format
        similarity_matrix = np.zeros((len(self.candidate_names), len(self.jd_names)))
        
        for jd_idx, jd_name in enumerate(self.jd_names):
            for resume_idx, resume_name in enumerate(self.candidate_names):
                if jd_name in similarities and resume_name in similarities[jd_name]:
                    similarity_matrix[resume_idx, jd_idx] = similarities[jd_name][resume_name]['total']
        
        end_time = time.time()
        st.info(f"🎯 Weighted embedding matching completed in {end_time - start_time:.2f} seconds")
        
        return similarity_matrix, similarities
    
    def scale_yoe(self, candidate_years: float, required_years: float) -> float:
        """Scale years of experience with linear interpolation."""
        if required_years <= 0:
            return 1.0
        
        ratio = min(candidate_years / required_years, 1.0)
        # Use bounded scaling: minimum 0.1, maximum 1.0
        return max(0.1, ratio)
    
    def extract_years_of_experience(self, text: str) -> float:
        """Extract years of experience from text using regex patterns."""
        import re
        
        # Common patterns for years of experience
        patterns = [
            r'(\d+(?:\.\d+)?)\s*\+?\s*years?\s*(?:of\s*)?(?:experience|exp)',
            r'(\d+(?:\.\d+)?)\s*\+?\s*yrs?\s*(?:of\s*)?(?:experience|exp)',
            r'(\d+(?:\.\d+)?)\s*\+?\s*years?\s*in',
            r'(\d+(?:\.\d+)?)\s*\+?\s*years?\s*working',
            r'(\d+(?:\.\d+)?)\s*\+?\s*years?\s*with',
            r'(\d+(?:\.\d+)?)\s*\+?\s*years?\s*of\s*professional',
            r'(\d+(?:\.\d+)?)\s*\+?\s*years?\s*in\s*the\s*field',
            r'(\d+(?:\.\d+)?)\s*\+?\s*years?\s*of\s*industry'
        ]
        
        years_found = []
        for pattern in patterns:
            matches = re.findall(pattern, text.lower())
            for match in matches:
                try:
                    years = float(match)
                    if 0.1 <= years <= 50:  # Reasonable range
                        years_found.append(years)
                except ValueError:
                    continue
        
        # Return the maximum years found (most relevant)
        return max(years_found) if years_found else 0.0
    
    def get_skill_similarity(self, skill1: str, skill2: str) -> float:
        """Calculate similarity between two skills using predefined mappings and embeddings."""
        skill1_lower = skill1.lower().strip()
        skill2_lower = skill2.lower().strip()
        
        # Exact match
        if skill1_lower == skill2_lower:
            return 1.0
        
        # Predefined skill similarity mappings
        skill_similarities = {
            # Data Processing & Streaming
            'kafka': {'flink': 0.8, 'kinesis': 0.8, 'pulsar': 0.7, 'rabbitmq': 0.6},
            'spark': {'pyspark': 0.9, 'hadoop': 0.7, 'flink': 0.8, 'storm': 0.6},
            'hadoop': {'spark': 0.7, 'hive': 0.8, 'pig': 0.6, 'mapreduce': 0.9},
            
            # Machine Learning Frameworks
            'pytorch': {'tensorflow': 0.8, 'keras': 0.7, 'scikit-learn': 0.6, 'mxnet': 0.6},
            'tensorflow': {'pytorch': 0.8, 'keras': 0.9, 'scikit-learn': 0.6, 'mxnet': 0.6},
            'keras': {'tensorflow': 0.9, 'pytorch': 0.7, 'scikit-learn': 0.6},
            
            # ML Platforms & Tools
            'mlflow': {'weights & biases': 0.7, 'neptune': 0.6, 'comet': 0.6, 'tensorboard': 0.5},
            'weights & biases': {'mlflow': 0.7, 'neptune': 0.8, 'comet': 0.7},
            'neptune': {'mlflow': 0.6, 'weights & biases': 0.8, 'comet': 0.7},
            
            # Cloud Platforms
            'aws': {'azure': 0.6, 'gcp': 0.6, 'google cloud': 0.6},
            'azure': {'aws': 0.6, 'gcp': 0.6, 'google cloud': 0.6},
            'gcp': {'aws': 0.6, 'azure': 0.6, 'google cloud': 0.9},
            'google cloud': {'aws': 0.6, 'azure': 0.6, 'gcp': 0.9},
            
            # Databases
            'postgresql': {'postgres': 0.9, 'mysql': 0.7, 'oracle': 0.6, 'sql server': 0.6},
            'mysql': {'postgresql': 0.7, 'postgres': 0.7, 'mariadb': 0.8},
            'mongodb': {'cassandra': 0.6, 'couchdb': 0.5, 'dynamodb': 0.6},
            'redis': {'memcached': 0.7, 'hazelcast': 0.6},
            
            # Programming Languages
            'python': {'r': 0.6, 'julia': 0.5, 'matlab': 0.5},
            'java': {'scala': 0.7, 'kotlin': 0.6, 'groovy': 0.6},
            'javascript': {'typescript': 0.8, 'node.js': 0.7, 'react': 0.6},
            'typescript': {'javascript': 0.8, 'node.js': 0.7},
            
            # Container & Orchestration
            'docker': {'podman': 0.8, 'containerd': 0.7},
            'kubernetes': {'docker swarm': 0.6, 'nomad': 0.5, 'rancher': 0.7},
            'helm': {'kustomize': 0.6, 'skaffold': 0.5},
            
            # CI/CD & DevOps
            'jenkins': {'gitlab ci': 0.7, 'github actions': 0.7, 'azure devops': 0.6},
            'gitlab ci': {'jenkins': 0.7, 'github actions': 0.8, 'azure devops': 0.7},
            'github actions': {'jenkins': 0.7, 'gitlab ci': 0.8, 'azure devops': 0.7},
            
            # Data Warehouses
            'snowflake': {'bigquery': 0.7, 'redshift': 0.7, 'databricks': 0.6},
            'bigquery': {'snowflake': 0.7, 'redshift': 0.6, 'databricks': 0.6},
            'redshift': {'snowflake': 0.7, 'bigquery': 0.6, 'databricks': 0.6},
            
            # Visualization
            'tableau': {'power bi': 0.7, 'looker': 0.6, 'qlik': 0.6},
            'power bi': {'tableau': 0.7, 'looker': 0.6, 'qlik': 0.6},
            'looker': {'tableau': 0.6, 'power bi': 0.6, 'qlik': 0.7},
            
            # Version Control
            'git': {'svn': 0.5, 'mercurial': 0.4},
            'github': {'gitlab': 0.8, 'bitbucket': 0.7},
            'gitlab': {'github': 0.8, 'bitbucket': 0.7}
        }
        
        # Check predefined mappings
        if skill1_lower in skill_similarities:
            if skill2_lower in skill_similarities[skill1_lower]:
                return skill_similarities[skill1_lower][skill2_lower]
        
        # Reverse check
        if skill2_lower in skill_similarities:
            if skill1_lower in skill_similarities[skill2_lower]:
                return skill_similarities[skill2_lower][skill1_lower]
        
        # If no predefined mapping, use embedding similarity as fallback
        if self.model:
            try:
                embedding1 = self._get_embedding(skill1, 'skill')
                embedding2 = self._get_embedding(skill2, 'skill')
                similarity = cosine_similarity([embedding1], [embedding2])[0][0]
                # Only return high similarity (>0.7) for embedding-based matches
                return similarity if similarity > 0.7 else 0.0
            except Exception:
                pass
        
        return 0.0
    
    def compute_graded_similarity(self, resume_sections: dict, jd_requirements: dict) -> dict:
        """Compute graded similarity with partial credit for preferred skills and YOE scaling."""
        scores = {
            'required_skills': 0.0,
            'preferred_skills': 0.0,
            'yoe_score': 0.0,
            'total_score': 0.0,
            'breakdown': []
        }
        
        # Extract years of experience from resume
        experience_text = resume_sections.get('experience', '') + ' ' + resume_sections.get('summary', '')
        candidate_yoe = self.extract_years_of_experience(experience_text)
        
        # Process required skills (binary gates)
        required_skills = jd_requirements.get('skills', [])
        required_matches = 0
        required_total = 0
        
        for skill_req, importance in required_skills:
            if importance >= 1.0:  # Required skill
                required_total += 1
                skill_matched = False
                
                # Check against resume skills
                resume_skills_text = resume_sections.get('skills', '') + ' ' + resume_sections.get('experience', '')
                resume_skills = resume_skills_text.lower()
                
                # Check for exact match or high similarity
                if skill_req.lower() in resume_skills:
                    required_matches += 1
                    skill_matched = True
                    scores['breakdown'].append(f"✅ {skill_req} (required): Exact match")
                else:
                    # Check for similar skills
                    for word in resume_skills.split():
                        word = word.strip('.,!?()[]{}')
                        similarity = self.get_skill_similarity(skill_req, word)
                        if similarity >= 0.8:  # High similarity threshold for required skills
                            required_matches += 1
                            skill_matched = True
                            scores['breakdown'].append(f"✅ {skill_req} (required): Similar to {word} (similarity: {similarity:.2f})")
                            break
                
                if not skill_matched:
                    scores['breakdown'].append(f"❌ {skill_req} (required): Not found")
        
        # Calculate required skills score
        if required_total > 0:
            scores['required_skills'] = required_matches / required_total
        else:
            scores['required_skills'] = 1.0  # No required skills
        
        # Process preferred skills (weighted bonuses)
        preferred_bonus = 0.0
        preferred_total = 0
        
        for skill_req, importance in required_skills:
            if importance < 1.0:  # Preferred/nice-to-have skill
                preferred_total += 1
                best_match_score = 0.0
                best_match_skill = ""
                
                # Check against resume skills
                resume_skills_text = resume_sections.get('skills', '') + ' ' + resume_sections.get('experience', '')
                resume_skills = resume_skills_text.lower()
                
                # Check for exact match
                if skill_req.lower() in resume_skills:
                    best_match_score = 1.0
                    best_match_skill = skill_req
                else:
                    # Check for similar skills
                    for word in resume_skills.split():
                        word = word.strip('.,!?()[]{}')
                        similarity = self.get_skill_similarity(skill_req, word)
                        if similarity > best_match_score:
                            best_match_score = similarity
                            best_match_skill = word
                
                # Add bonus based on match quality
                if best_match_score > 0.0:
                    bonus = best_match_score * importance  # Weight by importance level
                    preferred_bonus += bonus
                    scores['breakdown'].append(f"🎯 {skill_req} (preferred): Matched with {best_match_skill} (score: {bonus:.2f})")
                else:
                    scores['breakdown'].append(f"⚪ {skill_req} (preferred): Not found")
        
        # Calculate preferred skills score
        if preferred_total > 0:
            scores['preferred_skills'] = preferred_bonus / preferred_total
        else:
            scores['preferred_skills'] = 1.0  # No preferred skills
        
        # Process Years of Experience scaling
        yoe_requirements = []
        jd_text = ' '.join([req[0] for req in jd_requirements.get('experience', [])])
        yoe_patterns = [
            r'(\d+(?:\.\d+)?)\s*\+?\s*years?\s*(?:of\s*)?(?:experience|exp)',
            r'(\d+(?:\.\d+)?)\s*\+?\s*yrs?\s*(?:of\s*)?(?:experience|exp)',
            r'(\d+(?:\.\d+)?)\s*\+?\s*years?\s*preferred',
            r'(\d+(?:\.\d+)?)\s*\+?\s*years?\s*required'
        ]
        
        import re
        for pattern in yoe_patterns:
            matches = re.findall(pattern, jd_text.lower())
            for match in matches:
                try:
                    required_yoe = float(match)
                    if 1 <= required_yoe <= 20:  # Reasonable range
                        yoe_requirements.append(required_yoe)
                except ValueError:
                    continue
        
        if yoe_requirements:
            # Use the maximum required YOE
            max_required_yoe = max(yoe_requirements)
            scores['yoe_score'] = self.scale_yoe(candidate_yoe, max_required_yoe)
            scores['breakdown'].append(f"📅 YOE: Candidate has {candidate_yoe:.1f} years, required {max_required_yoe:.1f} years (score: {scores['yoe_score']:.2f})")
        else:
            scores['yoe_score'] = 1.0  # No YOE requirement
            scores['breakdown'].append(f"📅 YOE: No specific requirement (candidate has {candidate_yoe:.1f} years)")
        
        # Calculate composite score
        # Formula: (required_match * 0.6) + (preferred_match * 0.3) + (yoe_scaled * 0.1)
        scores['total_score'] = (
            scores['required_skills'] * 0.6 +
            scores['preferred_skills'] * 0.3 +
            scores['yoe_score'] * 0.1
        )
        
        return scores
    
    def extract_experience_elements(self, experience_text: str) -> list:
        """Extract core elements from experience section."""
        import re
        
        experience_entries = []
        
        # Split by common separators (bullet points, line breaks, etc.)
        lines = re.split(r'[•\n\r]+', experience_text)
        
        for line in lines:
            line = line.strip()
            if not line or len(line) < 10:  # Skip very short lines
                continue
            
            # Extract job title and company using common patterns
            job_title = ""
            company = ""
            duration = ""
            achievements = []
            
            # Pattern 1: "Job Title at Company (2020-2023)"
            title_company_pattern = r'([^,]+?)\s+(?:at|@)\s+([^(]+?)\s*\(([^)]+)\)'
            match = re.search(title_company_pattern, line, re.IGNORECASE)
            if match:
                job_title = match.group(1).strip()
                company = match.group(2).strip()
                duration = match.group(3).strip()
            else:
                # Pattern 2: "Job Title, Company, Duration"
                parts = [p.strip() for p in line.split(',')]
                if len(parts) >= 2:
                    job_title = parts[0]
                    company = parts[1]
                    if len(parts) >= 3:
                        duration = parts[2]
            
            # Extract achievements (lines starting with action verbs)
            achievement_verbs = [
                'built', 'developed', 'led', 'implemented', 'deployed', 'created', 'designed',
                'managed', 'optimized', 'improved', 'increased', 'reduced', 'delivered',
                'launched', 'established', 'coordinated', 'supervised', 'mentored',
                'architected', 'scaled', 'automated', 'integrated', 'migrated', 'transformed'
            ]
            
            # Look for achievement statements in the same line or following lines
            achievement_pattern = r'(?:^|\s)(?:' + '|'.join(achievement_verbs) + r')\s+[^.!?]*(?:[.!?]|$)'
            achievement_matches = re.findall(achievement_pattern, line, re.IGNORECASE)
            
            for match in achievement_matches:
                achievement = match.strip()
                if len(achievement) > 15:  # Only keep substantial achievements
                    achievements.append(achievement)
            
            # Only add if we have meaningful information
            if job_title or company or achievements:
                experience_entries.append({
                    'job_title': job_title,
                    'company': company,
                    'duration': duration,
                    'achievements': achievements[:2]  # Limit to top 2 achievements
                })
        
        return experience_entries
    
    def calculate_total_experience(self, experience_entries: list) -> float:
        """Calculate total years of experience from entries."""
        import re
        
        total_years = 0.0
        
        for entry in experience_entries:
            duration = entry.get('duration', '')
            if not duration:
                continue
            
            # Extract years from duration strings
            year_patterns = [
                r'(\d{4})\s*[-–]\s*(\d{4})',  # 2020-2023
                r'(\d{4})\s*[-–]\s*(?:present|current|now)',  # 2020-present
                r'(\d+(?:\.\d+)?)\s*years?',  # 2.5 years
                r'(\d+)\s*\+?\s*years?'  # 3+ years
            ]
            
            for pattern in year_patterns:
                matches = re.findall(pattern, duration.lower())
                if matches:
                    if len(matches[0]) == 2:  # Date range
                        try:
                            start_year = int(matches[0][0])
                            end_year = int(matches[0][1]) if matches[0][1].isdigit() else 2024
                            years = end_year - start_year
                            total_years += max(0, years)
                        except ValueError:
                            pass
                    else:  # Single year value
                        try:
                            years = float(matches[0])
                            total_years += years
                        except ValueError:
                            pass
                    break
        
        return total_years
    
    def identify_recurring_themes(self, experience_entries: list) -> list:
        """Identify recurring themes across experience entries."""
        theme_keywords = {
            'machine_learning': ['ml', 'machine learning', 'ai', 'artificial intelligence', 'model', 'prediction', 'algorithm'],
            'data_science': ['data science', 'analytics', 'statistics', 'data analysis', 'insights'],
            'software_development': ['software', 'development', 'programming', 'coding', 'engineering'],
            'cloud_computing': ['aws', 'azure', 'gcp', 'cloud', 'infrastructure'],
            'data_engineering': ['etl', 'pipeline', 'data warehouse', 'big data', 'spark', 'hadoop'],
            'devops': ['devops', 'deployment', 'ci/cd', 'docker', 'kubernetes', 'automation'],
            'product_management': ['product', 'management', 'strategy', 'roadmap', 'stakeholder'],
            'leadership': ['lead', 'managed', 'team', 'supervised', 'mentored', 'directed']
        }
        
        theme_counts = {theme: 0 for theme in theme_keywords.keys()}
        
        for entry in experience_entries:
            text = f"{entry.get('job_title', '')} {entry.get('company', '')} {' '.join(entry.get('achievements', []))}".lower()
            
            for theme, keywords in theme_keywords.items():
                for keyword in keywords:
                    if keyword in text:
                        theme_counts[theme] += 1
                        break
        
        # Return themes that appear in multiple entries
        recurring_themes = [theme for theme, count in theme_counts.items() if count >= 2]
        return recurring_themes
    
    def generate_professional_summary(self, resume_sections: dict) -> str:
        """Generate a professional summary from experience and skills."""
        
        # Check if there's already a summary
        existing_summary = resume_sections.get('summary', '').strip()
        experience_text = resume_sections.get('experience', '').strip()
        skills_text = resume_sections.get('skills', '').strip()
        
        if not experience_text:
            return existing_summary if existing_summary else "Professional experience details not available."
        
        # Extract experience elements
        experience_entries = self.extract_experience_elements(experience_text)
        
        if not experience_entries:
            return existing_summary if existing_summary else "Experience details could not be parsed."
        
        # Calculate total experience
        total_years = self.calculate_total_experience(experience_entries)
        
        # Identify recurring themes
        recurring_themes = self.identify_recurring_themes(experience_entries)
        
        # Extract key skills
        key_skills = []
        if skills_text:
            # Extract top skills (first 5-7 skills mentioned)
            skill_words = skills_text.lower().split(',')
            key_skills = [skill.strip().title() for skill in skill_words[:7] if len(skill.strip()) > 2]
        
        # Generate summary based on available information
        summary_parts = []
        
        # Start with role and experience level
        if experience_entries:
            # Get the most recent or highest-level role
            primary_role = experience_entries[0].get('job_title', 'Professional')
            if not primary_role:
                primary_role = 'Professional'
            
            # Determine experience level
            if total_years >= 5:
                exp_level = f"{total_years:.0f}+ years"
            elif total_years >= 1:
                exp_level = f"{total_years:.1f} years"
            else:
                exp_level = "entry-level"
            
            summary_parts.append(f"{primary_role} with {exp_level} of experience")
        
        # Add company context
        companies = [entry.get('company', '') for entry in experience_entries if entry.get('company')]
        if companies:
            unique_companies = list(dict.fromkeys(companies))  # Remove duplicates while preserving order
            if len(unique_companies) <= 2:
                company_context = " and ".join(unique_companies)
            else:
                company_context = f"{unique_companies[0]} and {len(unique_companies)-1} other companies"
            summary_parts.append(f"across {company_context}")
        
        # Add recurring themes
        if recurring_themes:
            theme_descriptions = {
                'machine_learning': 'machine learning and AI',
                'data_science': 'data science and analytics',
                'software_development': 'software development',
                'cloud_computing': 'cloud computing',
                'data_engineering': 'data engineering',
                'devops': 'DevOps and automation',
                'product_management': 'product management',
                'leadership': 'team leadership'
            }
            
            theme_texts = [theme_descriptions.get(theme, theme) for theme in recurring_themes[:3]]
            if theme_texts:
                summary_parts.append(f"focusing on {', '.join(theme_texts)}")
        
        # Add key achievements
        key_achievements = []
        for entry in experience_entries[:2]:  # Top 2 roles
            achievements = entry.get('achievements', [])
            if achievements:
                # Take the most impactful achievement
                achievement = achievements[0]
                # Simplify the achievement
                achievement = re.sub(r'\b(?:built|developed|implemented|created|designed)\b', 'delivered', achievement.lower())
                key_achievements.append(achievement)
        
        # Construct the summary
        if summary_parts:
            summary = ". ".join(summary_parts) + "."
        else:
            summary = f"Professional with {total_years:.1f} years of experience."
        
        # Add achievement highlights
        if key_achievements:
            achievement_text = ". ".join(key_achievements[:2])
            summary += f" {achievement_text.capitalize()}."
        
        # Add skills if space allows
        if key_skills and len(summary) < 200:
            skills_text = ", ".join(key_skills[:5])
            summary += f" Skilled in {skills_text}."
        
        # If we have an existing summary, enhance it
        if existing_summary and len(existing_summary) > 20:
            # Use existing summary as base and add experience highlights
            if len(summary) < 150:  # If generated summary is short
                summary = f"{existing_summary} {summary}"
            else:
                summary = existing_summary  # Use existing if it's substantial
        
        # Ensure summary is concise (max 3 sentences, ~60 words)
        sentences = summary.split('. ')
        if len(sentences) > 3:
            summary = '. '.join(sentences[:3]) + '.'
        
        # Limit to approximately 60 words
        words = summary.split()
        if len(words) > 60:
            summary = ' '.join(words[:60]) + '...'
        
        return summary.strip()
    
    def preprocess_resume(self, parsed_resume: dict) -> str:
        """Preprocess and clean resume text for better embedding quality."""
        # Define irrelevant sections to remove
        irrelevant_sections = ["contact", "links", "certifications", "references", "personal"]
        
        # Combine important sections with fallback logic
        summary = parsed_resume.get("summary", parsed_resume.get("profile", ""))
        skills = parsed_resume.get("skills", "")
        experience = parsed_resume.get("experience", "")
        education = parsed_resume.get("education", "")
        
        # Merge the most informative content
        merged_content = f"{summary}\n{skills}\n{experience}\n{education}"
        
        # Remove low-signal sections
        for section in irrelevant_sections:
            merged_content = merged_content.replace(section, "")
        
        # Normalize whitespace and clean text
        merged_content = " ".join(merged_content.split())
        
        # Remove common noise patterns
        noise_patterns = [
            r'https?://\S+',  # URLs
            r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',  # Phone numbers
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',  # Email addresses
            r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{4}\b',  # Dates
            r'\b\d{4}\b',  # Years
            r'[^\w\s.,!?-]',  # Special characters except basic punctuation
        ]
        
        import re
        for pattern in noise_patterns:
            merged_content = re.sub(pattern, ' ', merged_content, flags=re.IGNORECASE)
        
        # Normalize whitespace again after cleaning
        merged_content = " ".join(merged_content.split())
        
        # Truncate overly long text for embedding stability
        return merged_content[:4000]  # roughly ~700–800 tokens
    
    def get_weighted_resume_embedding(self, parsed_resume: dict) -> np.ndarray:
        """Generate weighted resume embedding emphasizing skills and experience."""
        # Use default weights from improved algorithm
        return self.get_weighted_resume_embedding_with_custom_weights(
            parsed_resume,
            education_weight=0.1,
            skills_weight=0.4,
            experience_weight=0.4,
            summary_weight=0.2
        )
    
    def get_weighted_resume_embedding_with_custom_weights(self, parsed_resume: dict, 
                                                        education_weight: float = 0.1,
                                                        skills_weight: float = 0.4,
                                                        experience_weight: float = 0.4,
                                                        summary_weight: float = 0.2) -> np.ndarray:
        """Generate weighted resume embedding with custom weights."""
        summary_text = parsed_resume.get("summary", parsed_resume.get("profile", ""))
        skills_text = parsed_resume.get("skills", "")
        experience_text = parsed_resume.get("experience", "")
        education_text = parsed_resume.get("education", "")
        
        # Encode each section (skip if empty)
        vectors = {}
        weights = {}
        
        if summary_text and len(summary_text.strip()) > 10:
            vectors["summary"] = self._get_embedding(summary_text, 'summary')
            weights["summary"] = summary_weight
            print(f"DEBUG: Added summary embedding (weight: {summary_weight})")
        
        if skills_text and len(skills_text.strip()) > 10:
            vectors["skills"] = self._get_embedding(skills_text, 'skills')
            weights["skills"] = skills_weight
            print(f"DEBUG: Added skills embedding (weight: {skills_weight})")
        
        if experience_text and len(experience_text.strip()) > 20:
            vectors["experience"] = self._get_embedding(experience_text, 'experience')
            weights["experience"] = experience_weight
            print(f"DEBUG: Added experience embedding (weight: {experience_weight})")
        
        if education_text and len(education_text.strip()) > 10:
            vectors["education"] = self._get_embedding(education_text, 'education')
            weights["education"] = education_weight
            print(f"DEBUG: Added education embedding (weight: {education_weight})")
        
        if not vectors:
            print("WARNING: No valid sections found for embedding")
            print(f"DEBUG: Resume sections - Summary: '{summary_text[:50]}...', Skills: '{skills_text[:50]}...', Experience: '{experience_text[:50]}...', Education: '{education_text[:50]}...'")
            
            # Try to use the raw resume text as fallback
            raw_text = f"{summary_text} {skills_text} {experience_text} {education_text}".strip()
            if len(raw_text) > 20:
                print("DEBUG: Using raw text as fallback for embedding")
                fallback_vector = self._get_embedding(raw_text, 'fallback')
                return fallback_vector
            else:
                print("ERROR: Resume has insufficient content for meaningful embedding")
                # Last resort: try to get raw text from the parsed_resume dict
                if 'raw_text' in parsed_resume and len(parsed_resume['raw_text'].strip()) > 20:
                    print("DEBUG: Using raw_text from parsed_resume as last resort")
                    last_resort_vector = self._get_embedding(parsed_resume['raw_text'], 'last_resort')
                    return last_resort_vector
                else:
                    print("CRITICAL: No usable content found - trying emergency text extraction")
                    # Emergency fallback: try to extract any meaningful text from the file
                    try:
                        # This is a last resort - try to get any text from the file
                        emergency_text = self._emergency_text_extraction(parsed_resume)
                        if emergency_text and len(emergency_text.strip()) > 20:
                            print(f"DEBUG: Emergency text extraction found {len(emergency_text)} characters")
                            emergency_vector = self._get_embedding(emergency_text, 'emergency')
                            return emergency_vector
                        else:
                            print("CRITICAL: Emergency extraction also failed - trying PDF-specific fallback")
                            # PDF-specific fallback: try to extract from raw text with better parsing
                            pdf_fallback_text = self._pdf_specific_fallback(parsed_resume)
                            if pdf_fallback_text and len(pdf_fallback_text.strip()) > 20:
                                print(f"DEBUG: PDF-specific fallback found {len(pdf_fallback_text)} characters")
                                pdf_vector = self._get_embedding(pdf_fallback_text, 'pdf_fallback')
                                return pdf_vector
                            else:
                                print("CRITICAL: All fallback methods failed - returning zero vector")
                                return np.zeros(384)  # Default embedding size
                    except Exception as e:
                        print(f"CRITICAL: Emergency extraction failed: {e}")
                        return np.zeros(384)  # Default embedding size
        
        # Normalize weights
        total_weight = sum(weights.values())
        normalized_weights = {k: v / total_weight for k, v in weights.items()}
        
        # Weighted combination
        weighted_vec = np.zeros(384)  # Default embedding size
        for section, vector in vectors.items():
            weight = normalized_weights[section]
            weighted_vec += weight * vector
            print(f"DEBUG: Added {section} contribution (normalized weight: {weight:.3f})")
        
        return weighted_vec
    
    def _pdf_specific_fallback(self, parsed_resume: dict) -> str:
        """PDF-specific fallback for when normal section extraction fails."""
        try:
            # Get raw text and apply aggressive PDF parsing
            raw_text = parsed_resume.get('raw_text', '')
            if not raw_text:
                return ""
            
            print("DEBUG: Applying PDF-specific parsing to raw text")
            
            # Split the text more aggressively for PDFs
            # Look for common PDF patterns
            pdf_patterns = [
                r'([A-Z][a-z]+ [A-Z][a-z]+)',  # Names like "Daniel Park"
                r'(\d{4}–\d{4}|\d{4}-\d{4})',  # Date ranges
                r'([A-Z][a-z]+ [A-Z][a-z]+ [A-Z][a-z]+)',  # Full names
                r'(Python|SQL|Java|JavaScript|React|Angular|Docker|Kubernetes)',  # Common tech skills
                r'(Bachelor|Master|PhD|University|College)',  # Education keywords
                r'(Engineer|Analyst|Manager|Director|Developer)',  # Job titles
            ]
            
            # Extract meaningful chunks
            meaningful_chunks = []
            for pattern in pdf_patterns:
                matches = re.findall(pattern, raw_text)
                meaningful_chunks.extend(matches)
            
            # Also try to extract sentences that contain key information
            sentences = re.split(r'[.!?]', raw_text)
            for sentence in sentences:
                sentence = sentence.strip()
                if len(sentence) > 20 and any(keyword in sentence.lower() for keyword in 
                    ['python', 'sql', 'experience', 'engineer', 'data', 'software', 'development', 'analysis']):
                    meaningful_chunks.append(sentence)
            
            if meaningful_chunks:
                pdf_text = ' '.join(meaningful_chunks)
                print(f"DEBUG: PDF-specific fallback extracted {len(meaningful_chunks)} meaningful chunks")
                return pdf_text
            
            return ""
            
        except Exception as e:
            print(f"PDF-specific fallback error: {e}")
            return ""
    
    def _emergency_text_extraction(self, parsed_resume: dict) -> str:
        """Emergency text extraction when all normal methods fail."""
        try:
            # Try to get any text from any possible key in the parsed_resume
            possible_keys = ['raw_text', 'text', 'content', 'full_text', 'document_text']
            
            for key in possible_keys:
                if key in parsed_resume and parsed_resume[key]:
                    text = str(parsed_resume[key]).strip()
                    if len(text) > 20:
                        print(f"DEBUG: Emergency extraction found text in '{key}' key")
                        return text
            
            # If no direct text, try to concatenate all string values
            all_text_parts = []
            for key, value in parsed_resume.items():
                if isinstance(value, str) and len(value.strip()) > 5:
                    all_text_parts.append(value.strip())
                elif isinstance(value, list):
                    for item in value:
                        if isinstance(item, str) and len(item.strip()) > 5:
                            all_text_parts.append(item.strip())
            
            if all_text_parts:
                emergency_text = " ".join(all_text_parts)
                print(f"DEBUG: Emergency extraction concatenated {len(all_text_parts)} text parts")
                return emergency_text
            
            return ""
            
        except Exception as e:
            print(f"Emergency extraction error: {e}")
            return ""
    
    def compute_weighted_similarity(self, parsed_resume: dict, jd_text: str) -> tuple:
        """Compute weighted similarity between resume and JD with improved preprocessing."""
        # Use default weights from improved algorithm
        return self.compute_weighted_similarity_with_custom_weights(
            parsed_resume, jd_text,
            education_weight=0.1,
            skills_weight=0.4,
            experience_weight=0.4,
            summary_weight=0.2
        )
    
    def compute_weighted_similarity_with_custom_weights(self, parsed_resume: dict, jd_text: str, 
                                                      education_weight: float = 0.1, 
                                                      skills_weight: float = 0.4, 
                                                      experience_weight: float = 0.4, 
                                                      summary_weight: float = 0.2) -> tuple:
        """Compute weighted similarity between resume and JD with custom weights."""
        # Preprocess both inputs
        clean_resume = self.preprocess_resume(parsed_resume)
        clean_jd = " ".join(jd_text.split())[:4000]  # normalize JD length
        
        print(f"DEBUG: Cleaned resume length: {len(clean_resume)} chars")
        print(f"DEBUG: Cleaned JD length: {len(clean_jd)} chars")
        
        # Generate embeddings with custom weights
        resume_vec = self.get_weighted_resume_embedding_with_custom_weights(
            parsed_resume,
            education_weight=education_weight,
            skills_weight=skills_weight,
            experience_weight=experience_weight,
            summary_weight=summary_weight
        )
        jd_vec = self._get_embedding(clean_jd, 'jd')
        
        # Compute cosine similarity
        sim = cosine_similarity(
            np.array(resume_vec).reshape(1, -1),
            np.array(jd_vec).reshape(1, -1)
        )[0][0]
        
        # Scale similarity for human-readable display with better discrimination
        scaling_mode = st.session_state.get('scaling_mode', 'Improved (Better Discrimination)')
        
        if scaling_mode == "Linear":
            # Linear scaling: raw cosine similarity
            scaled_sim = sim
        elif scaling_mode == "Original (Sigmoid)":
            # Original sigmoid scaling
            scaled_sim = 1 / (1 + np.exp(-10 * (sim - 0.3)))
        else:  # "Improved (Better Discrimination)"
            # Use improved piecewise linear scaling
            if sim <= 0.1:
                # Very low scores: compress to 0-20%
                scaled_sim = sim * 2.0
            elif sim <= 0.3:
                # Low scores: linear scaling 20-60%
                scaled_sim = 0.2 + (sim - 0.1) * 2.0
            elif sim <= 0.7:
                # Medium scores: linear scaling 60-90%
                scaled_sim = 0.6 + (sim - 0.3) * 0.75
            else:
                # High scores: preserve differences in 90-100% range
                scaled_sim = 0.9 + (sim - 0.7) * 0.33
        
        # Ensure bounds and apply minimum threshold
        scaled_sim = max(0.0, min(1.0, scaled_sim))
        
        # Apply minimum threshold to prevent zero scores (unless truly zero)
        if scaled_sim == 0.0 and sim > 0.0:
            scaled_sim = 0.01  # Minimum 1% for non-zero raw similarities
        
        display_score = round(scaled_sim * 100, 1)  # Convert to percentage
        
        print(f"DEBUG: Raw cosine similarity: {sim:.4f}")
        print(f"DEBUG: Scaled similarity: {scaled_sim:.4f}")
        print(f"DEBUG: Display score: {display_score}%")
        
        return sim, display_score
    
    def debug_similarity_breakdown(self, parsed_resume: dict, jd_text: str) -> dict:
        """Debug similarity breakdown by section for interpretability."""
        jd_vec = self._get_embedding(jd_text, 'jd')
        breakdown = {}
        
        for section in ["summary", "skills", "experience", "education"]:
            text = parsed_resume.get(section, "")
            if not text or len(text.strip()) < 10:
                breakdown[section] = 0.0
                print(f"DEBUG: {section.title()} Similarity: 0.000 (empty/insufficient content)")
                continue
            
            vec = self._get_embedding(text, section)
            sim = cosine_similarity(
                np.array(vec).reshape(1, -1),
                np.array(jd_vec).reshape(1, -1)
            )[0][0]
            
            breakdown[section] = sim
            print(f"DEBUG: {section.title()} Similarity: {sim:.3f}")
        
        return breakdown
    
    def compute_improved_similarity_matrix(self):
        """Compute similarity matrix using improved preprocessing and weighted embeddings."""
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
                else:
                    sections = self.extract_sections(text)
            else:
                sections = self.extract_sections(text)
            
            resume_sections[name] = sections
        
        for name, text in self.job_descriptions.items():
            jd_requirements[name] = self.extract_jd_requirements_with_importance(text)
        
        # Compute improved similarities
        similarities = {}
        
        for jd_name, jd_reqs in jd_requirements.items():
            similarities[jd_name] = {}
            
            # Combine JD requirements into single text
            jd_text_parts = []
            for req_type, reqs in jd_reqs.items():
                if reqs:
                    jd_text_parts.extend([req[0] for req in reqs])
            jd_text = " ".join(jd_text_parts)
            
            for resume_name, resume_sections_data in resume_sections.items():
                print(f"\nDEBUG: Computing improved similarity for {resume_name} vs {jd_name}")
                
                # Debug: Check if resume sections are empty
                section_lengths = {k: len(v) if v else 0 for k, v in resume_sections_data.items()}
                print(f"DEBUG: Resume sections lengths: {section_lengths}")
                
                # Compute weighted similarity
                raw_sim, display_score = self.compute_weighted_similarity(resume_sections_data, jd_text)
                
                # Get debug breakdown
                breakdown = self.debug_similarity_breakdown(resume_sections_data, jd_text)
                
                # Store detailed results
                similarities[jd_name][resume_name] = {
                    'raw_similarity': raw_sim,
                    'display_score': display_score,
                    'breakdown': breakdown
                }
                
                print(f"DEBUG: Final result - Raw: {raw_sim:.4f}, Display: {display_score}%")
        
        # Convert to matrix format using display scores
        similarity_matrix = np.zeros((len(self.candidate_names), len(self.jd_names)))
        
        for jd_idx, jd_name in enumerate(self.jd_names):
            for resume_idx, resume_name in enumerate(self.candidate_names):
                if jd_name in similarities and resume_name in similarities[jd_name]:
                    # Use display score (0-100) converted to 0-1 range
                    similarity_matrix[resume_idx, jd_idx] = similarities[jd_name][resume_name]['display_score'] / 100
        
        end_time = time.time()
        st.info(f"🎯 Improved similarity matching completed in {end_time - start_time:.2f} seconds")
        
        # Store similarity matrix for get_top_matches method
        self.similarity_matrix = similarity_matrix
        
        return similarity_matrix, similarities
    
    def compute_graded_similarity_matrix(self):
        """Compute similarity matrix using graded scoring system."""
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
        
        for name, text in self.job_descriptions.items():
            jd_requirements[name] = self.extract_jd_requirements_with_importance(text)
        
        # Compute graded similarities
        similarities = {}
        
        for jd_name, jd_reqs in jd_requirements.items():
            similarities[jd_name] = {}
            
            for resume_name, resume_sections_data in resume_sections.items():
                # Compute graded similarity
                graded_scores = self.compute_graded_similarity(resume_sections_data, jd_reqs)
                
                # Store detailed results
                similarities[jd_name][resume_name] = {
                    'total': graded_scores['total_score'],
                    'required_skills': graded_scores['required_skills'],
                    'preferred_skills': graded_scores['preferred_skills'],
                    'yoe_score': graded_scores['yoe_score'],
                    'breakdown': graded_scores['breakdown']
                }
                
                print(f"DEBUG: Graded similarity for {resume_name} vs {jd_name}:")
                print(f"  Required skills: {graded_scores['required_skills']:.2f}")
                print(f"  Preferred skills: {graded_scores['preferred_skills']:.2f}")
                print(f"  YOE score: {graded_scores['yoe_score']:.2f}")
                print(f"  Total score: {graded_scores['total_score']:.2f}")
        
        # Convert to matrix format
        similarity_matrix = np.zeros((len(self.candidate_names), len(self.jd_names)))
        
        for jd_idx, jd_name in enumerate(self.jd_names):
            for resume_idx, resume_name in enumerate(self.candidate_names):
                if jd_name in similarities and resume_name in similarities[jd_name]:
                    similarity_matrix[resume_idx, jd_idx] = similarities[jd_name][resume_name]['total']
        
        end_time = time.time()
        st.info(f"🎯 Graded scoring matching completed in {end_time - start_time:.2f} seconds")
        
        # Store similarity matrix for get_top_matches method
        self.similarity_matrix = similarity_matrix
        
        return similarity_matrix, similarities
    
    def diagnose_problematic_resume(self, resume_name: str) -> dict:
        """Diagnose a specific resume that's showing zero scores."""
        print(f"\n🔍 DIAGNOSING PROBLEMATIC RESUME: {resume_name}")
        print("=" * 60)
        
        diagnosis = {
            'resume_name': resume_name,
            'file_exists': False,
            'text_extraction': False,
            'section_extraction': False,
            'sections_content': {},
            'raw_text_length': 0,
            'embedding_generation': False,
            'issues': []
        }
        
        try:
            # Find the resume file
            resume_files = list(Path("/Users/junfeibai/Desktop/5560/test/").glob("*"))
            resume_file = None
            
            for file_path in resume_files:
                if resume_name.lower().replace('-resume-', '').replace('-', ' ') in file_path.name.lower():
                    resume_file = file_path
                    break
            
            if not resume_file:
                diagnosis['issues'].append("Resume file not found")
                return diagnosis
            
            diagnosis['file_exists'] = True
            print(f"✅ Found file: {resume_file}")
            
            # Test text extraction
            try:
                raw_text = self.processor.extract_text(str(resume_file))
                diagnosis['raw_text_length'] = len(raw_text)
                diagnosis['text_extraction'] = True
                print(f"✅ Text extraction: {len(raw_text)} characters")
                print(f"   Preview: {raw_text[:200]}...")
            except Exception as e:
                diagnosis['issues'].append(f"Text extraction failed: {e}")
                print(f"❌ Text extraction failed: {e}")
                return diagnosis
            
            # Test section extraction
            try:
                sections = self.extract_sections_enhanced(str(resume_file))
                diagnosis['section_extraction'] = True
                diagnosis['sections_content'] = sections
                
                print(f"✅ Section extraction successful")
                for section, content in sections.items():
                    if section != 'raw_text':
                        print(f"   {section}: {len(content)} chars - '{content[:50]}...'")
                
            except Exception as e:
                diagnosis['issues'].append(f"Section extraction failed: {e}")
                print(f"❌ Section extraction failed: {e}")
                return diagnosis
            
            # Test embedding generation
            try:
                embedding = self.get_weighted_resume_embedding(sections)
                diagnosis['embedding_generation'] = True
                print(f"✅ Embedding generation: {len(embedding)} dimensions")
                print(f"   Non-zero elements: {np.count_nonzero(embedding)}")
                
                if np.all(embedding == 0):
                    diagnosis['issues'].append("Generated zero embedding vector")
                    print("❌ Generated zero embedding vector!")
                else:
                    print("✅ Non-zero embedding generated")
                    
            except Exception as e:
                diagnosis['issues'].append(f"Embedding generation failed: {e}")
                print(f"❌ Embedding generation failed: {e}")
            
        except Exception as e:
            diagnosis['issues'].append(f"Diagnosis failed: {e}")
            print(f"❌ Diagnosis failed: {e}")
        
        print(f"\n📊 DIAGNOSIS SUMMARY:")
        print(f"   File exists: {diagnosis['file_exists']}")
        print(f"   Text extraction: {diagnosis['text_extraction']}")
        print(f"   Section extraction: {diagnosis['section_extraction']}")
        print(f"   Embedding generation: {diagnosis['embedding_generation']}")
        print(f"   Issues: {len(diagnosis['issues'])}")
        
        if diagnosis['issues']:
            print(f"\n🚨 ISSUES FOUND:")
            for issue in diagnosis['issues']:
                print(f"   - {issue}")
        
        return diagnosis
    
    def _compute_section_similarity(self, resume_text: str, jd_requirements: List[str], section_name: str = "content", parsed_sections: dict = None) -> float:
        """Compute similarity between resume section and JD requirements with validation."""
        if not resume_text.strip() or not jd_requirements:
            return 0.0
        
        # Combine JD requirements
        jd_text = " ".join(jd_requirements)
        
        # Get embeddings with validation and fallback
        resume_embedding = self._get_embedding(resume_text, section_name, parsed_sections)
        jd_embedding = self._get_embedding(jd_text, f"{section_name}_jd")
        
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
    
    def _validate_content_for_embedding(self, text: str, section_name: str = "content") -> tuple[str, bool]:
        """Validate content length and quality before embedding."""
        if not text or not text.strip():
            return "", False
        
        text = text.strip()
        
        # Check minimum length requirements
        min_lengths = {
            'summary': 20,
            'education': 15,
            'skills': 10,
            'experience': 30,
            'content': 10
        }
        
        min_length = min_lengths.get(section_name, 10)
        
        if len(text) < min_length:
            print(f"WARNING: {section_name} content too short ({len(text)} chars < {min_length} min)")
            return text, False
        
        # Check for meaningful content (not just repeated characters or symbols)
        unique_chars = len(set(text.lower()))
        if unique_chars < 5:
            print(f"WARNING: {section_name} content lacks diversity ({unique_chars} unique chars)")
            return text, False
        
        # Check for excessive whitespace or special characters
        alpha_ratio = sum(1 for c in text if c.isalpha()) / len(text)
        if alpha_ratio < 0.3:
            print(f"WARNING: {section_name} content has low alphabetic ratio ({alpha_ratio:.2f})")
            return text, False
        
        return text, True
    
    def _create_fallback_content(self, parsed_sections: dict, section_name: str) -> str:
        """Create fallback content when primary section is missing or insufficient."""
        fallback_strategies = {
            'summary': [
                ('experience', 200),  # First 200 chars of experience
                ('education', 100),   # First 100 chars of education
                ('skills', 50)        # First 50 chars of skills
            ],
            'experience': [
                ('summary', 150),     # Use summary if experience is missing
                ('skills', 100)       # Fallback to skills
            ],
            'education': [
                ('summary', 100),     # Use summary if education is missing
                ('experience', 50)    # Fallback to experience
            ],
            'skills': [
                ('summary', 100),     # Use summary if skills are missing
                ('experience', 50)    # Fallback to experience
            ]
        }
        
        strategies = fallback_strategies.get(section_name, [])
        fallback_parts = []
        
        for fallback_section, max_length in strategies:
            if fallback_section in parsed_sections and parsed_sections[fallback_section]:
                content = parsed_sections[fallback_section].strip()
                if content:
                    # Truncate to max_length and add to fallback
                    truncated = content[:max_length]
                    fallback_parts.append(truncated)
                    print(f"DEBUG: Using {fallback_section} as fallback for {section_name}")
        
        if fallback_parts:
            fallback_text = " ".join(fallback_parts)
            print(f"DEBUG: Created fallback {section_name} content: {len(fallback_text)} chars")
            return fallback_text
        
        return ""
    
    def _get_embedding(self, text: str, section_name: str = "content", parsed_sections: dict = None) -> np.ndarray:
        """Get embedding for text with content validation and fallback."""
        cache_key = hash(text)
        if cache_key in self.embeddings_cache:
            return self.embeddings_cache[cache_key]
        
        # Validate content length and quality
        validated_text, is_valid = self._validate_content_for_embedding(text, section_name)
        
        # If content is invalid and we have parsed sections, try fallback
        if not is_valid and parsed_sections:
            fallback_text = self._create_fallback_content(parsed_sections, section_name)
            if fallback_text:
                validated_text, is_valid = self._validate_content_for_embedding(fallback_text, section_name)
                if is_valid:
                    print(f"DEBUG: Using fallback content for {section_name}")
                    text = fallback_text
        
        # If still invalid, use original text but log warning
        if not is_valid:
            print(f"WARNING: Proceeding with potentially low-quality {section_name} content")
            validated_text = text
        
        # Generate embedding
        try:
            embedding = self.model.encode(validated_text)
            self.embeddings_cache[cache_key] = embedding
            return embedding
        except Exception as e:
            print(f"ERROR: Failed to generate embedding for {section_name}: {e}")
            # Return zero embedding as fallback
            return np.zeros(384)  # Default embedding size for all-MiniLM-L6-v2
    
    def generate_explanation(self, resume_name: str, jd_name: str, similarities: Dict) -> str:
        """Generate natural language explanation for match."""
        if jd_name not in similarities or resume_name not in similarities[jd_name]:
            return "No match data available."
        
        match_data = similarities[jd_name][resume_name]
        
        # Handle different data structures from different modes
        if 'breakdown' in match_data:
            # Improved similarity mode structure
            sections = match_data['breakdown']
            total_score = match_data.get('display_score', 0) / 100  # Convert percentage to decimal
        elif 'sections' in match_data:
            # Semantic mode structure
            sections = match_data['sections']
            total_score = match_data.get('total', 0)
        else:
            return "No section data available for explanation."
        
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
        
        # Extract summary/objective with enhanced detection
        in_summary = False
        summary_keywords = [
            'summary', 'objective', 'profile', 'about', 'overview', 
            'professional summary', 'professional overview', 'about me',
            'executive summary', 'career summary', 'personal summary',
            'professional profile', 'career objective', 'personal statement'
        ]
        
        for line in lines:
            line_lower = line.lower()
            # Check for summary section headers
            if any(keyword in line_lower for keyword in summary_keywords):
                in_summary = True
                continue
            elif in_summary and line.strip():
                # Stop if we hit another major section
                if any(word in line_lower for word in ['experience', 'education', 'skills', 'contact', 'work history', 'employment']):
                    break
                summary['summary'] += line.strip() + ' '
        
        # Add fallback: if summary is missing, concatenate experience + education
        if not summary['summary'].strip():
            fallback_parts = []
            if summary['experience']:
                fallback_parts.extend(summary['experience'][:2])  # First 2 experience entries
            if summary['education']:
                fallback_parts.extend(summary['education'][:1])   # First education entry
            if fallback_parts:
                summary['summary'] = ' '.join(fallback_parts)
        
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

def create_heatmap(similarity_matrix: np.ndarray, candidate_names: List[str], jd_names: List[str]):
    """Create an interactive heatmap of similarity scores."""
    
    # Ensure unique names to avoid non-unique index error
    unique_candidate_names = []
    unique_jd_names = []
    
    # Handle duplicate candidate names
    candidate_name_counts = {}
    for name in candidate_names:
        if name in candidate_name_counts:
            candidate_name_counts[name] += 1
            unique_candidate_names.append(f"{name} ({candidate_name_counts[name]})")
        else:
            candidate_name_counts[name] = 1
            unique_candidate_names.append(name)
    
    # Handle duplicate JD names
    jd_name_counts = {}
    for name in jd_names:
        if name in jd_name_counts:
            jd_name_counts[name] += 1
            unique_jd_names.append(f"{name} ({jd_name_counts[name]})")
        else:
            jd_name_counts[name] = 1
            unique_jd_names.append(name)
    
    df = pd.DataFrame(
        similarity_matrix,
        index=unique_candidate_names,
        columns=unique_jd_names
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
        ["🧠 Semantic (Sentence-BERT) Mode", "🚀 Improved Similarity Mode"],
        help="Choose between semantic matching or improved similarity with preprocessing"
    )
    
    # Show mode indicator
    if "Semantic" in matching_mode:
        st.sidebar.success("🧠 **Semantic Mode Active**")
        st.sidebar.info("**Features:** Sentence-BERT embeddings, section breakdown, professional summaries")
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            st.sidebar.error("⚠️ SentenceTransformers not installed. Please run: `pip install sentence-transformers`")
    elif "Improved Similarity" in matching_mode:
        st.sidebar.success("🚀 **Improved Similarity Mode Active**")
        st.sidebar.info("**Features:** Text preprocessing, weighted embeddings, improved scaling")
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            st.sidebar.error("⚠️ SentenceTransformers not installed. Please run: `pip install sentence-transformers`")
        
        # Scaling mode controls for improved similarity
        st.sidebar.subheader("📊 Scaling Controls")
        scaling_mode = st.sidebar.selectbox(
            "Scaling Method",
            ["Improved (Better Discrimination)", "Original (Sigmoid)", "Linear"],
            help="Choose how similarity scores are scaled for display"
        )
        
        if scaling_mode == "Original (Sigmoid)":
            st.sidebar.warning("⚠️ Original sigmoid scaling may compress high scores")
        elif scaling_mode == "Improved (Better Discrimination)":
            st.sidebar.success("✅ Improved scaling preserves differences in high scores")
        else:
            st.sidebar.info("ℹ️ Linear scaling shows raw cosine similarity")
        
        # Store scaling mode in session state
        st.session_state.scaling_mode = scaling_mode
    
    # Semantic mode controls
    if "Semantic" in matching_mode:
        st.sidebar.subheader("🎛️ Weight Controls")
        
        # Initialize weights in session state if not present (matching improved algorithm defaults)
        if 'weights' not in st.session_state:
            st.session_state.weights = {
                'education': 0.1,
                'skills': 0.4,
                'experience': 0.4,
                'summary': 0.2
            }
        
        # Get raw slider values
        raw_education_weight = st.sidebar.slider(
            "Education Weight",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state.weights['education'],
            step=0.05,
            help="Weight for education section matching"
        )
        raw_skills_weight = st.sidebar.slider(
            "Skills Weight",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state.weights['skills'],
            step=0.05,
            help="Weight for skills section matching"
        )
        raw_experience_weight = st.sidebar.slider(
            "Experience Weight",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state.weights['experience'],
            step=0.05,
            help="Weight for experience section matching"
        )
        raw_summary_weight = st.sidebar.slider(
            "Summary Weight",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state.weights['summary'],
            step=0.05,
            help="Weight for summary section matching"
        )
        
        # Check if raw weights have changed (before normalization)
        current_raw_weights = {
            'education': raw_education_weight,
            'skills': raw_skills_weight,
            'experience': raw_experience_weight,
            'summary': raw_summary_weight
        }
        
        weights_changed = current_raw_weights != st.session_state.weights
        if weights_changed:
            st.session_state.weights = current_raw_weights
            # Clear similarity data to force recomputation
            if 'similarity_matrix' in st.session_state:
                del st.session_state.similarity_matrix
            if 'section_matrices' in st.session_state:
                del st.session_state.section_matrices
            if 'similarities' in st.session_state:
                del st.session_state.similarities
            st.session_state.weights_changed = True
            st.sidebar.success("🔄 Weights updated! Similarity will be recomputed.")
        
        # Normalize weights for computation (after change detection)
        total_weight = raw_education_weight + raw_skills_weight + raw_experience_weight + raw_summary_weight
        if total_weight > 0:
            education_weight = raw_education_weight / total_weight
            skills_weight = raw_skills_weight / total_weight
            experience_weight = raw_experience_weight / total_weight
            summary_weight = raw_summary_weight / total_weight
        else:
            # Fallback to equal weights if total is 0
            education_weight = skills_weight = experience_weight = summary_weight = 0.25
        
        # Show current normalized weights
        st.sidebar.markdown("**Current Weights:**")
        st.sidebar.write(f"Education: {education_weight:.2f}")
        st.sidebar.write(f"Skills: {skills_weight:.2f}")
        st.sidebar.write(f"Experience: {experience_weight:.2f}")
        st.sidebar.write(f"Summary: {summary_weight:.2f}")
    
    # Weighted embedding mode controls
    elif "Weighted Embedding" in matching_mode:
        st.sidebar.subheader("🎯 Weighted Embedding Controls")
        st.sidebar.info("**High-level alignment focus:** Summary (50%) + Skills (30%) + Experience (20%)")
        
        summary_weight = st.sidebar.slider(
            "Summary Weight",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.05,
            help="Weight for summary section (high-level alignment terms like 'CTR', 'MLflow')"
        )
        skills_weight = st.sidebar.slider(
            "Skills Weight",
            min_value=0.0,
            max_value=1.0,
            value=0.3,
            step=0.05,
            help="Weight for skills section (technical alignment)"
        )
        experience_weight = st.sidebar.slider(
            "Experience Weight",
            min_value=0.0,
            max_value=1.0,
            value=0.2,
            step=0.05,
            help="Weight for experience section (detailed alignment)"
        )
        
        # Normalize weights
        total_weight = summary_weight + skills_weight + experience_weight
        if total_weight > 0:
            summary_weight /= total_weight
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
    # Check if we need to recreate matcher due to mode change
    matcher_needs_recreation = (
        'matcher' not in st.session_state or 
        'matching_mode' not in st.session_state or 
        st.session_state.matching_mode != matching_mode
    )
    
    if st.sidebar.button("🔄 Load Documents") or matcher_needs_recreation:
        with st.spinner("Loading documents..."):
            if "Semantic" in matching_mode or "Improved Similarity" in matching_mode:
                matcher = ResumeSemanticMatcher(data_dir)
            else:
                matcher = ResumeJDMatcher(data_dir)
            
            matcher.load_documents()
            
            if matcher.resumes and matcher.job_descriptions:
                # Clear any existing similarity data to prevent dimension mismatches
                if 'similarity_matrix' in st.session_state:
                    del st.session_state.similarity_matrix
                if 'section_matrices' in st.session_state:
                    del st.session_state.section_matrices
                if 'similarities' in st.session_state:
                    del st.session_state.similarities
                
                st.session_state.matcher = matcher
                st.session_state.matching_mode = matching_mode
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
        
        # Check if we need to recompute due to weight changes
        should_recompute = (
            st.button("🔍 Compute Similarity", type="primary") or 
            st.session_state.get('weights_changed', False)
        )
        
        if should_recompute:
            if "Semantic" in matching_mode:
                with st.spinner("Computing semantic similarity with Sentence-BERT..."):
                    result = matcher.compute_semantic_similarity(
                        education_weight=education_weight,
                        skills_weight=skills_weight,
                        experience_weight=experience_weight,
                        summary_weight=summary_weight
                    )
                    
                    if result is not None:
                        similarity_matrix, section_matrices, similarities = result
                        st.session_state.similarity_matrix = similarity_matrix
                        st.session_state.section_matrices = section_matrices
                        st.session_state.similarities = similarities
                        st.session_state.weights_changed = False  # Reset the flag
                        st.success("✅ Semantic similarity computation completed!")
                    else:
                        st.error("❌ Failed to compute semantic similarity!")
            elif "Improved Similarity" in matching_mode:
                with st.spinner("Computing improved similarity with preprocessing..."):
                    result = matcher.compute_improved_similarity_matrix()
                    
                    if result is not None:
                        similarity_matrix, similarities = result
                        st.session_state.similarity_matrix = similarity_matrix
                        st.session_state.similarities = similarities
                        st.session_state.weights_changed = False  # Reset the flag
                        st.success("✅ Improved similarity computation completed!")
                    else:
                        st.error("❌ Failed to compute improved similarity!")
            else:
                st.error("❌ Invalid matching mode selected!")
    
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
            tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(["🔥 Heatmap", "📈 Top Matches", "📋 Detailed Scores", "🎯 Section Breakdown", "👤 Resume Details", "📄 Job Description", "✨ Professional Summaries"])
        elif "Improved Similarity" in current_mode:
            tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(["🔥 Heatmap", "📈 Top Matches", "📋 Detailed Scores", "🚀 Similarity Breakdown", "👤 Resume Details", "📄 Job Description", "✨ Professional Summaries"])
        else:
            st.error("❌ Invalid mode configuration!")
        
        with tab1:
            st.subheader("Similarity Heatmap")
            
            # Debug: Check dimensions
            print(f"DEBUG: Similarity matrix shape: {similarity_matrix.shape}")
            print(f"DEBUG: Candidate names count: {len(matcher.candidate_names)}")
            print(f"DEBUG: JD names count: {len(matcher.jd_names)}")
            
            # Ensure matrix dimensions match candidate names
            if similarity_matrix.shape[0] != len(matcher.candidate_names):
                st.warning(f"⚠️ Matrix shape mismatch: {similarity_matrix.shape[0]} rows vs {len(matcher.candidate_names)} candidates")
                # Create a new matrix with correct dimensions
                new_matrix = np.zeros((len(matcher.candidate_names), len(matcher.jd_names)))
                # Copy existing data if possible
                min_rows = min(similarity_matrix.shape[0], len(matcher.candidate_names))
                min_cols = min(similarity_matrix.shape[1], len(matcher.jd_names))
                new_matrix[:min_rows, :min_cols] = similarity_matrix[:min_rows, :min_cols]
                similarity_matrix = new_matrix
                st.info(f"✅ Adjusted matrix to shape: {similarity_matrix.shape}")
            
            if similarity_matrix.shape[1] != len(matcher.jd_names):
                st.warning(f"⚠️ Matrix shape mismatch: {similarity_matrix.shape[1]} cols vs {len(matcher.jd_names)} JDs")
                # Create a new matrix with correct dimensions
                new_matrix = np.zeros((len(matcher.candidate_names), len(matcher.jd_names)))
                # Copy existing data if possible
                min_rows = min(similarity_matrix.shape[0], len(matcher.candidate_names))
                min_cols = min(similarity_matrix.shape[1], len(matcher.jd_names))
                new_matrix[:min_rows, :min_cols] = similarity_matrix[:min_rows, :min_cols]
                similarity_matrix = new_matrix
                st.info(f"✅ Adjusted matrix to shape: {similarity_matrix.shape}")
            
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
            
            # Ensure unique names to avoid non-unique index error
            unique_candidate_names = []
            unique_jd_names = []
            
            # Handle duplicate candidate names
            candidate_name_counts = {}
            for name in matcher.candidate_names:
                if name in candidate_name_counts:
                    candidate_name_counts[name] += 1
                    unique_candidate_names.append(f"{name} ({candidate_name_counts[name]})")
                else:
                    candidate_name_counts[name] = 1
                    unique_candidate_names.append(name)
            
            # Handle duplicate JD names
            jd_name_counts = {}
            for name in matcher.jd_names:
                if name in jd_name_counts:
                    jd_name_counts[name] += 1
                    unique_jd_names.append(f"{name} ({jd_name_counts[name]})")
                else:
                    jd_name_counts[name] = 1
                    unique_jd_names.append(name)
            
            detailed_df = pd.DataFrame(
                similarity_matrix,
                index=unique_candidate_names,
                columns=unique_jd_names
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
        
        # Graded breakdown tab (only for graded scoring mode)
        if "Graded Scoring" in current_mode and 'similarities' in st.session_state:
            with tab4:
                st.subheader("⭐ Graded Scoring Breakdown")
                
                similarities = st.session_state.similarities
                
                # Select candidate and JD for detailed breakdown
                col1, col2 = st.columns(2)
                with col1:
                    selected_candidate = st.selectbox(
                        "Select Candidate:",
                        matcher.candidate_names,
                        key="graded_candidate"
                    )
                with col2:
                    selected_jd = st.selectbox(
                        "Select Job Description:",
                        matcher.jd_names,
                        key="graded_jd"
                    )
                
                if selected_jd in similarities and selected_candidate in similarities[selected_jd]:
                    match_data = similarities[selected_jd][selected_candidate]
                    
                    # Display score breakdown
                    st.subheader("📊 Score Breakdown")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric(
                            "Required Skills",
                            f"{match_data['required_skills']:.2f}",
                            help="Binary gates for must-have skills"
                        )
                    with col2:
                        st.metric(
                            "Preferred Skills",
                            f"{match_data['preferred_skills']:.2f}",
                            help="Weighted bonuses for nice-to-have skills"
                        )
                    with col3:
                        st.metric(
                            "YOE Score",
                            f"{match_data['yoe_score']:.2f}",
                            help="Years of experience scaling"
                        )
                    with col4:
                        st.metric(
                            "Total Score",
                            f"{match_data['total']:.2f}",
                            help="Composite score: 60% required + 30% preferred + 10% YOE"
                        )
                    
                    # Display detailed breakdown
                    st.subheader("🔍 Detailed Breakdown")
                    
                    for item in match_data['breakdown']:
                        if item.startswith("✅"):
                            st.success(item)
                        elif item.startswith("❌"):
                            st.error(item)
                        elif item.startswith("🎯"):
                            st.info(item)
                        elif item.startswith("⚪"):
                            st.warning(item)
                        elif item.startswith("📅"):
                            st.info(item)
                        else:
                            st.write(item)
                    
                    # Display formula explanation
                    st.subheader("🧮 Scoring Formula")
                    st.info("""
                    **Composite Score Formula:**
                    - Required Skills: 60% weight (binary gates)
                    - Preferred Skills: 30% weight (weighted bonuses)
                    - Years of Experience: 10% weight (linear scaling)
                    
                    **Skill Similarity Examples:**
                    - Kafka ↔ Flink: 0.8 similarity
                    - PyTorch ↔ TensorFlow: 0.8 similarity
                    - MLflow ↔ Weights & Biases: 0.7 similarity
                    """)
        
                else:
                    st.warning("No similarity data available for the selected candidate and job description.")
        
        # Similarity breakdown tab (only for improved similarity mode)
        if "Improved Similarity" in current_mode and 'similarities' in st.session_state:
            with tab4:
                st.subheader("🚀 Improved Similarity Breakdown")
                
                similarities = st.session_state.similarities
                
                # Select candidate and JD for detailed breakdown
                col1, col2 = st.columns(2)
                with col1:
                    selected_candidate = st.selectbox(
                        "Select Candidate:",
                        matcher.candidate_names,
                        key="improved_candidate"
                    )
                with col2:
                    selected_jd = st.selectbox(
                        "Select Job Description:",
                        matcher.jd_names,
                        key="improved_jd"
                    )
                
                if selected_jd in similarities and selected_candidate in similarities[selected_jd]:
                    match_data = similarities[selected_jd][selected_candidate]
                    
                    # Display overall metrics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Raw Similarity", f"{match_data['raw_similarity']:.4f}")
                    with col2:
                        st.metric("Display Score", f"{match_data['display_score']:.1f}%")
                    with col3:
                        st.metric("Match Quality", 
                                "Excellent" if match_data['display_score'] > 80 else
                                "Good" if match_data['display_score'] > 60 else
                                "Fair" if match_data['display_score'] > 40 else "Poor")
                    
                    # Section breakdown
                    st.subheader("📊 Section-by-Section Breakdown")
                    breakdown = match_data['breakdown']
                    
                    breakdown_data = []
                    for section, score in breakdown.items():
                        breakdown_data.append({
                            'Section': section.title(),
                            'Raw Similarity': f"{score:.4f}",
                            'Display Score': f"{score * 100:.1f}%",
                            'Quality': "High" if score > 0.6 else "Medium" if score > 0.3 else "Low"
                        })
                    
                    breakdown_df = pd.DataFrame(breakdown_data)
                    st.dataframe(breakdown_df, use_container_width=True)
                    
                    # Visual breakdown
                    st.subheader("📈 Similarity Visualization")
                    fig = px.bar(
                        breakdown_df, 
                        x='Section', 
                        y=[float(score) for score in breakdown_df['Raw Similarity']],
                        title=f"Section Similarity Scores: {selected_candidate} vs {selected_jd}",
                        color=[float(score) for score in breakdown_df['Raw Similarity']],
                        color_continuous_scale='RdYlGn'
                    )
                    fig.update_layout(yaxis_title="Similarity Score", showlegend=False)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Processing details
                    st.subheader("🔧 Processing Details")
                    st.info("""
                    **Improved Similarity Processing:**
                    - ✅ Text preprocessing removes noise (URLs, phone numbers, emails)
                    - ✅ Weighted embeddings emphasize skills (40%) and experience (40%)
                    - ✅ Sigmoid scaling improves score distribution
                    - ✅ Section-by-section analysis for interpretability
                    """)
                else:
                    st.warning("No similarity data available for the selected candidate and job description.")
        
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
        
        # Job Description tab
        with tab6:
            st.subheader("📄 Job Description Dashboard")
            st.markdown("*View and analyze job descriptions with summaries and requirements*")
            
            # Job description selection
            selected_jd = st.selectbox(
                "Select a job description to view:",
                matcher.jd_names,
                key="jd_selector"
            )
            
            if selected_jd and selected_jd in matcher.job_descriptions:
                jd_text = matcher.job_descriptions[selected_jd]
                
                # Display job description summary
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("### 📋 Job Information")
                    # Extract basic job info
                    lines = jd_text.split('\n')
                    title = ""
                    company = ""
                    location = ""
                    
                    for line in lines[:10]:  # Check first 10 lines
                        line = line.strip()
                        if not title and any(word in line.lower() for word in ['engineer', 'analyst', 'manager', 'developer', 'scientist']):
                            title = line
                        elif not company and any(word in line.lower() for word in ['inc', 'corp', 'llc', 'company', 'technologies']):
                            company = line
                        elif not location and any(word in line.lower() for word in ['remote', 'hybrid', 'office', 'location']):
                            location = line
                    
                    if title:
                        st.write(f"**Title:** {title}")
                    if company:
                        st.write(f"**Company:** {company}")
                    if location:
                        st.write(f"**Location:** {location}")
                    
                    # Show word count and key metrics
                    word_count = len(jd_text.split())
                    st.write(f"**Word Count:** {word_count}")
                    
                    # Extract key skills mentioned
                    skill_keywords = ['python', 'sql', 'java', 'javascript', 'react', 'angular', 'docker', 'kubernetes', 'aws', 'azure', 'machine learning', 'data science', 'analytics']
                    mentioned_skills = [skill for skill in skill_keywords if skill in jd_text.lower()]
                    
                    if mentioned_skills:
                        st.write(f"**Key Skills Mentioned:** {', '.join(mentioned_skills[:8])}")
                
                with col2:
                    st.markdown("### 📝 Job Description Preview")
                    # Show first 300 characters
                    preview_text = jd_text[:300] + "..." if len(jd_text) > 300 else jd_text
                    st.text_area("", preview_text, height=200, disabled=True)
                
                # Full job description
                st.markdown("### 📄 Full Job Description")
                with st.expander("View Complete Job Description", expanded=False):
                    st.text(jd_text)
                
                # Requirements extraction
                st.markdown("### ✅ Key Requirements")
                requirements = []
                
                # Look for requirement patterns
                lines = jd_text.split('\n')
                for line in lines:
                    line = line.strip()
                    if line and (line.startswith('•') or line.startswith('-') or line.startswith('*') or 
                               any(word in line.lower() for word in ['required', 'must have', 'should have', 'experience', 'skills'])):
                        requirements.append(line)
                
                if requirements:
                    for req in requirements[:10]:  # Show first 10 requirements
                        st.write(f"• {req}")
                else:
                    st.write("No specific requirements found in structured format.")
                
                # Top candidates for this job
                if 'similarity_matrix' in st.session_state:
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
                    
                    # Show best match details
                    if len(top_indices) > 0:
                        best_candidate = matcher.candidate_names[top_indices[0]]
                        best_score = jd_scores[top_indices[0]]
                        
                        st.success(f"🎯 **Best Match:** {best_candidate} with {best_score * 100:.1f}% similarity")
            else:
                st.info("Please select a job description to view its details.")
        
        # Professional Summaries tab (available in both modes)
        with tab7:
            st.subheader("✨ AI-Generated Professional Summaries")
            st.markdown("*Intelligent summaries generated from experience and skills analysis*")
            
            # Generate summaries for all candidates
            summaries_data = []
            
            for candidate_name in matcher.candidate_names:
                # Get the resume text
                resume_text = matcher.resumes.get(candidate_name, "")
                if not resume_text:
                    continue
                
                # Extract sections using the matcher's method
                if hasattr(matcher, 'extract_sections'):
                    resume_sections = matcher.extract_sections(resume_text)
                else:
                    # Fallback to basic section extraction
                    resume_sections = {
                        'summary': '',
                        'experience': '',
                        'skills': '',
                        'education': ''
                    }
                
                # Generate professional summary
                if hasattr(matcher, 'generate_professional_summary'):
                    professional_summary = matcher.generate_professional_summary(resume_sections)
                else:
                    professional_summary = "Summary generation not available in this mode."
                
                # Get original summary for comparison
                original_summary = resume_sections.get('summary', 'No original summary available')
                
                summaries_data.append({
                    'Candidate': candidate_name,
                    'AI-Generated Summary': professional_summary,
                    'Original Summary': original_summary[:200] + "..." if len(original_summary) > 200 else original_summary,
                    'Experience Length': len(resume_sections.get('experience', '')),
                    'Skills Count': len(resume_sections.get('skills', '').split(',')) if resume_sections.get('skills') else 0
                })
            
            if summaries_data:
                summaries_df = pd.DataFrame(summaries_data)
                
                # Display metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Candidates", len(summaries_df))
                with col2:
                    avg_length = summaries_df['AI-Generated Summary'].str.len().mean()
                    st.metric("Avg Summary Length", f"{avg_length:.0f} chars")
                with col3:
                    summaries_with_experience = len(summaries_df[summaries_df['Experience Length'] > 0])
                    st.metric("With Experience Data", f"{summaries_with_experience}/{len(summaries_df)}")
                
                # Display summaries
                st.subheader("📋 Generated Summaries")
                
                for idx, row in summaries_df.iterrows():
                    with st.expander(f"👤 {row['Candidate']}", expanded=False):
                        st.markdown("**🤖 AI-Generated Summary:**")
                        st.info(row['AI-Generated Summary'])
                        
                        if row['Original Summary'] != 'No original summary available':
                            st.markdown("**📄 Original Summary:**")
                            st.text(row['Original Summary'])
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Experience Length", f"{row['Experience Length']} chars")
                        with col2:
                            st.metric("Skills Count", row['Skills Count'])
                
                # Download option
                csv_data = summaries_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Professional Summaries as CSV",
                    data=csv_data,
                    file_name="professional_summaries.csv",
                    mime="text/csv"
                )
            else:
                st.warning("No candidate data available for summary generation.")
    
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
