"""
Base matcher class with shared functionality.
Eliminates code duplication between different matcher implementations.
"""
import re
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from ..config import config
from ..logging_config import get_logger
from ..utils.document_processor import DocumentProcessor
from ..utils.path_validation import validate_directory, validate_file_for_processing

logger = get_logger(__name__)


class BaseMatcher:
    """Base class for all matchers with common functionality."""
    
    def __init__(self, data_dir: str):
        """
        Initialize base matcher.
        
        Args:
            data_dir: Directory containing resumes and job descriptions
        """
        self.data_dir = Path(data_dir)
        self.processor = DocumentProcessor()
        self.resumes: Dict[str, str] = {}
        self.job_descriptions: Dict[str, str] = {}
        self.similarity_matrix: Optional[np.ndarray] = None
        self.candidate_names: List[str] = []
        self.jd_names: List[str] = []
        self.original_filenames: Dict[str, str] = {}
        self.extracted_names: Dict[str, str] = {}
    
    def load_documents(self) -> None:
        """
        Load all resumes and job descriptions from the directory.
        
        Raises:
            PathValidationError: If directory is invalid
        """
        try:
            # Validate directory
            self.data_dir = validate_directory(self.data_dir)
        except Exception as e:
            logger.error(f"Invalid directory {self.data_dir}: {e}")
            raise
        
        # Clear existing data to prevent duplicates
        self.resumes.clear()
        self.job_descriptions.clear()
        self.candidate_names.clear()
        self.jd_names.clear()
        self.original_filenames.clear()
        self.extracted_names.clear()
        
        # Load job descriptions first (files starting with "JD")
        jd_files = [
            f for f in self.data_dir.glob("*")
            if f.is_file() and f.name.lower().startswith("jd")
        ]
        
        loaded_jds = 0
        for file_path in jd_files:
            try:
                # Validate file
                validated_path = validate_file_for_processing(file_path, self.data_dir)
                text = self.processor.extract_text(validated_path)
                
                if text.strip():
                    name = validated_path.stem
                    self.job_descriptions[name] = text
                    self.jd_names.append(name)
                    loaded_jds += 1
                    logger.debug(f"Loaded JD: {name} from {validated_path.name}")
            except Exception as e:
                logger.warning(f"Failed to load JD file {file_path}: {e}")
                continue
        
        # Load ALL other files as candidate resumes
        all_files = [f for f in self.data_dir.glob("*") if f.is_file()]
        candidate_files = [
            f for f in all_files
            if not f.name.lower().startswith("jd")
        ]
        
        # Sort files by name for consistent ordering
        candidate_files.sort(key=lambda x: x.name.lower())
        
        loaded_resumes = 0
        for i, file_path in enumerate(candidate_files, 1):
            try:
                # Validate file
                validated_path = validate_file_for_processing(file_path, self.data_dir)
                text = self.processor.extract_text(validated_path)
                
                if text.strip():
                    # Extract candidate name from resume content
                    candidate_name = self.extract_candidate_name(text)
                    # Create standardized name: Name-resume
                    standardized_name = f"{candidate_name}-resume-{i}"
                    
                    self.resumes[standardized_name] = text
                    self.candidate_names.append(standardized_name)
                    self.original_filenames[standardized_name] = validated_path.name
                    self.extracted_names[standardized_name] = candidate_name
                    loaded_resumes += 1
                    logger.debug(
                        f"Loaded resume: {standardized_name} from {validated_path.name} "
                        f"(extracted name: {candidate_name})"
                    )
            except Exception as e:
                logger.warning(f"Failed to load resume file {file_path}: {e}")
                continue
        
        logger.info(
            f"Loaded {loaded_resumes} resumes and {loaded_jds} job descriptions "
            f"from {self.data_dir}"
        )
    
    def extract_candidate_name(self, text: str) -> str:
        """
        Extract candidate name from resume text.
        
        Args:
            text: Resume text
        
        Returns:
            Extracted candidate name or "Unknown Candidate"
        """
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
                name = re.sub(r'[^\w\s]', '', name).strip()
                if name and len(name.split()) >= 1:
                    logger.debug(f"Extracted name using pattern 1: {name}")
                    return name
            
            # Pattern 2: "Name (Title)" format
            match = re.match(r'^([A-Za-z\s]+)\s*\([^)]+\)', line)
            if match:
                name = match.group(1).strip()
                name = re.sub(r'[^\w\s]', '', name).strip()
                if len(name.split()) >= 1:
                    logger.debug(f"Extracted name using pattern 2: {name}")
                    return name
            
            # Pattern 3: Just name on first line (clean version)
            if i == 0 and len(line.split()) >= 1:
                clean_line = re.sub(
                    r'^(name|Name|contact|Contact)\s*:?\s*', '', line, flags=re.IGNORECASE
                )
                clean_line = re.sub(r'[^\w\s]', '', clean_line).strip()
                
                # Check if it looks like a name
                if (clean_line and
                    re.match(r'^[A-Za-z\s]+$', clean_line) and
                    2 <= len(clean_line) <= config.max_name_length and
                    not any(word.lower() in [
                        'contact', 'email', 'phone', 'address', 'summary', 'objective'
                    ] for word in clean_line.split())):
                    logger.debug(f"Extracted name using pattern 3: {clean_line}")
                    return clean_line
            
            # Pattern 4: Name followed by title on next line
            if i < len(lines) - 1:
                next_line = lines[i + 1].strip()
                clean_line = re.sub(r'[^\w\s]', '', line).strip()
                if (len(clean_line.split()) >= 1 and
                    not any(word.lower() in [
                        'contact', 'email', 'phone', 'address', 'summary', 'objective'
                    ] for word in clean_line.split()) and
                    any(word.lower() in [
                        'engineer', 'developer', 'scientist', 'analyst',
                        'manager', 'specialist'
                    ] for word in next_line.split())):
                    logger.debug(f"Extracted name using pattern 4: {clean_line}")
                    return clean_line
        
        # Fallback: return first meaningful line (cleaned)
        for line in lines[:5]:
            line = line.strip()
            if line:
                clean_line = re.sub(
                    r'^(name|Name|contact|Contact)\s*:?\s*', '', line, flags=re.IGNORECASE
                )
                clean_line = re.sub(r'[^\w\s]', '', clean_line).strip()
                if clean_line and len(clean_line.split()) >= 1:
                    logger.debug(f"Extracted name using fallback: {clean_line}")
                    return clean_line
        
        logger.warning("Could not extract candidate name, using 'Unknown Candidate'")
        return "Unknown Candidate"
    
    def preprocess_text(self, text: str) -> str:
        """
        Clean and preprocess text for better matching.
        
        Args:
            text: Raw text to preprocess
        
        Returns:
            Preprocessed text
        """
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters but keep spaces
        text = re.sub(r'[^\w\s]', ' ', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove common stop words (basic list)
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have',
            'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should'
        }
        words = text.split()
        words = [word for word in words if word not in stop_words and len(word) > 2]
        
        return ' '.join(words)
    
    def get_top_matches(self, top_k: int = 3) -> pd.DataFrame:
        """
        Get top matches for each job description.
        
        Args:
            top_k: Number of top matches to return
        
        Returns:
            DataFrame with top matches
        """
        if self.similarity_matrix is None:
            logger.warning("Similarity matrix not computed yet")
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
        """
        Extract key information from resume.
        
        Args:
            resume_text: Full resume text
        
        Returns:
            Dictionary with extracted information
        """
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
            elif any(city in line_lower for city in [
                'san francisco', 'new york', 'seattle', 'chicago',
                'boston', 'austin'
            ]):
                summary['location'] = line.strip()
        
        # Extract skills (look for skills sections)
        in_skills_section = False
        for line in lines:
            line_lower = line.lower()
            if 'skill' in line_lower and ('technical' in line_lower or 'core' in line_lower):
                in_skills_section = True
                continue
            elif in_skills_section and line.strip():
                if any(word in line_lower for word in [
                    'experience', 'education', 'work', 'project'
                ]):
                    in_skills_section = False
                    break
                # Extract skills from this line
                skills = re.findall(r'\b[A-Za-z][A-Za-z0-9\s]+\b', line)
                summary['skills'].extend([
                    skill.strip() for skill in skills if len(skill.strip()) > 2
                ])
        
        # Extract experience (look for job titles and companies)
        for line in lines:
            if (any(word in line.lower() for word in [
                'engineer', 'developer', 'scientist', 'analyst', 'manager'
            ]) and '|' in line):
                summary['experience'].append(line.strip())
        
        # Extract education
        for line in lines:
            if any(word in line.lower() for word in [
                'university', 'college', 'bachelor', 'master', 'phd', 'degree'
            ]):
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
            if any(keyword in line_lower for keyword in summary_keywords):
                in_summary = True
                continue
            elif in_summary and line.strip():
                if any(word in line_lower for word in [
                    'experience', 'education', 'skills', 'contact',
                    'work history', 'employment'
                ]):
                    break
                summary['summary'] += line.strip() + ' '
        
        # Add fallback: if summary is missing, concatenate experience + education
        if not summary['summary'].strip():
            fallback_parts = []
            if summary['experience']:
                fallback_parts.extend(summary['experience'][:2])
            if summary['education']:
                fallback_parts.extend(summary['education'][:1])
            if fallback_parts:
                summary['summary'] = ' '.join(fallback_parts)
        
        return summary
    
    def extract_jd_requirements(self, jd_text: str) -> Dict:
        """
        Extract requirements from job description.
        
        Args:
            jd_text: Job description text
        
        Returns:
            Dictionary with extracted requirements
        """
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
            if any(word in line.lower() for word in [
                'engineer', 'developer', 'scientist', 'analyst', 'manager'
            ]):
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
            if any(word in line_lower for word in [
                'requirement', 'qualification', 'must have', 'should have'
            ]):
                in_requirements = True
                continue
            elif in_requirements and line.strip():
                if any(word in line_lower for word in [
                    'responsibilit', 'benefit', 'compensation', 'salary'
                ]):
                    in_requirements = False
                    break
                if line.strip().startswith('-') or line.strip().startswith('•'):
                    requirements['requirements'].append(line.strip())
        
        # Extract skills
        for line in lines:
            if any(skill in line.lower() for skill in [
                'python', 'java', 'sql', 'javascript', 'react', 'aws', 'docker'
            ]):
                requirements['skills_required'].append(line.strip())
        
        return requirements
    
    def get_directory_info(self) -> Dict:
        """
        Get information about files in the directory.
        
        Returns:
            Dictionary with file information
        """
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

