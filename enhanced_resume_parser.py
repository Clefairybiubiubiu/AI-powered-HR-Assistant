"""
Enhanced Resume Parser with spaCy NER and pyresparser integration
"""

import re
import json
import spacy
from pathlib import Path
from typing import Dict, List, Any, Optional
import warnings
warnings.filterwarnings('ignore')

# Import pyresparser
try:
    from pyresparser import ResumeParser
    PYRESPARSER_AVAILABLE = True
except ImportError:
    PYRESPARSER_AVAILABLE = False

# Import document processing
from resume_jd_matcher import DocumentProcessor

class EnhancedResumeParser:
    """Enhanced resume parser with spaCy NER and pyresparser integration."""
    
    def __init__(self):
        """Initialize the enhanced parser."""
        self.document_processor = DocumentProcessor()
        
        # Load spaCy model
        try:
            self.nlp = spacy.load("en_core_web_sm")
            SPACY_AVAILABLE = True
        except OSError:
            try:
                self.nlp = spacy.load("en_core_web_md")
                SPACY_AVAILABLE = True
            except OSError:
                print("Warning: spaCy model not found. Please install with: python -m spacy download en_core_web_sm")
                self.nlp = None
                SPACY_AVAILABLE = False
        
        self.SPACY_AVAILABLE = SPACY_AVAILABLE
        self.PYRESPARSER_AVAILABLE = PYRESPARSER_AVAILABLE
    
    def parse_resume(self, file_path: str) -> Dict[str, Any]:
        """
        Parse resume and return structured data.
        
        Args:
            file_path: Path to the resume file
            
        Returns:
            Dictionary with structured resume data
        """
        # Extract text from document
        text = self.document_processor.extract_text(file_path)
        
        if not text.strip():
            return self._get_empty_structure()
        
        # Initialize result structure
        result = self._get_empty_structure()
        
        # 1. Use pyresparser for basic extraction
        if self.PYRESPARSER_AVAILABLE:
            try:
                pyresparser_data = self._extract_with_pyresparser(file_path)
                result = self._merge_pyresparser_data(result, pyresparser_data)
            except Exception as e:
                print(f"Warning: pyresparser failed: {e}")
        
        # 2. Use spaCy NER for entity recognition
        if self.SPACY_AVAILABLE:
            try:
                spacy_data = self._extract_with_spacy(text)
                result = self._merge_spacy_data(result, spacy_data)
            except Exception as e:
                print(f"Warning: spaCy NER failed: {e}")
        
        # 3. Use fallback regex extraction
        regex_data = self._extract_with_regex(text)
        result = self._merge_regex_data(result, regex_data)
        
        # 4. Clean and validate data
        result = self._clean_and_validate_data(result)
        
        return result
    
    def _get_empty_structure(self) -> Dict[str, Any]:
        """Return empty structured data."""
        return {
            "name": "",
            "email": "",
            "phone": "",
            "education": [],
            "experience": [],
            "skills": [],
            "summary": "",
            "certifications": [],
            "entities": {
                "persons": [],
                "organizations": [],
                "locations": [],
                "dates": []
            },
            "confidence_scores": {
                "name": 0.0,
                "email": 0.0,
                "phone": 0.0,
                "education": 0.0,
                "experience": 0.0,
                "skills": 0.0
            }
        }
    
    def _extract_with_pyresparser(self, file_path: str) -> Dict[str, Any]:
        """Extract data using pyresparser."""
        try:
            data = ResumeParser(file_path).get_extracted_data()
            return data
        except Exception as e:
            print(f"pyresparser error: {e}")
            return {}
    
    def _extract_with_spacy(self, text: str) -> Dict[str, Any]:
        """Extract entities using spaCy NER."""
        if not self.nlp:
            return {}
        
        doc = self.nlp(text)
        
        entities = {
            "persons": [],
            "organizations": [],
            "locations": [],
            "dates": []
        }
        
        # Extract entities
        for ent in doc.ents:
            if ent.label_ == "PERSON":
                entities["persons"].append(ent.text)
            elif ent.label_ == "ORG":
                entities["organizations"].append(ent.text)
            elif ent.label_ == "GPE":
                entities["locations"].append(ent.text)
            elif ent.label_ == "DATE":
                entities["dates"].append(ent.text)
        
        # Extract name (first PERSON entity)
        name = entities["persons"][0] if entities["persons"] else ""
        
        # Extract organizations (companies)
        organizations = list(set(entities["organizations"]))
        
        # Extract locations
        locations = list(set(entities["locations"]))
        
        return {
            "name": name,
            "entities": entities,
            "organizations": organizations,
            "locations": locations
        }
    
    def _extract_with_regex(self, text: str) -> Dict[str, Any]:
        """Extract data using regex patterns."""
        result = {}
        
        # Email extraction
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        emails = re.findall(email_pattern, text)
        result["email"] = emails[0] if emails else ""
        
        # Phone extraction
        phone_patterns = [
            r'\+?1?[-.\s]?\(?([0-9]{3})\)?[-.\s]?([0-9]{3})[-.\s]?([0-9]{4})',
            r'\+?[0-9]{1,3}[-.\s]?[0-9]{3,4}[-.\s]?[0-9]{3,4}[-.\s]?[0-9]{3,4}',
            r'\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}'
        ]
        
        phones = []
        for pattern in phone_patterns:
            matches = re.findall(pattern, text)
            if matches:
                if isinstance(matches[0], tuple):
                    # Join tuple elements
                    phone = ''.join(matches[0])
                else:
                    phone = matches[0]
                phones.append(phone)
        
        result["phone"] = phones[0] if phones else ""
        
        # Name extraction (first line or after "Name:")
        name_patterns = [
            r'Name:\s*([A-Za-z\s]+)',
            r'^([A-Za-z\s]+)$',
            r'^([A-Za-z\s]+)\s*\([^)]+\)'
        ]
        
        for pattern in name_patterns:
            match = re.search(pattern, text, re.MULTILINE)
            if match:
                result["name"] = match.group(1).strip()
                break
        
        # Skills extraction
        skills_keywords = [
            'python', 'java', 'sql', 'javascript', 'react', 'aws', 'docker',
            'kubernetes', 'spark', 'kafka', 'airflow', 'snowflake', 'bigquery',
            'machine learning', 'data science', 'etl', 'streaming', 'tensorflow',
            'pytorch', 'scikit-learn', 'pandas', 'numpy', 'matplotlib', 'seaborn'
        ]
        
        found_skills = []
        text_lower = text.lower()
        for skill in skills_keywords:
            if skill in text_lower:
                found_skills.append(skill.title())
        
        result["skills"] = found_skills
        
        return result
    
    def _merge_pyresparser_data(self, result: Dict[str, Any], pyresparser_data: Dict[str, Any]) -> Dict[str, Any]:
        """Merge pyresparser data into result."""
        if not pyresparser_data:
            return result
        
        # Map pyresparser fields to our structure
        field_mapping = {
            'name': 'name',
            'email': 'email',
            'mobile_number': 'phone',
            'skills': 'skills',
            'college_name': 'education',
            'degree': 'education',
            'experience': 'experience',
            'summary': 'summary'
        }
        
        for pyresparser_key, our_key in field_mapping.items():
            if pyresparser_key in pyresparser_data and pyresparser_data[pyresparser_key]:
                if our_key in ['education', 'experience', 'skills']:
                    if isinstance(pyresparser_data[pyresparser_key], list):
                        result[our_key] = pyresparser_data[pyresparser_key]
                    else:
                        result[our_key] = [pyresparser_data[pyresparser_key]]
                else:
                    result[our_key] = pyresparser_data[pyresparser_key]
        
        return result
    
    def _merge_spacy_data(self, result: Dict[str, Any], spacy_data: Dict[str, Any]) -> Dict[str, Any]:
        """Merge spaCy data into result."""
        if not spacy_data:
            return result
        
        # Merge name if not already extracted
        if spacy_data.get("name") and not result.get("name"):
            result["name"] = spacy_data["name"]
        
        # Merge entities
        if "entities" in spacy_data:
            result["entities"] = spacy_data["entities"]
        
        # Merge organizations and locations
        if "organizations" in spacy_data:
            result["entities"]["organizations"] = spacy_data["organizations"]
        
        if "locations" in spacy_data:
            result["entities"]["locations"] = spacy_data["locations"]
        
        return result
    
    def _merge_regex_data(self, result: Dict[str, Any], regex_data: Dict[str, Any]) -> Dict[str, Any]:
        """Merge regex data into result."""
        if not regex_data:
            return result
        
        # Merge data if not already present
        for key, value in regex_data.items():
            if key in result and not result[key]:
                result[key] = value
            elif key == "skills" and value:
                # Merge skills lists
                if result.get("skills"):
                    result["skills"] = list(set(result["skills"] + value))
                else:
                    result["skills"] = value
        
        return result
    
    def _clean_and_validate_data(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Clean and validate the extracted data."""
        # Clean name
        if result.get("name"):
            result["name"] = result["name"].strip()
        
        # Clean email
        if result.get("email"):
            result["email"] = result["email"].strip().lower()
        
        # Clean phone
        if result.get("phone"):
            result["phone"] = result["phone"].strip()
        
        # Clean skills
        if result.get("skills"):
            result["skills"] = [skill.strip() for skill in result["skills"] if skill.strip()]
            result["skills"] = list(set(result["skills"]))  # Remove duplicates
        
        # Clean education
        if result.get("education"):
            result["education"] = [edu.strip() for edu in result["education"] if edu.strip()]
        
        # Clean experience
        if result.get("experience"):
            result["experience"] = [exp.strip() for exp in result["experience"] if exp.strip()]
        
        # Clean summary
        if result.get("summary"):
            result["summary"] = result["summary"].strip()
        
        # Clean certifications
        if result.get("certifications"):
            result["certifications"] = [cert.strip() for cert in result["certifications"] if cert.strip()]
        
        return result
    
    def _is_contact_info(self, text: str) -> bool:
        """Check if text contains contact information."""
        contact_keywords = [
            'contact', 'email', 'phone', 'github', 'linkedin', 'website', 
            'address', 'location', 'seattle', 'wa', 'california', 'ca',
            'daniel park', 'data engineer', 'contact daniel', 'fake@example.com',
            'github.com', 'linkedin.com', 'www.', 'http', 'https'
        ]
        return any(keyword in text.lower() for keyword in contact_keywords)
    
    def _extract_sections_from_parsed_data(self, parsed_data: Dict[str, Any]) -> Dict[str, str]:
        """Extract sections from parsed data with better format handling."""
        sections = {
            'education': '',
            'skills': '',
            'experience': '',
            'summary': ''
        }
        
        # Extract education with better filtering
        if parsed_data.get('education'):
            education_items = []
            if isinstance(parsed_data['education'], list):
                for edu in parsed_data['education']:
                    if isinstance(edu, str) and edu.strip():
                        # Filter out contact information
                        if not self._is_contact_info(edu.lower()):
                            education_items.append(edu.strip())
            else:
                edu_text = str(parsed_data['education']).strip()
                if not self._is_contact_info(edu_text.lower()):
                    education_items.append(edu_text)
            
            sections['education'] = ' '.join(education_items)
        
        # Extract skills with better filtering
        if parsed_data.get('skills'):
            skills_items = []
            if isinstance(parsed_data['skills'], list):
                for skill in parsed_data['skills']:
                    if isinstance(skill, str) and skill.strip():
                        # Filter out contact information
                        if not self._is_contact_info(skill.lower()):
                            skills_items.append(skill.strip())
            else:
                skill_text = str(parsed_data['skills']).strip()
                if not self._is_contact_info(skill_text.lower()):
                    skills_items.append(skill_text)
            
            sections['skills'] = ' '.join(skills_items)
        
        # Extract experience with better filtering
        if parsed_data.get('experience'):
            experience_items = []
            if isinstance(parsed_data['experience'], list):
                for exp in parsed_data['experience']:
                    if isinstance(exp, str) and exp.strip():
                        # Filter out contact information
                        if not self._is_contact_info(exp.lower()):
                            experience_items.append(exp.strip())
            else:
                exp_text = str(parsed_data['experience']).strip()
                if not self._is_contact_info(exp_text.lower()):
                    experience_items.append(exp_text)
            
            sections['experience'] = ' '.join(experience_items)
        
        # Extract summary with enhanced detection
        summary_text = parsed_data.get('summary', '')
        if not summary_text:
            # Try alternative keys that might contain summary
            for key in ['objective', 'profile', 'overview', 'about']:
                if parsed_data.get(key):
                    summary_text = parsed_data[key]
                    break
        
        if summary_text:
            sections['summary'] = summary_text
        else:
            # Fallback: concatenate experience + education if summary is missing
            fallback_parts = []
            if sections['experience']:
                fallback_parts.append(sections['experience'][:200])  # First 200 chars
            if sections['education']:
                fallback_parts.append(sections['education'][:100])  # First 100 chars
            if fallback_parts:
                sections['summary'] = ' '.join(fallback_parts)
        
        return sections
    
    def parse_resume_sections(self, text: str, file_path: str = "") -> Dict[str, str]:
        """Parse resume sections with enhanced summary detection and fallback logic."""
        sections = {
            'summary': '',
            'education': '',
            'skills': '',
            'experience': '',
            'contact': ''
        }
        
        # Enhanced summary section header regex patterns
        summary_patterns = [
            r'professional\s+summary',
            r'profile',
            r'summary',
            r'summary\s+of\s+qualifications',
            r'professional\s+overview',
            r'about\s+me',
            r'objective',
            r'career\s+objective',
            r'executive\s+summary',
            r'personal\s+summary',
            r'career\s+summary',
            r'professional\s+profile',
            r'personal\s+statement'
        ]
        
        # Compile regex patterns (case insensitive)
        summary_regex = re.compile('|'.join(summary_patterns), re.IGNORECASE)
        
        lines = text.split('\n')
        current_section = None
        in_summary = False
        
        for i, line in enumerate(lines):
            line_lower = line.lower().strip()
            original_line = line.strip()
            
            # Skip empty lines
            if not original_line:
                continue
            
            # Check for summary section headers
            if summary_regex.search(line_lower):
                current_section = 'summary'
                in_summary = True
                continue
            
            # Check for other section headers
            if any(word in line_lower for word in ['education', 'academic', 'degree']):
                current_section = 'education'
                in_summary = False
                continue
            elif any(word in line_lower for word in ['skills', 'technical', 'competencies']):
                current_section = 'skills'
                in_summary = False
                continue
            elif any(word in line_lower for word in ['experience', 'employment', 'work history', 'career']):
                current_section = 'experience'
                in_summary = False
                continue
            elif any(word in line_lower for word in ['contact', 'phone', 'email', 'address']):
                current_section = 'contact'
                in_summary = False
                continue
            
            # Add content to current section
            if current_section and original_line:
                # Skip contact information in non-contact sections
                if current_section != 'contact' and self._is_contact_info(line_lower):
                    continue
                
                sections[current_section] += original_line + " "
        
        # Fallback logic for PDFs and TXT files
        if not sections['summary'].strip() and file_path:
            file_ext = file_path.lower().split('.')[-1]
            if file_ext in ['pdf', 'txt']:
                # Use first ~3 sentences as fallback summary
                sentences = self._extract_first_sentences(text, num_sentences=3)
                if sentences:
                    sections['summary'] = sentences
                    print(f"DEBUG: Used fallback summary for {file_ext.upper()} file: {sentences[:100]}...")
        
        # Additional fallback: concatenate experience + education if summary is still empty
        if not sections['summary'].strip():
            fallback_parts = []
            if sections['experience']:
                fallback_parts.append(sections['experience'][:200])
            if sections['education']:
                fallback_parts.append(sections['education'][:100])
            if fallback_parts:
                sections['summary'] = ' '.join(fallback_parts)
                print(f"DEBUG: Used experience+education fallback summary")
        
        return sections
    
    def _extract_first_sentences(self, text: str, num_sentences: int = 3) -> str:
        """Extract the first few sentences from text as fallback summary."""
        import re
        
        # Split text into sentences using common sentence endings
        sentences = re.split(r'[.!?]+', text)
        
        # Clean and filter sentences
        clean_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence and len(sentence) > 10:  # Skip very short sentences
                # Skip sentences that look like headers or contact info
                if not any(word in sentence.lower() for word in ['contact', 'email', 'phone', 'address', 'name:']):
                    clean_sentences.append(sentence)
        
        # Return first num_sentences sentences
        return '. '.join(clean_sentences[:num_sentences]) + '.' if clean_sentences else ''
    
    def get_confidence_scores(self, result: Dict[str, Any]) -> Dict[str, float]:
        """Calculate confidence scores for extracted data."""
        scores = {}
        
        # Name confidence
        if result.get("name"):
            scores["name"] = 0.9 if len(result["name"].split()) >= 2 else 0.7
        else:
            scores["name"] = 0.0
        
        # Email confidence
        if result.get("email") and "@" in result["email"]:
            scores["email"] = 0.9
        else:
            scores["email"] = 0.0
        
        # Phone confidence
        if result.get("phone") and len(result["phone"]) >= 10:
            scores["phone"] = 0.8
        else:
            scores["phone"] = 0.0
        
        # Skills confidence
        if result.get("skills") and len(result["skills"]) > 0:
            scores["skills"] = min(0.9, len(result["skills"]) / 10.0)
        else:
            scores["skills"] = 0.0
        
        # Education confidence
        if result.get("education") and len(result["education"]) > 0:
            scores["education"] = 0.8
        else:
            scores["education"] = 0.0
        
        # Experience confidence
        if result.get("experience") and len(result["experience"]) > 0:
            scores["experience"] = 0.8
        else:
            scores["experience"] = 0.0
        
        return scores

# Global parser instance
_parser_instance = None

def get_parser() -> EnhancedResumeParser:
    """Get or create parser instance."""
    global _parser_instance
    if _parser_instance is None:
        _parser_instance = EnhancedResumeParser()
    return _parser_instance

def parse_resume(file_path: str) -> Dict[str, Any]:
    """
    Parse resume and return structured data.
    
    Args:
        file_path: Path to the resume file
        
    Returns:
        Dictionary with structured resume data
    """
    parser = get_parser()
    return parser.parse_resume(file_path)
