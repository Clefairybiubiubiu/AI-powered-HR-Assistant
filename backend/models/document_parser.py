"""
ResumeParser class using spaCy for advanced text processing and keyword extraction
"""
import spacy
import re
from typing import Dict, List, Set, Tuple
from collections import Counter
import nltk
from nltk.corpus import stopwords

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')


class ResumeParser:
    """
    Advanced resume parser using spaCy for NLP processing and keyword extraction
    """
    
    def __init__(self, model_name: str = "en_core_web_sm"):
        """
        Initialize the ResumeParser with spaCy model
        
        Args:
            model_name: spaCy model name (default: "en_core_web_sm")
        """
        try:
            self.nlp = spacy.load(model_name)
        except OSError:
            print(f"Warning: spaCy model '{model_name}' not found.")
            print("Please install it with: python -m spacy download en_core_web_sm")
            self.nlp = None
        
        # Initialize stopwords
        self.stop_words = set(stopwords.words('english'))
        
        # Technical skills database
        self.technical_skills = {
            'programming': [
                'python', 'java', 'javascript', 'typescript', 'c++', 'c#', 'php', 'ruby', 'go', 'rust',
                'swift', 'kotlin', 'scala', 'r', 'matlab', 'sql', 'html', 'css', 'react', 'angular',
                'vue', 'node.js', 'django', 'flask', 'spring', 'express', 'laravel', 'rails', 'fastapi'
            ],
            'databases': [
                'mysql', 'postgresql', 'mongodb', 'redis', 'elasticsearch', 'cassandra', 'dynamodb',
                'oracle', 'sqlite', 'mariadb', 'neo4j', 'firebase'
            ],
            'cloud': [
                'aws', 'azure', 'gcp', 'docker', 'kubernetes', 'terraform', 'jenkins', 'gitlab',
                'github actions', 'ci/cd', 'microservices', 'serverless', 'lambda', 'ec2', 's3'
            ],
            'data_science': [
                'pandas', 'numpy', 'scikit-learn', 'tensorflow', 'pytorch', 'keras', 'jupyter',
                'matplotlib', 'seaborn', 'plotly', 'tableau', 'power bi', 'spark', 'hadoop'
            ],
            'tools': [
                'git', 'github', 'gitlab', 'jira', 'confluence', 'slack', 'teams', 'zoom',
                'figma', 'sketch', 'adobe', 'photoshop', 'illustrator', 'vscode', 'pycharm'
            ]
        }
        
        # Education keywords
        self.education_keywords = {
            'degrees': ['bachelor', 'master', 'phd', 'doctorate', 'associate', 'diploma', 'certificate'],
            'fields': ['computer science', 'engineering', 'mathematics', 'statistics', 'data science', 
                      'business', 'economics', 'physics', 'chemistry', 'biology'],
            'institutions': ['university', 'college', 'institute', 'school']
        }
        
        # Experience keywords
        self.experience_keywords = {
            'positions': ['engineer', 'developer', 'analyst', 'manager', 'director', 'lead', 'senior', 'junior'],
            'companies': ['inc', 'corp', 'llc', 'ltd', 'company', 'technologies', 'solutions'],
            'timeframes': ['years', 'months', 'experience', 'worked', 'employed']
        }
    
    def parse_resume(self, text: str) -> Dict[str, List[str]]:
        """
        Parse resume text and extract keywords for skills, education, and experience
        
        Args:
            text: Raw resume text
            
        Returns:
            Dictionary with keys: 'skills', 'education', 'experience'
        """
        if not self.nlp:
            return self._fallback_parsing(text)
        
        # Clean and preprocess text
        cleaned_text = self._clean_text(text)
        
        # Process with spaCy
        doc = self.nlp(cleaned_text)
        
        # Extract keywords
        skills = self._extract_skills(doc)
        education = self._extract_education(doc)
        experience = self._extract_experience(doc)
        
        return {
            'skills': skills,
            'education': education,
            'experience': experience
        }
    
    def _clean_text(self, text: str) -> str:
        """Clean and preprocess text"""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s.,!?;:()]', ' ', text)
        return text.strip()
    
    def _extract_skills(self, doc) -> List[str]:
        """Extract technical skills using spaCy processing"""
        skills = []
        
        # Extract skills from technical skills database
        text_lower = doc.text.lower()
        for category, skill_list in self.technical_skills.items():
            for skill in skill_list:
                if skill.lower() in text_lower:
                    skills.append(skill)
        
        # Extract skills from noun phrases and named entities
        for chunk in doc.noun_chunks:
            chunk_text = chunk.text.lower().strip()
            if (len(chunk_text.split()) <= 3 and  # Short phrases only
                not any(word in self.stop_words for word in chunk_text.split()) and
                chunk_text not in skills):
                
                # Check if it's a technical term
                if self._is_technical_term(chunk_text):
                    skills.append(chunk.text)
        
        # Extract skills from named entities
        for ent in doc.ents:
            if ent.label_ in ['ORG', 'PRODUCT', 'TECHNOLOGY']:
                ent_text = ent.text.lower().strip()
                if (len(ent_text.split()) <= 2 and
                    not any(word in self.stop_words for word in ent_text.split()) and
                    ent_text not in skills):
                    skills.append(ent.text)
        
        # Remove duplicates and return
        return list(set(skills))
    
    def _extract_education(self, doc) -> List[str]:
        """Extract education information"""
        education = []
        text_lower = doc.text.lower()
        
        # Extract degree information
        for degree in self.education_keywords['degrees']:
            if degree in text_lower:
                # Find the full degree phrase
                degree_pattern = rf'\b{re.escape(degree)}\s+(?:of|in)?\s*[a-z\s]+'
                matches = re.findall(degree_pattern, text_lower)
                for match in matches:
                    education.append(match.strip())
        
        # Extract field of study
        for field in self.education_keywords['fields']:
            if field in text_lower:
                education.append(field.title())
        
        # Extract institution names
        for token in doc:
            if (token.pos_ == 'PROPN' and  # Proper noun
                any(inst in token.text.lower() for inst in self.education_keywords['institutions'])):
                education.append(token.text)
        
        # Extract education-related noun phrases
        for chunk in doc.noun_chunks:
            chunk_text = chunk.text.lower()
            if any(edu_word in chunk_text for edu_word in ['degree', 'diploma', 'certificate', 'bachelor', 'master', 'phd']):
                education.append(chunk.text)
        
        return list(set(education))
    
    def _extract_experience(self, doc) -> List[str]:
        """Extract work experience information"""
        experience = []
        
        # Extract job titles and positions
        for chunk in doc.noun_chunks:
            chunk_text = chunk.text.lower()
            if any(pos in chunk_text for pos in self.experience_keywords['positions']):
                experience.append(chunk.text)
        
        # Extract company names
        for ent in doc.ents:
            if ent.label_ == 'ORG':
                ent_text = ent.text.lower()
                if any(comp in ent_text for comp in self.experience_keywords['companies']):
                    experience.append(ent.text)
        
        # Extract experience-related phrases
        for token in doc:
            if (token.pos_ == 'NOUN' and 
                any(exp_word in token.text.lower() for exp_word in ['experience', 'role', 'position', 'job'])):
                # Get the full phrase containing this token
                for chunk in doc.noun_chunks:
                    if token in chunk:
                        experience.append(chunk.text)
                        break
        
        # Extract time-related experience
        time_patterns = [
            r'\d+\+?\s*years?\s*(?:of\s*)?experience',
            r'\d+\+?\s*years?\s*(?:in|with)',
            r'(?:senior|junior|lead|principal)\s+\w+',
            r'\w+\s+(?:engineer|developer|analyst|manager)'
        ]
        
        for pattern in time_patterns:
            matches = re.findall(pattern, doc.text, re.IGNORECASE)
            experience.extend(matches)
        
        return list(set(experience))
    
    def _is_technical_term(self, text: str) -> bool:
        """Check if a term is likely to be technical"""
        technical_indicators = [
            'api', 'framework', 'library', 'database', 'server', 'cloud', 'web',
            'mobile', 'desktop', 'frontend', 'backend', 'fullstack', 'devops',
            'machine learning', 'ai', 'data', 'analytics', 'visualization'
        ]
        
        return any(indicator in text for indicator in technical_indicators)
    
    def _fallback_parsing(self, text: str) -> Dict[str, List[str]]:
        """Fallback parsing when spaCy is not available"""
        text_lower = text.lower()
        
        # Simple keyword extraction
        skills = []
        for category, skill_list in self.technical_skills.items():
            for skill in skill_list:
                if skill.lower() in text_lower:
                    skills.append(skill)
        
        education = []
        for degree in self.education_keywords['degrees']:
            if degree in text_lower:
                education.append(degree.title())
        
        experience = []
        for position in self.experience_keywords['positions']:
            if position in text_lower:
                experience.append(position.title())
        
        return {
            'skills': list(set(skills)),
            'education': list(set(education)),
            'experience': list(set(experience))
        }
    
    def get_keyword_frequency(self, text: str) -> Dict[str, int]:
        """
        Get frequency of keywords in the text
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with keyword frequencies
        """
        if not self.nlp:
            return {}
        
        doc = self.nlp(text)
        keyword_freq = {}
        
        # Count technical skills
        text_lower = text.lower()
        for category, skills in self.technical_skills.items():
            for skill in skills:
                count = text_lower.count(skill.lower())
                if count > 0:
                    keyword_freq[skill] = count
        
        return keyword_freq
    
    def extract_named_entities(self, text: str) -> List[Tuple[str, str]]:
        """
        Extract named entities from text
        
        Args:
            text: Input text
            
        Returns:
            List of (entity, label) tuples
        """
        if not self.nlp:
            return []
        
        doc = self.nlp(text)
        entities = []
        
        for ent in doc.ents:
            entities.append((ent.text, ent.label_))
        
        return entities
    
    def get_text_statistics(self, text: str) -> Dict[str, any]:
        """
        Get comprehensive text statistics
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with text statistics
        """
        if not self.nlp:
            return {}
        
        doc = self.nlp(text)
        
        # Basic statistics
        stats = {
            'word_count': len([token for token in doc if not token.is_space]),
            'sentence_count': len(list(doc.sents)),
            'character_count': len(text),
            'average_word_length': sum(len(token.text) for token in doc if not token.is_space) / max(1, len([token for token in doc if not token.is_space])),
            'unique_words': len(set(token.text.lower() for token in doc if not token.is_space and not token.is_punct)),
            'pos_tags': Counter([token.pos_ for token in doc]),
            'named_entities': len(list(doc.ents))
        }
        
        return stats
