"""
Skill extraction service using NLP
"""
from typing import Any, List, Dict
import spacy
import re
from typing import List, Dict, Set
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


class SkillExtractor:
    """Extract skills and qualifications from text using NLP"""
    
    def __init__(self):
        # Initialize spaCy model
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            print("Warning: spaCy model 'en_core_web_sm' not found. Please install it with: python -m spacy download en_core_web_sm")
            self.nlp = None
        
        # Common technical skills database
        self.technical_skills = {
            'programming': [
                'python', 'java', 'javascript', 'typescript', 'c++', 'c#', 'php', 'ruby', 'go', 'rust',
                'swift', 'kotlin', 'scala', 'r', 'matlab', 'sql', 'html', 'css', 'react', 'angular',
                'vue', 'node.js', 'django', 'flask', 'spring', 'express', 'laravel', 'rails'
            ],
            'databases': [
                'mysql', 'postgresql', 'mongodb', 'redis', 'elasticsearch', 'cassandra', 'dynamodb',
                'oracle', 'sqlite', 'mariadb', 'neo4j'
            ],
            'cloud': [
                'aws', 'azure', 'gcp', 'docker', 'kubernetes', 'terraform', 'jenkins', 'gitlab',
                'github actions', 'ci/cd', 'microservices', 'serverless'
            ],
            'data_science': [
                'pandas', 'numpy', 'scikit-learn', 'tensorflow', 'pytorch', 'keras', 'jupyter',
                'matplotlib', 'seaborn', 'plotly', 'tableau', 'power bi', 'spark', 'hadoop'
            ],
            'tools': [
                'git', 'github', 'gitlab', 'jira', 'confluence', 'slack', 'teams', 'zoom',
                'figma', 'sketch', 'adobe', 'photoshop', 'illustrator'
            ]
        }
        
        # Soft skills
        self.soft_skills = [
            'leadership', 'communication', 'teamwork', 'problem solving', 'critical thinking',
            'time management', 'adaptability', 'creativity', 'emotional intelligence',
            'negotiation', 'presentation', 'mentoring', 'collaboration', 'project management'
        ]
        
        # Stop words
        self.stop_words = set(stopwords.words('english'))
    
    def extract_skills(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract skills from text with confidence scores
        """
        if not text:
            return []
        
        skills = []
        text_lower = text.lower()
        
        # Extract technical skills
        for category, skill_list in self.technical_skills.items():
            for skill in skill_list:
                if skill.lower() in text_lower:
                    confidence = self._calculate_confidence(text, skill)
                    skills.append({
                        'name': skill,
                        'confidence': confidence,
                        'category': category
                    })
        
        # Extract soft skills
        for skill in self.soft_skills:
            if skill.lower() in text_lower:
                confidence = self._calculate_confidence(text, skill)
                skills.append({
                    'name': skill,
                    'confidence': confidence,
                    'category': 'soft_skills'
                })
        
        # Extract additional skills using NLP
        if self.nlp:
            nlp_skills = self._extract_skills_with_nlp(text)
            skills.extend(nlp_skills)
        
        # Remove duplicates and sort by confidence
        unique_skills = self._deduplicate_skills(skills)
        return sorted(unique_skills, key=lambda x: x['confidence'], reverse=True)
    
    def _calculate_confidence(self, text: str, skill: str) -> float:
        """
        Calculate confidence score for a skill based on context
        """
        text_lower = text.lower()
        skill_lower = skill.lower()
        
        # Base confidence
        confidence = 0.5
        
        # Increase confidence based on context
        context_patterns = [
            r'experienced\s+in\s+' + re.escape(skill_lower),
            r'proficient\s+in\s+' + re.escape(skill_lower),
            r'skilled\s+in\s+' + re.escape(skill_lower),
            r'expertise\s+in\s+' + re.escape(skill_lower),
            r'years?\s+of\s+' + re.escape(skill_lower),
            r'strong\s+' + re.escape(skill_lower),
            r'advanced\s+' + re.escape(skill_lower)
        ]
        
        for pattern in context_patterns:
            if re.search(pattern, text_lower):
                confidence += 0.2
        
        # Check for multiple mentions
        mention_count = text_lower.count(skill_lower)
        if mention_count > 1:
            confidence += 0.1 * min(mention_count - 1, 3)
        
        return min(confidence, 1.0)
    
    def _extract_skills_with_nlp(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract skills using spaCy NLP
        """
        if not self.nlp:
            return []
        
        doc = self.nlp(text)
        skills = []
        
        # Extract noun phrases that might be skills
        for chunk in doc.noun_chunks:
            if len(chunk.text.split()) <= 3:  # Short phrases only
                text_lower = chunk.text.lower()
                if not any(word in self.stop_words for word in text_lower.split()):
                    # Check if it's not already in our known skills
                    if not any(skill['name'].lower() == text_lower for skill in skills):
                        skills.append({
                            'name': chunk.text,
                            'confidence': 0.3,
                            'category': 'extracted'
                        })
        
        return skills
    
    def _deduplicate_skills(self, skills: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Remove duplicate skills, keeping the one with highest confidence
        """
        skill_dict = {}
        
        for skill in skills:
            name_lower = skill['name'].lower()
            if name_lower not in skill_dict or skill['confidence'] > skill_dict[name_lower]['confidence']:
                skill_dict[name_lower] = skill
        
        return list(skill_dict.values())
    
    def categorize_skills(self, skills: List[Dict[str, Any]]) -> Dict[str, List[str]]:
        """
        Categorize skills by type
        """
        categories = {}
        
        for skill in skills:
            category = skill.get('category', 'other')
            if category not in categories:
                categories[category] = []
            categories[category].append(skill['name'])
        
        return categories
