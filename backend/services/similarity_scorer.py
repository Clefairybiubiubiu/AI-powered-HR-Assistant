"""
Similarity scoring service using Sentence-BERT
"""
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Dict, Tuple
from sklearn.metrics.pairwise import cosine_similarity
import re


class SimilarityScorer:
    """Calculate semantic similarity between candidates and jobs"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the similarity scorer with a Sentence-BERT model
        """
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name
    
    def calculate_fit_score(self, candidate_text: str, job_description: str, 
                          candidate_skills: List[str] = None, 
                          job_requirements: List[str] = None) -> Dict[str, float]:
        """
        Calculate comprehensive fit score between candidate and job
        """
        scores = {}
        
        # Overall text similarity
        scores['text_similarity'] = self._calculate_text_similarity(candidate_text, job_description)
        
        # Skills matching
        if candidate_skills and job_requirements:
            scores['skills_match'] = self._calculate_skills_similarity(candidate_skills, job_requirements)
        else:
            scores['skills_match'] = 0.0
        
        # Experience level matching
        scores['experience_match'] = self._calculate_experience_similarity(candidate_text, job_description)
        
        # Education matching
        scores['education_match'] = self._calculate_education_similarity(candidate_text, job_description)
        
        # Calculate overall weighted score
        weights = {
            'text_similarity': 0.3,
            'skills_match': 0.4,
            'experience_match': 0.2,
            'education_match': 0.1
        }
        
        overall_score = sum(scores[key] * weights[key] for key in scores.keys())
        scores['overall'] = overall_score
        
        return scores
    
    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate semantic similarity between two texts
        """
        if not text1 or not text2:
            return 0.0
        
        # Clean and prepare texts
        text1_clean = self._clean_text(text1)
        text2_clean = self._clean_text(text2)
        
        # Generate embeddings
        embeddings = self.model.encode([text1_clean, text2_clean])
        
        # Calculate cosine similarity
        similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
        
        return float(similarity)
    
    def _calculate_skills_similarity(self, candidate_skills: List[str], job_skills: List[str]) -> float:
        """
        Calculate similarity between candidate and job skills
        """
        if not candidate_skills or not job_skills:
            return 0.0
        
        # Convert to lowercase for comparison
        candidate_skills_lower = [skill.lower() for skill in candidate_skills]
        job_skills_lower = [skill.lower() for skill in job_skills]
        
        # Calculate exact matches
        exact_matches = set(candidate_skills_lower) & set(job_skills_lower)
        exact_match_ratio = len(exact_matches) / len(job_skills_lower) if job_skills_lower else 0
        
        # Calculate semantic similarity for non-exact matches
        if len(candidate_skills_lower) > 0 and len(job_skills_lower) > 0:
            # Generate embeddings for all skills
            all_skills = candidate_skills_lower + job_skills_lower
            embeddings = self.model.encode(all_skills)
            
            candidate_embeddings = embeddings[:len(candidate_skills_lower)]
            job_embeddings = embeddings[len(candidate_skills_lower):]
            
            # Calculate similarity matrix
            similarity_matrix = cosine_similarity(candidate_embeddings, job_embeddings)
            
            # Find best matches for each job skill
            semantic_matches = 0
            for i in range(len(job_skills_lower)):
                if job_skills_lower[i] not in exact_matches:
                    max_similarity = np.max(similarity_matrix[:, i])
                    if max_similarity > 0.7:  # Threshold for semantic similarity
                        semantic_matches += 1
            
            semantic_match_ratio = semantic_matches / len(job_skills_lower) if job_skills_lower else 0
        else:
            semantic_match_ratio = 0
        
        # Combine exact and semantic matches
        total_match_ratio = min(exact_match_ratio + semantic_match_ratio * 0.5, 1.0)
        
        return float(total_match_ratio)
    
    def _calculate_experience_similarity(self, candidate_text: str, job_text: str) -> float:
        """
        Calculate experience level similarity
        """
        # Extract experience levels
        candidate_exp = self._extract_experience_level(candidate_text)
        job_exp = self._extract_experience_level(job_text)
        
        if not candidate_exp or not job_exp:
            return 0.5  # Neutral score if can't determine
        
        # Map experience levels to numeric values
        exp_levels = {'entry': 1, 'junior': 2, 'mid': 3, 'senior': 4, 'lead': 5, 'principal': 6}
        
        candidate_level = exp_levels.get(candidate_exp, 3)
        job_level = exp_levels.get(job_exp, 3)
        
        # Calculate similarity (candidate can be higher level than required)
        if candidate_level >= job_level:
            return 1.0
        else:
            # Penalty for being underqualified
            return max(0.0, 1.0 - (job_level - candidate_level) * 0.2)
    
    def _calculate_education_similarity(self, candidate_text: str, job_text: str) -> float:
        """
        Calculate education requirement similarity
        """
        # Extract education requirements
        candidate_education = self._extract_education(candidate_text)
        job_education = self._extract_education(job_text)
        
        if not job_education:
            return 1.0  # No education requirements
        
        if not candidate_education:
            return 0.0  # No education found for candidate
        
        # Simple matching based on degree level
        education_levels = {
            'high school': 1,
            'associate': 2,
            'bachelor': 3,
            'master': 4,
            'phd': 5,
            'doctorate': 5
        }
        
        candidate_level = max([education_levels.get(edu, 0) for edu in candidate_education])
        job_level = max([education_levels.get(edu, 0) for edu in job_education])
        
        if candidate_level >= job_level:
            return 1.0
        else:
            return max(0.0, candidate_level / job_level)
    
    def _clean_text(self, text: str) -> str:
        """
        Clean text for better embedding quality
        """
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s.,!?;:]', ' ', text)
        return text.strip()
    
    def _extract_experience_level(self, text: str) -> str:
        """
        Extract experience level from text
        """
        text_lower = text.lower()
        
        if any(word in text_lower for word in ['senior', 'lead', 'principal', 'architect']):
            return 'senior'
        elif any(word in text_lower for word in ['mid', 'intermediate', '3-5 years', '4-6 years']):
            return 'mid'
        elif any(word in text_lower for word in ['junior', 'entry', '0-2 years', '1-3 years']):
            return 'junior'
        else:
            return 'mid'  # Default assumption
    
    def _extract_education(self, text: str) -> List[str]:
        """
        Extract education information from text
        """
        text_lower = text.lower()
        education = []
        
        education_keywords = [
            'bachelor', 'master', 'phd', 'doctorate', 'associate', 'high school',
            'bs', 'ms', 'mba', 'phd', 'ba', 'ma', 'bsc', 'msc'
        ]
        
        for keyword in education_keywords:
            if keyword in text_lower:
                education.append(keyword)
        
        return education
