"""
Similarity Scorer that integrates ResumeParser and SkillExtractor for comprehensive candidate-job matching
"""
import spacy
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from typing import Dict, List, Tuple, Any, Optional
import re
from collections import Counter

# Import our custom classes
from .document_parser import ResumeParser
from .skill_extractor import SkillExtractor


class SimilarityScorer:
    """
    Advanced similarity scorer that combines ResumeParser, SkillExtractor, and SentenceTransformer
    for comprehensive candidate-job matching
    """
    
    def __init__(self, 
                 sentence_transformer_model: str = "all-MiniLM-L6-v2",
                 spacy_model: str = "en_core_web_sm"):
        """
        Initialize the similarity scorer
        
        Args:
            sentence_transformer_model: SentenceTransformer model name
            spacy_model: spaCy model name
        """
        # Initialize components
        self.resume_parser = ResumeParser(spacy_model)
        self.skill_extractor = SkillExtractor()
        self.sentence_transformer = SentenceTransformer(sentence_transformer_model)
        
        # Scoring weights
        self.alpha = 0.4  # skill_match weight
        self.beta = 0.4   # experience_alignment weight
        self.gamma = 0.2  # education_match weight
        
        # Experience level mapping
        self.experience_levels = {
            'entry': 1,
            'junior': 1,
            'mid': 2,
            'intermediate': 2,
            'senior': 3,
            'lead': 3,
            'principal': 3,
            'architect': 3
        }
        
        # Education level mapping
        self.education_levels = {
            'high_school': 1,
            'diploma': 1,
            'certificate': 1,
            'associate': 2,
            'bachelor': 3,
            'master': 4,
            'phd': 5,
            'doctorate': 5
        }
    
    def compute_fit_score(self, 
                          resume_text: str, 
                          job_description: str,
                          job_requirements: Optional[str] = None) -> Dict[str, Any]:
        """
        Compute comprehensive fit score between resume and job description
        
        Args:
            resume_text: Raw resume text
            job_description: Job description text
            job_requirements: Optional separate requirements text
            
        Returns:
            Dictionary containing detailed scoring results
        """
        # Parse resume using ResumeParser
        resume_data = self.resume_parser.parse_resume(resume_text)
        
        # Extract skills from resume using SkillExtractor
        resume_skills = self.skill_extractor.extract_skills(resume_text)
        resume_skill_names = [skill['name'] for skill in resume_skills]
        
        # Extract skills from job description
        job_skills = self.skill_extractor.extract_skills(job_description)
        job_skill_names = [skill['name'] for skill in job_skills]
        
        # If separate requirements provided, extract from there too
        if job_requirements:
            req_skills = self.skill_extractor.extract_skills(job_requirements)
            job_skill_names.extend([skill['name'] for skill in req_skills])
            job_skill_names = list(set(job_skill_names))  # Remove duplicates
        
        # Calculate skill match score
        skill_match = self._calculate_skill_match(resume_skill_names, job_skill_names)
        
        # Calculate experience alignment
        experience_alignment = self._calculate_experience_alignment(
            resume_data, resume_text, job_description
        )
        
        # Calculate education match
        education_match = self._calculate_education_match(
            resume_data, resume_text, job_description
        )
        
        # Calculate semantic similarity for additional context
        semantic_similarity = self._calculate_semantic_similarity(
            resume_text, job_description
        )
        
        # Calculate overall fit score using weighted formula
        overall_score = (self.alpha * skill_match + 
                        self.beta * experience_alignment + 
                        self.gamma * education_match)
        
        # Prepare detailed results
        result = {
            'overall_score': float(overall_score),
            'skill_match': float(skill_match),
            'experience_alignment': float(experience_alignment),
            'education_match': float(education_match),
            'semantic_similarity': float(semantic_similarity),
            'weights': {
                'alpha': self.alpha,
                'beta': self.beta,
                'gamma': self.gamma
            },
            'resume_analysis': {
                'skills_found': resume_skill_names,
                'education_found': resume_data['education'],
                'experience_found': resume_data['experience'],
                'skills_with_confidence': resume_skills
            },
            'job_analysis': {
                'skills_required': job_skill_names,
                'skills_categories': self.skill_extractor.categorize_skills(job_skills)
            },
            'matching_details': {
                'matched_skills': list(set(resume_skill_names) & set(job_skill_names)),
                'missing_skills': list(set(job_skill_names) - set(resume_skill_names)),
                'extra_skills': list(set(resume_skill_names) - set(job_skill_names))
            }
        }
        
        return result
    
    def _calculate_skill_match(self, resume_skills: List[str], job_skills: List[str]) -> float:
        """
        Calculate skill matching score using both exact and semantic matching
        
        Args:
            resume_skills: List of skills from resume
            job_skills: List of skills from job description
            
        Returns:
            Skill match score (0.0 - 1.0)
        """
        if not job_skills:
            return 1.0  # No skills required
        
        if not resume_skills:
            return 0.0  # No skills found in resume
        
        # Calculate exact matches
        exact_matches = set(resume_skills) & set(job_skills)
        exact_match_ratio = len(exact_matches) / len(job_skills)
        
        # Calculate semantic similarity for non-exact matches
        if len(resume_skills) > 0 and len(job_skills) > 0:
            # Get embeddings for all skills
            all_skills = resume_skills + job_skills
            embeddings = self.sentence_transformer.encode(all_skills)
            
            resume_embeddings = embeddings[:len(resume_skills)]
            job_embeddings = embeddings[len(resume_skills):]
            
            # Calculate similarity matrix
            similarity_matrix = cosine_similarity(resume_embeddings, job_embeddings)
            
            # Find semantic matches (similarity > 0.7)
            semantic_matches = 0
            for i in range(len(job_skills)):
                if job_skills[i] not in exact_matches:
                    max_similarity = np.max(similarity_matrix[:, i])
                    if max_similarity > 0.7:
                        semantic_matches += 1
            
            semantic_match_ratio = semantic_matches / len(job_skills)
        else:
            semantic_match_ratio = 0
        
        # Combine exact and semantic matches
        total_match = min(exact_match_ratio + semantic_match_ratio * 0.5, 1.0)
        return total_match
    
    def _calculate_experience_alignment(self, 
                                      resume_data: Dict[str, List[str]], 
                                      resume_text: str, 
                                      job_description: str) -> float:
        """
        Calculate experience level alignment
        
        Args:
            resume_data: Parsed resume data
            resume_text: Raw resume text
            job_description: Job description text
            
        Returns:
            Experience alignment score (0.0 - 1.0)
        """
        # Extract experience level from resume
        resume_exp_level = self._extract_experience_level(resume_text)
        
        # Extract experience level from job description
        job_exp_level = self._extract_experience_level(job_description)
        
        if not resume_exp_level or not job_exp_level:
            return 0.5  # Neutral score if can't determine
        
        # Get numeric levels
        resume_level = self.experience_levels.get(resume_exp_level, 2)
        job_level = self.experience_levels.get(job_exp_level, 2)
        
        # Candidate can be higher level than required (no penalty)
        if resume_level >= job_level:
            return 1.0
        else:
            # Penalty for being underqualified
            penalty = (job_level - resume_level) * 0.2
            return max(0.0, 1.0 - penalty)
    
    def _calculate_education_match(self, 
                                 resume_data: Dict[str, List[str]], 
                                 resume_text: str, 
                                 job_description: str) -> float:
        """
        Calculate education requirement match
        
        Args:
            resume_data: Parsed resume data
            resume_text: Raw resume text
            job_description: Job description text
            
        Returns:
            Education match score (0.0 - 1.0)
        """
        # Extract education from resume
        resume_education = self._extract_education_levels(resume_text)
        
        # Extract education requirements from job
        job_education = self._extract_education_levels(job_description)
        
        if not job_education:
            return 1.0  # No education requirements
        
        if not resume_education:
            return 0.0  # No education found
        
        # Get highest education levels
        resume_level = max([self.education_levels.get(edu, 0) for edu in resume_education])
        job_level = max([self.education_levels.get(edu, 0) for edu in job_education])
        
        if resume_level >= job_level:
            return 1.0
        else:
            return max(0.0, resume_level / job_level)
    
    def _calculate_semantic_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate semantic similarity using SentenceTransformer
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Semantic similarity score (0.0 - 1.0)
        """
        if not text1 or not text2:
            return 0.0
        
        # Clean texts
        text1_clean = re.sub(r'\s+', ' ', text1).strip()
        text2_clean = re.sub(r'\s+', ' ', text2).strip()
        
        # Generate embeddings
        embeddings = self.sentence_transformer.encode([text1_clean, text2_clean])
        
        # Calculate cosine similarity
        similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
        return float(similarity)
    
    def _extract_experience_level(self, text: str) -> str:
        """
        Extract experience level from text
        
        Args:
            text: Input text
            
        Returns:
            Experience level string
        """
        text_lower = text.lower()
        
        # Check for explicit experience level indicators
        for level, keywords in {
            'senior': ['senior', 'lead', 'principal', 'architect', '5+ years', '6+ years', '7+ years'],
            'mid': ['mid', 'intermediate', '3-5 years', '4-6 years', '2-4 years'],
            'entry': ['entry', 'junior', '0-2 years', '1-3 years', 'fresh graduate', 'new graduate']
        }.items():
            if any(keyword in text_lower for keyword in keywords):
                return level
        
        # Check for years of experience patterns
        years_pattern = r'(\d+)\+?\s*years?\s*(?:of\s*)?experience'
        years_match = re.search(years_pattern, text_lower)
        if years_match:
            years = int(years_match.group(1))
            if years >= 5:
                return 'senior'
            elif years >= 3:
                return 'mid'
            else:
                return 'entry'
        
        return 'mid'  # Default assumption
    
    def _extract_education_levels(self, text: str) -> List[str]:
        """
        Extract education levels from text
        
        Args:
            text: Input text
            
        Returns:
            List of education levels found
        """
        text_lower = text.lower()
        education_levels = []
        
        # Check for degree types
        for level, keywords in {
            'phd': ['phd', 'doctorate', 'ph.d', 'doctoral'],
            'master': ['master', 'ms', 'ma', 'msc', 'mba', 'graduate'],
            'bachelor': ['bachelor', 'bs', 'ba', 'bsc', 'undergraduate', 'college'],
            'associate': ['associate', 'aa', 'as'],
            'high_school': ['high school', 'diploma', 'certificate']
        }.items():
            if any(keyword in text_lower for keyword in keywords):
                education_levels.append(level)
        
        return education_levels
    
    def get_detailed_analysis(self, 
                            resume_text: str, 
                            job_description: str,
                            job_requirements: Optional[str] = None) -> Dict[str, Any]:
        """
        Get detailed analysis including all intermediate results
        
        Args:
            resume_text: Raw resume text
            job_description: Job description text
            job_requirements: Optional separate requirements text
            
        Returns:
            Detailed analysis dictionary
        """
        # Get basic fit score
        fit_score = self.compute_fit_score(resume_text, job_description, job_requirements)
        
        # Get additional analysis
        resume_data = self.resume_parser.parse_resume(resume_text)
        resume_stats = self.resume_parser.get_text_statistics(resume_text)
        resume_entities = self.resume_parser.extract_named_entities(resume_text)
        
        # Enhanced analysis
        detailed_analysis = {
            **fit_score,
            'resume_statistics': resume_stats,
            'resume_entities': resume_entities,
            'resume_keyword_frequency': self.resume_parser.get_keyword_frequency(resume_text),
            'analysis_metadata': {
                'resume_word_count': resume_stats.get('word_count', 0),
                'resume_sentence_count': resume_stats.get('sentence_count', 0),
                'resume_unique_words': resume_stats.get('unique_words', 0),
                'resume_named_entities': resume_stats.get('named_entities', 0)
            }
        }
        
        return detailed_analysis
    
    def batch_score_candidates(self, 
                              candidates: List[Dict[str, str]], 
                              job_description: str,
                              job_requirements: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Score multiple candidates against a job description
        
        Args:
            candidates: List of candidate dictionaries with 'name' and 'resume' keys
            job_description: Job description text
            job_requirements: Optional separate requirements text
            
        Returns:
            List of scoring results for each candidate
        """
        results = []
        
        for candidate in candidates:
            try:
                score_result = self.compute_fit_score(
                    candidate['resume'], 
                    job_description, 
                    job_requirements
                )
                
                # Add candidate information
                score_result['candidate_name'] = candidate.get('name', 'Unknown')
                results.append(score_result)
                
            except Exception as e:
                # Handle individual candidate errors
                error_result = {
                    'candidate_name': candidate.get('name', 'Unknown'),
                    'overall_score': 0.0,
                    'error': str(e)
                }
                results.append(error_result)
        
        # Sort by overall score (descending)
        results.sort(key=lambda x: x.get('overall_score', 0), reverse=True)
        
        return results
