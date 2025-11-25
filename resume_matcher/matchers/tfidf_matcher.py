"""
TF-IDF based resume-JD matcher.
Uses traditional keyword-based matching with cosine similarity.
"""
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from ..config import config
from ..logging_config import get_logger
from .base_matcher import BaseMatcher

logger = get_logger(__name__)


class ResumeJDMatcher(BaseMatcher):
    """TF-IDF based matcher for matching resumes with job descriptions."""
    
    def compute_similarity(self) -> np.ndarray:
        """
        Compute TF-IDF and cosine similarity between resumes and job descriptions.
        
        Returns:
            Similarity matrix (resumes x job descriptions)
        """
        if not self.resumes or not self.job_descriptions:
            logger.error("No documents loaded!")
            return np.array([])
        
        logger.info("Computing TF-IDF similarity...")
        
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
            max_features=config.tfidf_max_features,
            ngram_range=config.tfidf_ngram_range,
            min_df=config.tfidf_min_df,
            max_df=config.tfidf_max_df
        )
        
        try:
            tfidf_matrix = vectorizer.fit_transform(all_docs)
            
            # Compute cosine similarity
            similarity_matrix = cosine_similarity(tfidf_matrix)
            
            # Extract resume-JD similarities
            num_resumes = len(self.resumes)
            num_jds = len(self.job_descriptions)
            
            self.similarity_matrix = similarity_matrix[:num_resumes, num_resumes:]
            
            logger.info(
                f"TF-IDF similarity computation completed. "
                f"Matrix shape: {self.similarity_matrix.shape}"
            )
            
            return self.similarity_matrix
            
        except Exception as e:
            logger.error(f"Failed to compute TF-IDF similarity: {e}", exc_info=True)
            raise

