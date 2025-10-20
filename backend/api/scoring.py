"""
Scoring API endpoints
"""
from fastapi import APIRouter, HTTPException
from typing import List, Dict, Any
from pydantic import BaseModel
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import re
from collections import Counter

from backend.models.candidate import Candidate, CandidateProfile
from backend.models.job import JobDescription
from backend.services.similarity_scorer import SimilarityScorer
from backend.utils.jd_formatter import normalize_jd_input
from backend.utils.candidate_formatter import normalize_candidate_input

router = APIRouter()

# Initialize services
similarity_scorer = SimilarityScorer()


class ScoringRequest(BaseModel):
    candidate_id: str
    job_id: str


class JobInput(BaseModel):
    title: str
    description: str
    requirements: str = ""


class CandidateInput(BaseModel):
    name: str
    resume_text: str


class BatchScoringRequest(BaseModel):
    jobs: List[JobInput]
    candidates: List[CandidateInput]


class MatchResult(BaseModel):
    job_title: str
    best_candidate: str
    score: float
    reason: str


class BatchScoringResponse(BaseModel):
    results: List[MatchResult]


@router.post("/calculate-fit", response_model=CandidateProfile)
async def calculate_fit_score(scoring_request: ScoringRequest):
    """
    Calculate fit score between candidate and job
    """
    try:
        # TODO: Retrieve candidate and job from database
        # For now, return a mock response
        raise HTTPException(status_code=501, detail="Not implemented yet")
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error calculating fit score: {str(e)}")


@router.post("/batch", response_model=BatchScoringResponse)
async def batch_scoring(request: BatchScoringRequest):
    """
    Batch scoring: Match all candidates to all jobs and return best matches
    """
    try:
        results = []
        
        # Process each job
        for job in request.jobs:
            # Normalize job data
            job_data = normalize_jd_input({
                "title": job.title,
                "description": job.description,
                "requirements": job.requirements
            })
            
            # Combine job text for embedding
            job_text = f"{job_data['title']} {job_data['description']} {job_data['requirements']}"
            
            best_candidate = None
            best_score = 0.0
            best_reason = ""
            
            # Compare with each candidate
            for candidate in request.candidates:
                # Normalize candidate data
                candidate_data = normalize_candidate_input(candidate.resume_text)
                
                # Calculate similarity score
                score_result = similarity_scorer.compute_fit_score(
                    candidate.resume_text, 
                    job_text
                )
                
                similarity_score = score_result['overall_score']
                
                # Generate reason
                reason = _generate_match_reason(
                    candidate.name,
                    candidate.resume_text,
                    job_text,
                    similarity_score
                )
                
                # Track best match for this job
                if similarity_score > best_score:
                    best_score = similarity_score
                    best_candidate = candidate.name
                    best_reason = reason
            
            # Add result for this job
            if best_candidate:
                results.append(MatchResult(
                    job_title=job.title,
                    best_candidate=best_candidate,
                    score=best_score,
                    reason=best_reason
                ))
        
        return BatchScoringResponse(results=results)
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error in batch scoring: {str(e)}")


@router.post("/batch-scoring", response_model=List[CandidateProfile])
async def batch_scoring_legacy(candidate_ids: List[str], job_id: str):
    """
    Calculate fit scores for multiple candidates against one job (legacy endpoint)
    """
    try:
        # TODO: Implement batch scoring
        raise HTTPException(status_code=501, detail="Not implemented yet")
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error in batch scoring: {str(e)}")


@router.get("/scores/{job_id}", response_model=List[CandidateProfile])
async def get_scores_for_job(job_id: str):
    """
    Get all candidate scores for a specific job
    """
    try:
        # TODO: Implement score retrieval
        raise HTTPException(status_code=501, detail="Not implemented yet")
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error retrieving scores: {str(e)}")


def _generate_match_reason(candidate_name: str, resume_text: str, job_text: str, score: float) -> str:
    """
    Generate a reason for why a candidate matches a job
    """
    try:
        # Extract top keywords from both texts
        resume_keywords = _extract_keywords(resume_text)
        job_keywords = _extract_keywords(job_text)
        
        # Find overlapping keywords
        overlapping = set(resume_keywords) & set(job_keywords)
        
        # Get top 3 overlapping keywords
        top_keywords = list(overlapping)[:3]
        
        if top_keywords:
            keywords_str = ", ".join(top_keywords)
            return f"{candidate_name}'s resume matches key skills found in the job requirements such as {keywords_str}."
        else:
            # Fallback: use score-based reason
            if score > 0.8:
                return f"{candidate_name} shows strong overall alignment with the job requirements (score: {score:.2f})."
            elif score > 0.6:
                return f"{candidate_name} demonstrates good compatibility with the job requirements (score: {score:.2f})."
            else:
                return f"{candidate_name} has some relevant experience for this role (score: {score:.2f})."
                
    except Exception:
        # Fallback reason if keyword extraction fails
        return f"{candidate_name} shows compatibility with the job requirements (score: {score:.2f})."


def _extract_keywords(text: str, top_n: int = 10) -> List[str]:
    """
    Extract top keywords from text using simple frequency analysis
    """
    try:
        # Clean and tokenize text
        text = re.sub(r'[^\w\s]', ' ', text.lower())
        words = text.split()
        
        # Remove common stop words
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
            'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did',
            'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can', 'this', 'that', 'these',
            'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them'
        }
        
        # Filter out stop words and short words
        filtered_words = [word for word in words if word not in stop_words and len(word) > 2]
        
        # Count word frequencies
        word_counts = Counter(filtered_words)
        
        # Return top keywords
        return [word for word, count in word_counts.most_common(top_n)]
        
    except Exception:
        return []
