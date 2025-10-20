"""
Job API endpoints
"""
from fastapi import APIRouter, HTTPException
from typing import List, Dict, Any
from pydantic import BaseModel

from backend.models.job import JobDescription, JobAnalysis
from backend.services.skill_extractor import SkillExtractor
from backend.utils.jd_formatter import normalize_jd_input

router = APIRouter()

# Initialize services
skill_extractor = SkillExtractor()


class JobDescriptionRequest(BaseModel):
    title: str = None
    description: str = None
    requirements: str = None
    location: str = None
    salary_range: str = None


class FlexibleJobRequest(BaseModel):
    """Flexible job request that accepts any fields"""
    data: Dict[str, Any]


@router.post("/analyze", response_model=JobAnalysis)
async def analyze_job(job_request: JobDescriptionRequest):
    """
    Analyze job description and extract requirements
    """
    try:
        # Extract skills from job description
        skills = skill_extractor.extract_skills(job_request.description)
        
        # Create job analysis
        job_analysis = JobAnalysis(
            job_description=JobDescription(
                title=job_request.title,
                description=job_request.description,
                requirements=job_request.requirements,
                location=job_request.location,
                salary_range=job_request.salary_range
            ),
            required_skills=skills,
            skill_categories=skill_extractor.categorize_skills(skills)
        )
        
        return job_analysis
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error analyzing job: {str(e)}")


@router.post("/analyze-flexible", response_model=JobAnalysis)
async def analyze_job_flexible(job_request: FlexibleJobRequest):
    """
    Analyze job description with flexible input format
    Automatically normalizes inconsistent or partial data
    """
    try:
        # Normalize the input data
        normalized_data = normalize_jd_input(job_request.data)
        
        # Extract skills from normalized description
        skills = skill_extractor.extract_skills(normalized_data['description'])
        
        # Create job analysis with normalized data
        job_analysis = JobAnalysis(
            job_description=JobDescription(
                title=normalized_data['title'],
                description=normalized_data['description'],
                requirements=normalized_data['requirements'],
                location=normalized_data['location'],
                salary_range=normalized_data['salary_range']
            ),
            required_skills=skills,
            skill_categories=skill_extractor.categorize_skills(skills)
        )
        
        return job_analysis
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error analyzing job: {str(e)}")


@router.get("/", response_model=List[JobDescription])
async def get_jobs():
    """
    Get all job descriptions
    """
    # TODO: Implement database storage and retrieval
    return []


@router.get("/{job_id}", response_model=JobDescription)
async def get_job(job_id: str):
    """
    Get specific job by ID
    """
    # TODO: Implement database retrieval
    raise HTTPException(status_code=404, detail="Job not found")
