from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from datetime import datetime


class JobDescription(BaseModel):
    title: str
    description: str
    requirements: str
    location: Optional[str] = None
    salary_range: Optional[str] = None
    company: Optional[str] = None
    employment_type: Optional[str] = None  # full-time, part-time, contract
    experience_level: Optional[str] = None  # entry, mid, senior
    created_at: datetime = datetime.now()


class JobAnalysis(BaseModel):
    job_description: JobDescription
    required_skills: List[str] = []
    skill_categories: Dict[str, List[str]] = {}
    experience_requirements: List[str] = []
    education_requirements: List[str] = []
    soft_skills: List[str] = []
    technical_skills: List[str] = []
    industry_keywords: List[str] = []
    complexity_score: float = 0.0
    created_at: datetime = datetime.now()


class JobMatch(BaseModel):
    job_id: str
    candidate_id: str
    overall_score: float
    skills_match: float
    experience_match: float
    education_match: float
    location_match: float
    salary_expectation_match: float
    matched_skills: List[str] = []
    missing_skills: List[str] = []
    recommendations: List[str] = []
    created_at: datetime = datetime.now()
