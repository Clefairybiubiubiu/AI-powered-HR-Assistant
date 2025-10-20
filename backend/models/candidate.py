from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from datetime import datetime


class Skill(BaseModel):
    name: str
    confidence: float
    category: Optional[str] = None


class Experience(BaseModel):
    title: str
    company: str
    duration: str
    description: str
    skills_used: List[str] = []


class Education(BaseModel):
    degree: str
    institution: str
    year: Optional[int] = None
    field: Optional[str] = None


class Candidate(BaseModel):
    name: str
    email: Optional[str] = None
    phone: Optional[str] = None
    skills: List[Skill] = []
    experience: List[Experience] = []
    education: List[Education] = []
    summary: Optional[str] = None
    raw_text: str
    created_at: datetime = datetime.now()


class CandidateProfile(BaseModel):
    candidate: Candidate
    skills_score: float
    experience_score: float
    education_score: float
    overall_fit: float
    matched_skills: List[str] = []
    missing_skills: List[str] = []
    recommendations: List[str] = []

