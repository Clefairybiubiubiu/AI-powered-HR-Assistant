"""
Candidate API endpoints
"""
from fastapi import APIRouter, HTTPException, UploadFile, File
from typing import List, Dict, Any
import json

from backend.models.candidate import Candidate, CandidateProfile
from backend.services.document_parser import DocumentParser
from backend.services.skill_extractor import SkillExtractor
from backend.utils.candidate_formatter import normalize_candidate_input

router = APIRouter()

# Initialize services
document_parser = DocumentParser()
skill_extractor = SkillExtractor()


@router.post("/parse", response_model=Candidate)
async def parse_resume(file: UploadFile = File(...)):
    """
    Parse uploaded resume file and extract candidate information
    Automatically normalizes inconsistent or partial data
    """
    try:
        # Read file content
        content = await file.read()
        
        # Parse document
        parsed_data = document_parser.parse_document(content, file.filename)
        
        # Normalize candidate data using the formatter
        normalized_data = normalize_candidate_input(parsed_data['text'], parsed_data)
        
        # Extract additional skills if needed
        skills = skill_extractor.extract_skills(parsed_data['text'])
        if normalized_data['skills']:
            # Combine normalized skills with extracted skills
            all_skills = list(set(normalized_data['skills'] + skills))
        else:
            all_skills = skills
        
        # Create candidate object with normalized data
        candidate = Candidate(
            name=normalized_data['name'] or parsed_data.get('name', 'Unknown'),
            email=normalized_data['email'] or parsed_data.get('email'),
            phone=normalized_data['phone'] or parsed_data.get('phone'),
            skills=all_skills,
            raw_text=parsed_data['text'],
            summary=normalized_data['experience_summary'] or parsed_data.get('summary')
        )
        
        return candidate
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error parsing resume: {str(e)}")


@router.post("/parse-flexible", response_model=Candidate)
async def parse_resume_flexible(data: Dict[str, Any]):
    """
    Parse candidate data with flexible input format
    Automatically normalizes inconsistent or partial data
    """
    try:
        # Extract text content from the data
        text_content = ""
        if 'text' in data:
            text_content = data['text']
        elif 'resume_text' in data:
            text_content = data['resume_text']
        elif 'content' in data:
            text_content = data['content']
        else:
            # Try to extract text from any string values
            for key, value in data.items():
                if isinstance(value, str) and len(value) > 50:
                    text_content += value + "\n"
        
        # Normalize candidate data
        normalized_data = normalize_candidate_input(text_content, data)
        
        # Extract skills
        skills = skill_extractor.extract_skills(text_content)
        if normalized_data['skills']:
            all_skills = list(set(normalized_data['skills'] + skills))
        else:
            all_skills = skills
        
        # Create candidate object
        candidate = Candidate(
            name=normalized_data['name'],
            email=normalized_data['email'],
            phone=normalized_data['phone'],
            skills=all_skills,
            raw_text=text_content,
            summary=normalized_data['experience_summary']
        )
        
        return candidate
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error parsing candidate data: {str(e)}")


@router.get("/", response_model=List[Candidate])
async def get_candidates():
    """
    Get all parsed candidates
    """
    # TODO: Implement database storage and retrieval
    return []


@router.get("/{candidate_id}", response_model=Candidate)
async def get_candidate(candidate_id: str):
    """
    Get specific candidate by ID
    """
    # TODO: Implement database retrieval
    raise HTTPException(status_code=404, detail="Candidate not found")
from fastapi import APIRouter, UploadFile, File, HTTPException
from backend.services.document_parser import DocumentParser

router = APIRouter(prefix="/api/v1/candidates", tags=["Candidates"])
parser = DocumentParser()

@router.post("/parse")
async def parse_candidate(file: UploadFile = File(...)):
    """Upload and parse a resume file (PDF, DOCX, TXT)"""
    try:
        content = await file.read()
        parsed_data = parser.parse_document(content, file.filename)
        return {"message": "File parsed successfully", "data": parsed_data}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
