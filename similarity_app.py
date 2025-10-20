"""
FastAPI application for resume-job similarity scoring
"""
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import re
from typing import Dict, List, Tuple, Any, Optional
import uvicorn
import io
import PyPDF2
import docx

# Initialize FastAPI app
app = FastAPI(
    title="Resume-Job Similarity API",
    description="Calculate similarity scores between resumes and job descriptions",
    version="1.0.0"
)

# Initialize SentenceTransformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Scoring weights
ALPHA = 0.4  # skill_match weight
BETA = 0.4   # experience_alignment weight  
GAMMA = 0.2  # education_match weight


def parse_pdf(file_content: bytes) -> str:
    """Parse PDF file content and extract text"""
    try:
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(file_content))
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        return text
    except Exception as e:
        raise ValueError(f"Error parsing PDF: {str(e)}")


def parse_docx(file_content: bytes) -> str:
    """Parse DOCX file content and extract text"""
    try:
        doc = docx.Document(io.BytesIO(file_content))
        text = ""
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
        return text
    except Exception as e:
        raise ValueError(f"Error parsing DOCX: {str(e)}")


def parse_uploaded_file(uploaded_file: UploadFile) -> str:
    """Parse uploaded file and extract text"""
    if not uploaded_file:
        raise ValueError("No file provided")
    
    file_content = uploaded_file.file.read()
    file_type = uploaded_file.content_type
    
    if file_type == "application/pdf":
        return parse_pdf(file_content)
    elif file_type in ["application/vnd.openxmlformats-officedocument.wordprocessingml.document", 
                       "application/msword"]:
        return parse_docx(file_content)
    elif file_type == "text/plain":
        return file_content.decode('utf-8')
    else:
        raise ValueError(f"Unsupported file format: {uploaded_file.filename.split('.')[-1] if uploaded_file.filename else 'unknown'}")


class SimilarityRequest(BaseModel):
    resume: str
    job_desc: str


class SimilarityResponse(BaseModel):
    similarity_score: float
    skill_match: float
    experience_alignment: float
    education_match: float
    details: Dict[str, Any]


class SimilarityCalculator:
    """Calculate similarity between resume and job description"""
    
    def __init__(self):
        self.model = model
        
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
            ]
        }
        
        # Experience level keywords
        self.experience_keywords = {
            'entry': ['entry', 'junior', '0-2 years', '1-3 years', 'fresh graduate', 'new graduate'],
            'mid': ['mid', 'intermediate', '3-5 years', '4-6 years', '2-4 years'],
            'senior': ['senior', 'lead', 'principal', '5+ years', '6+ years', '7+ years', 'architect']
        }
        
        # Education keywords
        self.education_keywords = {
            'high_school': ['high school', 'diploma', 'certificate'],
            'bachelor': ['bachelor', 'bs', 'ba', 'bsc', 'undergraduate', 'college'],
            'master': ['master', 'ms', 'ma', 'msc', 'mba', 'graduate'],
            'phd': ['phd', 'doctorate', 'doctoral', 'ph.d']
        }
    
    def calculate_similarity(self, resume: str, job_desc: str) -> Dict[str, any]:
        """Calculate comprehensive similarity score"""
        
        # Extract skills from both documents
        resume_skills = self._extract_skills(resume)
        job_skills = self._extract_skills(job_desc)
        
        # Calculate skill match
        skill_match = self._calculate_skill_match(resume_skills, job_skills)
        
        # Calculate experience alignment
        experience_alignment = self._calculate_experience_alignment(resume, job_desc)
        
        # Calculate education match
        education_match = self._calculate_education_match(resume, job_desc)
        
        # Calculate overall similarity score
        overall_score = (ALPHA * skill_match + 
                        BETA * experience_alignment + 
                        GAMMA * education_match)
        
        # Calculate semantic similarity using embeddings
        semantic_similarity = self._calculate_semantic_similarity(resume, job_desc)
        
        return {
            'similarity_score': float(overall_score),
            'skill_match': float(skill_match),
            'experience_alignment': float(experience_alignment),
            'education_match': float(education_match),
            'semantic_similarity': float(semantic_similarity),
            'resume_skills': resume_skills,
            'job_skills': job_skills,
            'matched_skills': list(set(resume_skills) & set(job_skills)),
            'missing_skills': list(set(job_skills) - set(resume_skills))
        }
    
    def _extract_skills(self, text: str) -> List[str]:
        """Extract technical skills from text"""
        text_lower = text.lower()
        found_skills = []
        
        for category, skills in self.technical_skills.items():
            for skill in skills:
                if skill.lower() in text_lower:
                    found_skills.append(skill)
        
        return list(set(found_skills))  # Remove duplicates
    
    def _calculate_skill_match(self, resume_skills: List[str], job_skills: List[str]) -> float:
        """Calculate skill matching score"""
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
            embeddings = self.model.encode(all_skills)
            
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
    
    def _calculate_experience_alignment(self, resume: str, job_desc: str) -> float:
        """Calculate experience level alignment"""
        resume_exp = self._extract_experience_level(resume)
        job_exp = self._extract_experience_level(job_desc)
        
        if not resume_exp or not job_exp:
            return 0.5  # Neutral if can't determine
        
        # Map to numeric levels
        exp_levels = {'entry': 1, 'mid': 2, 'senior': 3}
        
        resume_level = exp_levels.get(resume_exp, 2)
        job_level = exp_levels.get(job_exp, 2)
        
        # Candidate can be higher level than required
        if resume_level >= job_level:
            return 1.0
        else:
            # Penalty for being underqualified
            return max(0.0, 1.0 - (job_level - resume_level) * 0.3)
    
    def _calculate_education_match(self, resume: str, job_desc: str) -> float:
        """Calculate education requirement match"""
        resume_edu = self._extract_education_level(resume)
        job_edu = self._extract_education_level(job_desc)
        
        if not job_edu:
            return 1.0  # No education requirements
        
        if not resume_edu:
            return 0.0  # No education found
        
        # Map to numeric levels
        edu_levels = {'high_school': 1, 'bachelor': 2, 'master': 3, 'phd': 4}
        
        resume_level = max([edu_levels.get(edu, 0) for edu in resume_edu])
        job_level = max([edu_levels.get(edu, 0) for edu in job_edu])
        
        if resume_level >= job_level:
            return 1.0
        else:
            return max(0.0, resume_level / job_level)
    
    def _calculate_semantic_similarity(self, text1: str, text2: str) -> float:
        """Calculate semantic similarity using embeddings"""
        if not text1 or not text2:
            return 0.0
        
        # Clean texts
        text1_clean = re.sub(r'\s+', ' ', text1).strip()
        text2_clean = re.sub(r'\s+', ' ', text2).strip()
        
        # Generate embeddings
        embeddings = self.model.encode([text1_clean, text2_clean])
        
        # Calculate cosine similarity
        similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
        return float(similarity)
    
    def _extract_experience_level(self, text: str) -> str:
        """Extract experience level from text"""
        text_lower = text.lower()
        
        for level, keywords in self.experience_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                return level
        
        return 'mid'  # Default assumption
    
    def _extract_education_level(self, text: str) -> List[str]:
        """Extract education level from text"""
        text_lower = text.lower()
        education = []
        
        for level, keywords in self.education_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                education.append(level)
        
        return education


# Initialize calculator
calculator = SimilarityCalculator()


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Resume-Job Similarity API",
        "version": "1.0.0",
        "endpoints": {
            "similarity": "POST /similarity",
            "health": "GET /health"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "model": "all-MiniLM-L6-v2"}


@app.post("/similarity", response_model=SimilarityResponse)
async def calculate_similarity(request: SimilarityRequest):
    """
    Calculate similarity score between resume and job description
    
    Returns:
    - similarity_score: Overall weighted score (α * skill_match + β * experience_alignment + γ * education_match)
    - skill_match: Technical skills matching score
    - experience_alignment: Experience level alignment score  
    - education_match: Education requirement match score
    - details: Additional information about the analysis
    """
    try:
        if not request.resume or not request.job_desc:
            raise HTTPException(
                status_code=400, 
                detail="Both resume and job_desc fields are required"
            )
        
        # Calculate similarity
        result = calculator.calculate_similarity(request.resume, request.job_desc)
        
        return SimilarityResponse(
            similarity_score=result['similarity_score'],
            skill_match=result['skill_match'],
            experience_alignment=result['experience_alignment'],
            education_match=result['education_match'],
            details={
                'semantic_similarity': result['semantic_similarity'],
                'resume_skills': result['resume_skills'],
                'job_skills': result['job_skills'],
                'matched_skills': result['matched_skills'],
                'missing_skills': result['missing_skills'],
                'weights': {
                    'alpha': ALPHA,
                    'beta': BETA,
                    'gamma': GAMMA
                }
            }
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error calculating similarity: {str(e)}")


@app.post("/similarity-files", response_model=SimilarityResponse)
async def calculate_similarity_from_files(
    resume_file: UploadFile = File(...),
    job_desc: str = Form(...)
):
    """
    Calculate similarity score from uploaded resume file and job description text
    
    Args:
    - resume_file: Uploaded resume file (PDF, DOCX, TXT)
    - job_desc: Job description text
    
    Returns:
    - similarity_score: Overall weighted score
    - skill_match: Technical skills matching score
    - experience_alignment: Experience level alignment score  
    - education_match: Education requirement match score
    - details: Additional information about the analysis
    """
    try:
        if not resume_file:
            raise HTTPException(status_code=400, detail="Resume file is required")
        
        if not job_desc:
            raise HTTPException(status_code=400, detail="Job description is required")
        
        # Parse the uploaded resume file
        resume_text = parse_uploaded_file(resume_file)
        
        if not resume_text.strip():
            raise HTTPException(status_code=400, detail="No text could be extracted from the resume file")
        
        # Calculate similarity
        result = calculator.calculate_similarity(resume_text, job_desc)
        
        return SimilarityResponse(
            similarity_score=result['similarity_score'],
            skill_match=result['skill_match'],
            experience_alignment=result['experience_alignment'],
            education_match=result['education_match'],
            details={
                'semantic_similarity': result['semantic_similarity'],
                'resume_skills': result['resume_skills'],
                'job_skills': result['job_skills'],
                'matched_skills': result['matched_skills'],
                'missing_skills': result['missing_skills'],
                'weights': {
                    'alpha': ALPHA,
                    'beta': BETA,
                    'gamma': GAMMA
                },
                'parsed_resume_text': resume_text[:500] + "..." if len(resume_text) > 500 else resume_text
            }
        )
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing files: {str(e)}")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
