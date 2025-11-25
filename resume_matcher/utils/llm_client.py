"""
LLM API Client for enhanced HR Assistant functionality.

Uses Google Gemini (FREE tier) for AI-powered enhancements.
"""

import os
import re
import json
import logging
import hashlib
from typing import Optional, Dict, List, Any

# Try to import streamlit (optional, for caching)
try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False
    st = None

logger = logging.getLogger(__name__)


class LLMClient:
    """LLM client using Google Gemini (FREE tier)."""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize LLM client with Google Gemini.
        
        Args:
            api_key: API key (if None, will try to get from environment)
        """
        streamlit_api_key = None
        if STREAMLIT_AVAILABLE:
            streamlit_api_key = getattr(st.session_state, "gemini_api_key", None)
        self.api_key = api_key or streamlit_api_key or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        self.client = None
        self.model = None
        self.available = False
        
        if self.api_key:
            self._initialize_client()
    
    def _initialize_client(self):
        """Initialize the Google Gemini client."""
        try:
            import google.generativeai as genai
            
            # Configure Gemini
            genai.configure(api_key=self.api_key)
            
            # Try to use the latest available free model
            # gemini-2.0-flash is the current free tier model
            model_names = [
                'gemini-2.0-flash',      # Latest free tier (recommended)
                'gemini-2.5-flash',      # Alternative free tier
                'gemini-1.5-flash',      # Older free tier
                'gemini-1.5-pro',        # Pro version (may have limits)
            ]
            
            model_initialized = False
            for model_name in model_names:
                try:
                    self.model = genai.GenerativeModel(model_name)
                    logger.info(f"Using {model_name} model")
                    model_initialized = True
                    break
                except Exception as e:
                    logger.debug(f"Failed to initialize {model_name}: {e}")
                    continue
            
            if not model_initialized:
                raise Exception("Could not initialize any Gemini model. Please check your API key and model availability.")
            
            self.client = genai
            self.available = True
            logger.info("Google Gemini client initialized successfully (FREE tier)")
            
        except ImportError as e:
            logger.warning(f"Google Generative AI package not installed. Install with: pip install google-generativeai. Error: {e}")
            self.available = False
        except Exception as e:
            # Log the full error to help debug
            error_msg = str(e)
            logger.error(f"Failed to initialize Google Gemini client: {error_msg}")
            # If it's a protobuf version issue, provide helpful message
            if "GetPrototype" in error_msg or "protobuf" in error_msg.lower():
                logger.warning("This might be a protobuf version conflict. Try: pip install --upgrade protobuf")
            self.available = False
    
    def generate_text(
        self,
        prompt: str,
        max_tokens: int = 500,
        temperature: float = 0.7,
        model: Optional[str] = None
    ) -> Optional[str]:
        """
        Generate text using Google Gemini.
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate (Gemini uses max_output_tokens)
            temperature: Sampling temperature (0.0-2.0)
            model: Model name (uses default model if None, typically gemini-2.0-flash)
        
        Returns:
            Generated text or None if unavailable
        """
        if not self.available:
            return None
        
        try:
            # Use specified model or default
            if model:
                genai_model = self.client.GenerativeModel(model)
            else:
                genai_model = self.model
            
            # Configure generation parameters
            generation_config = {
                "temperature": temperature,
                "max_output_tokens": max_tokens,
            }
            
            # Generate content
            response = genai_model.generate_content(
                prompt,
                generation_config=generation_config
            )
            
            # Get the full text response
            full_text = response.text.strip()
            
            # Check if response was truncated (finish_reason indicates if generation stopped early)
            if hasattr(response, 'candidates') and response.candidates:
                candidate = response.candidates[0]
                if hasattr(candidate, 'finish_reason'):
                    finish_reason = str(candidate.finish_reason).upper()
                    # If truncated due to max tokens, log a warning
                    if 'MAX_TOKENS' in finish_reason or 'LENGTH' in finish_reason:
                        logger.warning(f"Response may be truncated at max_tokens={max_tokens}. Consider increasing max_tokens for longer responses.")
                    elif 'SAFETY' in finish_reason:
                        logger.warning("Response blocked by safety settings.")
                    elif 'RECITATION' in finish_reason:
                        logger.warning("Response blocked due to recitation detection.")
            
            return full_text
            
        except Exception as e:
            logger.error(f"Error generating text with Gemini: {e}")
            return None
    
    def generate_match_explanation(
        self,
        resume_name: str,
        jd_name: str,
        match_score: float,
        section_scores: Dict[str, float],
        resume_summary: str,
        jd_requirements: str
    ) -> str:
        """
        Generate a detailed explanation for a resume-JD match using Google Gemini.
        
        Args:
            resume_name: Name of the candidate
            jd_name: Name of the job description
            match_score: Overall match score (0-1)
            section_scores: Dictionary of section scores
            resume_summary: Summary of the resume
            jd_requirements: Job description requirements
        
        Returns:
            Natural language explanation
        """
        if not self.available:
            return self._fallback_explanation(resume_name, jd_name, match_score, section_scores)
        
        prompt = f"""Analyze the match between a candidate resume and a job description.

Candidate: {resume_name}
Job Description: {jd_name}
Overall Match Score: {match_score:.2%}

Section Scores:
{chr(10).join([f"- {section}: {score:.2%}" for section, score in section_scores.items()])}

Resume Summary:
{resume_summary[:500]}

Job Requirements:
{jd_requirements[:500]}

Provide a concise, professional explanation (2-3 sentences) of why this candidate matches or doesn't match the job requirements. Highlight:
1. The strongest alignment areas
2. Any gaps or concerns
3. Overall fit assessment

Explanation:"""
        
        # Use lower temperature for more deterministic explanations
        explanation = self.generate_text(prompt, max_tokens=200, temperature=0.3)
        return explanation or self._fallback_explanation(resume_name, jd_name, match_score, section_scores)
    
    def generate_professional_summary(
        self,
        experience: str,
        skills: str,
        education: str,
        raw_text: Optional[str] = None
    ) -> str:
        """
        Generate a professional summary from resume sections using Google Gemini.
        
        Args:
            experience: Experience section text
            skills: Skills section text
            education: Education section text
            raw_text: Raw resume text (optional)
        
        Returns:
            Professional summary
        """
        if not self.available:
            return self._fallback_summary(experience, skills, education)
        
        # Use full text without truncation to provide complete context
        experience_text = experience if experience else "Not specified"
        skills_text = skills if skills else "Not specified"
        education_text = education if education else "Not specified"
        
        raw_text_context = ""
        if raw_text:
            raw_text_context = f"""

Additional Resume Context (use this to fill gaps, do NOT truncate):
{raw_text[:4000]}
"""
        
        prompt = f"""Generate a comprehensive professional summary for a resume based on the following information. The summary should be detailed and complete, covering all important aspects of the candidate's background.

Experience:
{experience_text}

Skills:
{skills_text}

Education:
{education_text}

{raw_text_context}

Create a detailed, professional summary that:
1. Highlights the candidate's primary role, years of experience, and expertise areas
2. Mentions key technical skills, tools, and technologies in detail
3. Includes notable achievements, specializations, or industry experience
4. Includes relevant education and certifications
5. Is written in third person, professional tone
6. Provides enough detail to give a complete picture of the candidate
7. Uses complete sentences with no ellipses ("..." or "…") and no omissions of information

Make the summary comprehensive and informative (6-10 sentences), covering all relevant experience, skills, and education. Do not truncate or abbreviate any information.

Professional Summary:"""
        
        # Use lower temperature for more deterministic summaries, increased max_tokens for complete summaries
        # Increased to 2000 tokens to ensure complete summaries without truncation
        summary = self.generate_text(prompt, max_tokens=2000, temperature=0.3)
        
        # If Gemini still produced ellipses, retry once with explicit rewrite instructions
        if summary and ('...' in summary or '…' in summary) and raw_text:
            rewrite_prompt = f"""{prompt}

The previous summary response included ellipses. Rewrite the summary again, ensuring:
- Every sentence is complete
- No ellipses ("..." or "…") are used anywhere
- All details are fully written out without placeholders

Professional Summary:"""
            retry_summary = self.generate_text(rewrite_prompt, max_tokens=2000, temperature=0.25)
            if retry_summary:
                summary = retry_summary
        
        return summary or self._fallback_summary(experience, skills, education)
    
    def enhance_resume_parsing(
        self,
        raw_text: str,
        target_section: str = "all"
    ) -> Dict[str, Any]:
        """
        Use Google Gemini to enhance resume parsing and extract structured information.
        
        Args:
            raw_text: Raw resume text
            target_section: Section to extract ('education', 'skills', 'experience', 'summary', 'all')
        
        Returns:
            Dictionary with extracted information
        """
        if not self.available:
            return {}
        
        # Cache key for deterministic results
        cache_key = hashlib.sha256(f"{raw_text[:1000]}_{target_section}".encode('utf-8')).hexdigest()
        cache_storage_key = f"llm_parsing_cache_{cache_key}"
        
        # Check cache first (if available in session state)
        if STREAMLIT_AVAILABLE and cache_storage_key in st.session_state:
            return st.session_state[cache_storage_key]
        
        sections = ["education", "skills", "experience", "summary"] if target_section == "all" else [target_section]
        
        result = {}
        for section in sections:
            prompt = f"""Extract the {section} section from the following resume text. 
Provide only the relevant {section} information, cleaned and formatted.

Resume Text:
{raw_text[:2000]}

{section.title()} Information:"""
            
            # Use lower temperature for more deterministic extraction
            extracted = self.generate_text(prompt, max_tokens=300, temperature=0.1)
            if extracted:
                result[section] = extracted
        
        # Cache result in session state if available
        if STREAMLIT_AVAILABLE:
            st.session_state[cache_storage_key] = result
        
        return result
    
    def generate_skill_taxonomy(
        self,
        skills_text: str,
        resume_text: Optional[str] = None
    ) -> Dict[str, List[str]]:
        """Use Gemini to organize skills into meaningful categories."""
        if not self.available or not skills_text:
            return {}
        
        context_snippet = resume_text[:1200] if resume_text else ""
        cache_key = hashlib.sha256(f"skill_taxonomy_{skills_text}_{context_snippet}".encode("utf-8")).hexdigest()
        cache_storage_key = f"skill_taxonomy_cache_{cache_key}"
        if STREAMLIT_AVAILABLE and cache_storage_key in st.session_state:
            return st.session_state[cache_storage_key]
        
        prompt = f"""You are an assistant creating a structured skill taxonomy from a resume.
Group the skills into meaningful categories and output valid JSON with lowercase keys.

Required categories (omit any that do not apply):
- core_languages (e.g., Python, Java, SQL)
- data_platforms_and_tools (databases, warehouses, ETL, analytics)
- cloud_and_infrastructure (AWS, Azure, GCP, Kubernetes, Terraform, CI/CD)
- ml_ai_analytics (ML frameworks, MLOps, LLM tooling, analytics stacks)
- leadership_or_domain (product, industry, stakeholder, leadership skills)
- other_highlights (certifications or standout items)

Skills to categorize:
{skills_text}

Additional resume context (optional):
{context_snippet}

Return ONLY JSON (no prose). Example:
{{
  "core_languages": ["Python", "SQL"],
  "cloud_and_infrastructure": ["AWS", "Docker"],
  "other_highlights": ["AWS Certified Solutions Architect"]
}}"""
        
        response = self.generate_text(prompt, max_tokens=600, temperature=0.2)
        taxonomy: Dict[str, List[str]] = {}
        if response:
            cleaned = response.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.strip("`")
                if "\n" in cleaned:
                    cleaned = cleaned.split("\n", 1)[1]
            try:
                parsed = json.loads(cleaned)
                if isinstance(parsed, dict):
                    for key, value in parsed.items():
                        if isinstance(value, list):
                            normalized = [str(item).strip() for item in value if str(item).strip()]
                            if normalized:
                                taxonomy[key] = normalized
            except json.JSONDecodeError:
                taxonomy = {"insights": [cleaned]}
        
        if STREAMLIT_AVAILABLE:
            st.session_state[cache_storage_key] = taxonomy
        
        return taxonomy
    
    def compare_resume_versions(
        self,
        resume_a: str,
        resume_b: str,
        label_a: str,
        label_b: str
    ) -> str:
        """Summarize key differences between two resume versions using Gemini."""
        if not resume_a or not resume_b:
            return "Insufficient data to compare resume versions."
        
        if not self.available:
            return self._fallback_version_diff(label_a, label_b, resume_a, resume_b)
        
        cache_payload = f"{label_a}:{resume_a[:1200]}\n{label_b}:{resume_b[:1200]}"
        cache_key = hashlib.sha256(cache_payload.encode("utf-8")).hexdigest()
        cache_storage_key = f"resume_diff_cache_{cache_key}"
        if STREAMLIT_AVAILABLE and cache_storage_key in st.session_state:
            return st.session_state[cache_storage_key]
        
        prompt = f"""You will compare two versions of the same resume and describe how they differ.
Focus on upgrades, new content, removed or reduced content, and any potential regressions.

Version A ({label_a}):
{resume_a[:2000]}

Version B ({label_b}):
{resume_b[:2000]}

Provide the response in markdown with the following sections:
1. **Summary of Changes** (2 sentences)
2. **Improvements / New Additions** (bullet list)
3. **Removed or Reduced Details** (bullet list)
4. **Potential Regressions or Gaps** (bullet list, note if none)

Be specific about technologies, metrics, and responsibilities that changed."""
        
        diff_summary = self.generate_text(prompt, max_tokens=700, temperature=0.25)
        if not diff_summary:
            diff_summary = self._fallback_version_diff(label_a, label_b, resume_a, resume_b)
        
        if STREAMLIT_AVAILABLE:
            st.session_state[cache_storage_key] = diff_summary
        
        return diff_summary
    
    def extract_candidate_name(self, resume_text: str) -> str:
        """
        Extract candidate name from resume using Gemini.
        
        Args:
            resume_text: Full resume text
        
        Returns:
            Candidate name or empty string
        """
        if not self.available:
            return ""
        
        prompt = f"""Extract the candidate's full name from the following resume text.
Return only the name, nothing else. If no name is found, return "Unknown Candidate".

Resume Text (first 500 characters):
{resume_text[:500]}

Candidate Name:"""
        
        name = self.generate_text(prompt, max_tokens=50, temperature=0.1)
        if name:
            # Clean up the name
            name = name.strip().split('\n')[0].strip()
            # Remove common prefixes/suffixes
            name = re.sub(r'^(name|Name|Candidate|Resume)\s*:?\s*', '', name, flags=re.IGNORECASE)
            name = name.strip()
            if name and len(name) > 1:
                return name
        
        return ""
    
    def extract_jd_requirements_enhanced(self, jd_text: str) -> Dict[str, List[str]]:
        """
        Extract structured requirements from job description using Gemini.
        
        Args:
            jd_text: Job description text
        
        Returns:
            Dictionary with 'education', 'skills', 'experience' requirements
        """
        if not self.available:
            return {'education': [], 'skills': [], 'experience': []}
        
        prompt = f"""Extract requirements from this job description and organize them into three categories:
1. Education requirements (degrees, certifications, educational background)
2. Skills requirements (technical skills, tools, technologies)
3. Experience requirements (years of experience, work experience, industry experience)

Job Description:
{jd_text[:1500]}

Format your response as:
EDUCATION:
- [requirement 1]
- [requirement 2]

SKILLS:
- [skill 1]
- [skill 2]

EXPERIENCE:
- [requirement 1]
- [requirement 2]"""
        
        response = self.generate_text(prompt, max_tokens=400, temperature=0.3)
        if not response:
            return {'education': [], 'skills': [], 'experience': []}
        
        # Parse the response
        result = {'education': [], 'skills': [], 'experience': []}
        current_section = None
        
        for line in response.split('\n'):
            line = line.strip()
            if not line or line.startswith('-') and len(line) < 3:
                continue
            
            if 'EDUCATION' in line.upper():
                current_section = 'education'
            elif 'SKILLS' in line.upper():
                current_section = 'skills'
            elif 'EXPERIENCE' in line.upper():
                current_section = 'experience'
            elif current_section and line.startswith('-'):
                requirement = line[1:].strip()
                if requirement:
                    result[current_section].append(requirement)
        
        return result
    
    def generate_candidate_profile(self, resume_text: str) -> Dict[str, str]:
        """
        Generate a comprehensive candidate profile using Gemini API.
        
        Args:
            resume_text: Full resume text
        
        Returns:
            Dictionary with structured profile information
        """
        if not self.available:
            return {}
        
        # Cache key for deterministic results
        cache_key = hashlib.sha256(f"profile_{resume_text[:1000]}".encode('utf-8')).hexdigest()
        cache_storage_key = f"llm_profile_cache_{cache_key}"
        
        # Check cache first
        if STREAMLIT_AVAILABLE and cache_storage_key in st.session_state:
            return st.session_state[cache_storage_key]
        
        # Use full resume text without truncation to get complete information
        prompt = f"""Extract and structure the following resume information into a comprehensive candidate profile.
Provide the information in a clear, organized format with detailed summaries.

Resume Text:
{resume_text}

Extract and provide:
1. Contact Information: Name, Email, Phone, Location (format as: Name: [name], Email: [email], Phone: [phone], Location: [location])

2. Professional Summary: Write a comprehensive professional summary (6-10 sentences) that:
   - Highlights the candidate's primary role, years of experience, and expertise areas
   - Mentions key technical skills, tools, and technologies in detail
   - Includes notable achievements, specializations, or industry experience
   - Includes relevant education and certifications
   - Provides a complete picture of the candidate's professional background
   Format as a flowing paragraph, NOT bullet points.

3. Skills: Comma-separated list of all technical and professional skills mentioned in the resume

4. Work Experience: Write a comprehensive summary (4-8 sentences) of the candidate's work experience that:
   - Describes their career progression and key roles
   - Highlights major achievements and responsibilities
   - Mentions technologies, tools, and methodologies used
   - Shows years of experience and career growth
   Format as a flowing paragraph summary, NOT bullet points.

5. Education: Write a comprehensive summary (3-5 sentences) of the candidate's educational background that:
   - Lists all degrees, institutions, and graduation years
   - Mentions relevant coursework, honors, or specializations
   - Includes certifications if mentioned
   Format as a flowing paragraph summary, NOT bullet points.

Format the response clearly with section headers. Make sure all summaries are comprehensive and detailed, not brief or truncated.

Candidate Profile:"""
        
        profile_text = self.generate_text(prompt, max_tokens=1500, temperature=0.2)
        
        if not profile_text:
            return {}
        
        # Parse the structured response
        profile = {
            'contact_information': '',
            'professional_summary': '',
            'skills': '',
            'work_experience': '',
            'education': ''
        }
        
        # Extract sections from the response
        current_section = None
        lines = profile_text.split('\n')
        
        for line in lines:
            line_lower = line.lower().strip()
            
            # Detect section headers
            if 'contact' in line_lower or 'name:' in line_lower:
                current_section = 'contact_information'
                profile[current_section] = line.strip()
            elif 'summary' in line_lower or 'profile' in line_lower:
                current_section = 'professional_summary'
                if line_lower not in ['summary:', 'professional summary:', 'profile:']:
                    profile[current_section] = line.strip()
            elif 'skill' in line_lower and ('technical' in line_lower or 'competenc' in line_lower):
                current_section = 'skills'
            elif 'experience' in line_lower or 'work' in line_lower:
                current_section = 'work_experience'
            elif 'education' in line_lower or 'degree' in line_lower or 'university' in line_lower:
                current_section = 'education'
            elif current_section and line.strip():
                # Add content to current section
                if profile[current_section]:
                    profile[current_section] += '\n' + line.strip()
                else:
                    profile[current_section] = line.strip()
        
        # If parsing failed, use the raw text for summary
        if not profile['professional_summary']:
            # Try to extract first paragraph as summary
            paragraphs = profile_text.split('\n\n')
            if paragraphs:
                profile['professional_summary'] = paragraphs[0].strip()
        
        # Cache result
        if STREAMLIT_AVAILABLE:
            st.session_state[cache_storage_key] = profile
        
        return profile
    
    def extract_skills_list(self, resume_text: str) -> List[str]:
        """
        Extract a clean list of skills from resume using Gemini.
        
        Args:
            resume_text: Resume text (can be full or skills section)
        
        Returns:
            List of skills
        """
        if not self.available:
            return []
        
        prompt = f"""Extract all technical and professional skills from this resume text.
Return only a comma-separated list of skills, nothing else.

Resume Text:
{resume_text[:1000]}

Skills (comma-separated):"""
        
        response = self.generate_text(prompt, max_tokens=200, temperature=0.2)
        if not response:
            return []
        
        # Parse comma-separated skills
        skills = [s.strip() for s in response.split(',') if s.strip()]
        # Clean up each skill
        cleaned_skills = []
        for skill in skills:
            skill = re.sub(r'^(skills?|Skills?)\s*:?\s*', '', skill, flags=re.IGNORECASE)
            skill = skill.strip()
            if skill and len(skill) > 1:
                cleaned_skills.append(skill)
        
        return cleaned_skills[:20]  # Limit to top 20
    
    def _fallback_explanation(
        self,
        resume_name: str,
        jd_name: str,
        match_score: float,
        section_scores: Dict[str, float]
    ) -> str:
        """Fallback explanation when LLM is not available."""
        strong_sections = [s for s, score in section_scores.items() if score > 0.3]
        
        if not strong_sections:
            return f"Limited alignment between {resume_name} and {jd_name} requirements (match score: {match_score:.2%})."
        
        if len(strong_sections) == 1:
            return f"Strong match in {strong_sections[0]} ({section_scores[strong_sections[0]]:.2%}) between {resume_name} and {jd_name}."
        else:
            avg_score = sum(section_scores[s] for s in strong_sections) / len(strong_sections)
            return f"Strong alignment across {', '.join(strong_sections)} (avg: {avg_score:.2%}) between {resume_name} and {jd_name}."
    
    def _fallback_summary(self, experience: str, skills: str, education: str) -> str:
        """Fallback summary when LLM is not available."""
        parts = []
        
        if experience:
            parts.append(f"Professional with experience in {experience[:100]}...")
        if skills:
            parts.append(f"Proficient in {skills[:100]}...")
        if education:
            parts.append(f"Education: {education[:100]}...")
        
        return " ".join(parts) if parts else "Professional summary not available."
    
    def _fallback_version_diff(
        self,
        label_a: str,
        label_b: str,
        resume_a: str,
        resume_b: str
    ) -> str:
        """Simple heuristic comparison when LLM is unavailable."""
        len_a = len(resume_a or "")
        len_b = len(resume_b or "")
        delta = len_b - len_a
        direction = "longer" if delta > 0 else "shorter" if delta < 0 else "the same length as"
        summary = [
            f"{label_b} is {abs(delta)} characters {direction} {label_a}."
        ]
        return "\n".join(summary)

    def chat_with_robert(
        self,
        history: List[Dict[str, str]],
        context: Optional[str] = None,
        persona: Optional[str] = None,
        cache_key: Optional[str] = None
    ) -> str:
        """
        Conversational helper acting as the 'Robert' assistant.
        """
        if not self.available:
            return self._fallback_chat(history, context)
        
        if STREAMLIT_AVAILABLE and cache_key and cache_key in st.session_state:
            cached_response = st.session_state[cache_key]
            if cached_response:
                return cached_response
        
        system_prompt = persona or (
            "You are Robert, a proactive HR assistant built into a resume-to-job matching dashboard. "
            "Use the provided context (resume sections, JD highlights, errors) to explain issues and suggest concrete next steps. "
            "Always keep responses concise (3-5 sentences) and reference UI actions when relevant."
        )
        
        context_block = f"Context:\n{context.strip()}\n\n" if context else ""
        conversation_text = self._build_conversation_transcript(history)
        
        prompt = f"""{system_prompt}

{context_block}Conversation so far:
{conversation_text}
Robert:"""
        
        response = self.generate_text(prompt, max_tokens=600, temperature=0.35)
        final_response = response or self._fallback_chat(history, context)
        
        if STREAMLIT_AVAILABLE and cache_key:
            st.session_state[cache_key] = final_response
        
        return final_response

    def _build_conversation_transcript(self, history: List[Dict[str, str]], limit: int = 12) -> str:
        """Format conversation history for prompting."""
        transcript_lines = []
        for entry in history[-limit:]:
            role = entry.get("role", "user")
            label = "User" if role == "user" else "Robert"
            message = (entry.get("content") or "").strip()
            if message:
                transcript_lines.append(f"{label}: {message}")
        if not transcript_lines:
            transcript_lines.append("User: Hi Robert, can you help me with this?")
        return "\n".join(transcript_lines)

    def _fallback_chat(
        self,
        history: List[Dict[str, str]],
        context: Optional[str] = None
    ) -> str:
        """Fallback chat response when LLM is unavailable."""
        last_user = next((entry.get('content') for entry in reversed(history) if entry.get('role') == 'user' and entry.get('content')), "")
        action_hint = "Try enabling the AI enhancements toggle in the sidebar once you provide a Gemini API key." if STREAMLIT_AVAILABLE else ""
        if last_user:
            return f"I can’t access the AI assistant right now, but I can still help manually. Please share more details about the resume/JD issue you’re seeing. {action_hint}"
        return f"AI enhancements are currently disabled. {action_hint}"


# Global instance
_llm_client = None


def get_llm_client() -> LLMClient:
    """Get or create the global LLM client instance."""
    global _llm_client
    if _llm_client is None:
        _llm_client = LLMClient()
    return _llm_client


def is_llm_available() -> bool:
    """Check if LLM is available."""
    client = get_llm_client()
    return client.available
