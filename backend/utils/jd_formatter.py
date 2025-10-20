"""
Job Description Formatter and Normalizer

Automatically extracts and maps job description data into a standardized schema
regardless of input format (long text, inconsistent fields, etc.)
"""
import re
from typing import Dict, Any, Optional
import json


def normalize_jd_input(raw_input: Dict[str, Any]) -> Dict[str, str]:
    """
    Normalize job description input into standardized schema
    
    Args:
        raw_input: Raw job description data (can be messy or inconsistent)
        
    Returns:
        Dict with standardized fields: title, description, requirements, location, salary_range
    """
    
    # If input already matches schema, return as-is
    if _is_valid_schema(raw_input):
        return {
            "title": raw_input.get("title", ""),
            "description": raw_input.get("description", ""),
            "requirements": raw_input.get("requirements", ""),
            "location": raw_input.get("location", ""),
            "salary_range": raw_input.get("salary_range", "")
        }
    
    # Extract text content for analysis
    text_content = _extract_text_content(raw_input)
    
    # Extract each field using NLP/regex heuristics
    title = _extract_title(text_content, raw_input)
    description = _extract_description(text_content, raw_input)
    requirements = _extract_requirements(text_content, raw_input)
    location = _extract_location(text_content, raw_input)
    salary_range = _extract_salary_range(text_content, raw_input)
    
    return {
        "title": title,
        "description": description,
        "requirements": requirements,
        "location": location,
        "salary_range": salary_range
    }


def _is_valid_schema(data: Dict[str, Any]) -> bool:
    """Check if input already matches the expected schema"""
    required_fields = {"title", "description", "requirements", "location", "salary_range"}
    return all(field in data for field in required_fields)


def _extract_text_content(raw_input: Dict[str, Any]) -> str:
    """Extract all text content from raw input for analysis"""
    text_parts = []
    
    for key, value in raw_input.items():
        if isinstance(value, str) and value.strip():
            text_parts.append(value.strip())
        elif isinstance(value, list):
            text_parts.extend([str(item).strip() for item in value if str(item).strip()])
    
    return "\n".join(text_parts)


def _extract_title(text_content: str, raw_input: Dict[str, Any]) -> str:
    """Extract job title using multiple heuristics"""
    
    # Check for explicit title fields first
    title_fields = ["title", "job_title", "position", "role", "position_title"]
    for field in title_fields:
        if field in raw_input and raw_input[field]:
            return str(raw_input[field]).strip()
    
    # Look for title patterns in text
    title_patterns = [
        r"^(?:Job Title|Position|Role|Title):\s*(.+)$",
        r"^(?:#\s*)?([A-Z][^\\n]{10,80})$",  # First line that looks like a title
        r"^([A-Z][A-Za-z\s&]+(?:Engineer|Developer|Manager|Analyst|Specialist|Coordinator|Director|Lead|Senior|Junior|Intern)[A-Za-z\s]*)$"
    ]
    
    lines = text_content.split('\n')
    for line in lines[:5]:  # Check first 5 lines
        line = line.strip()
        if not line:
            continue
            
        for pattern in title_patterns:
            match = re.search(pattern, line, re.IGNORECASE)
            if match:
                title = match.group(1).strip()
                if len(title) > 5 and len(title) < 100:  # Reasonable title length
                    return title
    
    # Fallback: first non-empty line
    for line in lines:
        line = line.strip()
        if line and len(line) > 5:
            return line
    
    return ""


def _extract_description(text_content: str, raw_input: Dict[str, Any]) -> str:
    """Extract job description using section detection"""
    
    # Check for explicit description fields
    desc_fields = ["description", "job_description", "summary", "about", "overview"]
    for field in desc_fields:
        if field in raw_input and raw_input[field]:
            return str(raw_input[field]).strip()
    
    # Look for description sections
    desc_sections = [
        r"(?:About the job|Job Summary|Overview|Description|Job Description)[:.]?\s*(.+?)(?=\n\s*(?:Requirements|Qualifications|Skills|Responsibilities|Location|Salary|Compensation|Benefits|$))",
        r"(?:Summary|Overview)[:.]?\s*(.+?)(?=\n\s*(?:Requirements|Qualifications|Skills|Responsibilities|Location|Salary|Compensation|Benefits|$))",
        r"^(.+?)(?=\n\s*(?:Requirements|Qualifications|Skills|Responsibilities|Location|Salary|Compensation|Benefits|$))"
    ]
    
    for pattern in desc_sections:
        match = re.search(pattern, text_content, re.IGNORECASE | re.DOTALL)
        if match:
            desc = match.group(1).strip()
            if len(desc) > 20:  # Reasonable description length
                return desc
    
    # Fallback: first substantial paragraph
    paragraphs = [p.strip() for p in text_content.split('\n\n') if p.strip()]
    for para in paragraphs:
        if len(para) > 50 and not _looks_like_requirements(para):
            return para
    
    return ""


def _extract_requirements(text_content: str, raw_input: Dict[str, Any]) -> str:
    """Extract job requirements and qualifications"""
    
    # Check for explicit requirements fields
    req_fields = ["requirements", "qualifications", "skills", "must_have", "required"]
    for field in req_fields:
        if field in raw_input and raw_input[field]:
            return str(raw_input[field]).strip()
    
    # Look for requirements sections
    req_sections = [
        r"(?:Requirements|Qualifications|Skills|Must Have|Required|Necessary)[:.]?\s*(.+?)(?=\n\s*(?:Location|Salary|Compensation|Benefits|Responsibilities|$))",
        r"(?:Education|Experience|Technical Skills)[:.]?\s*(.+?)(?=\n\s*(?:Location|Salary|Compensation|Benefits|$))"
    ]
    
    for pattern in req_sections:
        match = re.search(pattern, text_content, re.IGNORECASE | re.DOTALL)
        if match:
            reqs = match.group(1).strip()
            if len(reqs) > 10:
                return reqs
    
    # Look for bullet points or numbered lists
    bullet_pattern = r"(?:^|\n)\s*[-•*]\s*(.+?)(?=\n\s*[-•*]|\n\s*[A-Z]|\n\s*[0-9]|\n\s*$)"
    bullets = re.findall(bullet_pattern, text_content, re.MULTILINE)
    if bullets:
        return "\n".join([f"• {bullet.strip()}" for bullet in bullets[:10]])  # Limit to 10 items
    
    return ""


def _extract_location(text_content: str, raw_input: Dict[str, Any]) -> str:
    """Extract job location"""
    
    # Check for explicit location fields
    loc_fields = ["location", "city", "address", "place", "where"]
    for field in loc_fields:
        if field in raw_input and raw_input[field]:
            return str(raw_input[field]).strip()
    
    # Look for location patterns
    location_patterns = [
        r"(?:Location|City|Address|Where)[:.]?\s*([A-Za-z\s,]+(?:,\s*[A-Z]{2})?)",
        r"(?:Remote|Hybrid|On-site|Office)[:.]?\s*([A-Za-z\s,]+)?",
        r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*,\s*[A-Z]{2})",  # City, State format
        r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*,\s*[A-Z][a-z]+)"  # City, Country format
    ]
    
    for pattern in location_patterns:
        match = re.search(pattern, text_content, re.IGNORECASE)
        if match:
            location = match.group(1).strip()
            if location and len(location) > 2:
                return location
    
    # Look for common location keywords
    location_keywords = ["Remote", "Hybrid", "On-site", "Office", "New York", "San Francisco", "London", "Toronto"]
    for keyword in location_keywords:
        if keyword.lower() in text_content.lower():
            return keyword
    
    return ""


def _extract_salary_range(text_content: str, raw_input: Dict[str, Any]) -> str:
    """Extract salary range information"""
    
    # Check for explicit salary fields
    salary_fields = ["salary", "compensation", "pay", "wage", "income", "salary_range"]
    for field in salary_fields:
        if field in raw_input and raw_input[field]:
            return str(raw_input[field]).strip()
    
    # Look for salary patterns
    salary_patterns = [
        r"(?:Salary|Compensation|Pay|Wage)[:.]?\s*(\$?[0-9,]+(?:\s*-\s*\$?[0-9,]+)?(?:\s*(?:k|K|thousand|million|per\s+year|per\s+hour|hourly|annually))?)",
        r"(\$[0-9,]+(?:\s*-\s*\$[0-9,]+)?(?:\s*(?:k|K|thousand|million|per\s+year|per\s+hour|hourly|annually))?)",
        r"([0-9,]+(?:\s*-\s*[0-9,]+)?(?:\s*(?:k|K|thousand|million|per\s+year|per\s+hour|hourly|annually))?)"
    ]
    
    for pattern in salary_patterns:
        match = re.search(pattern, text_content, re.IGNORECASE)
        if match:
            salary = match.group(1).strip()
            if salary and len(salary) > 3:
                return salary
    
    return ""


def _looks_like_requirements(text: str) -> bool:
    """Check if text looks like requirements/qualifications"""
    req_keywords = [
        "required", "must have", "qualifications", "skills", "experience",
        "degree", "bachelor", "master", "phd", "years", "proficient",
        "knowledge", "ability", "capable", "familiar", "expertise"
    ]
    
    text_lower = text.lower()
    return any(keyword in text_lower for keyword in req_keywords)


def format_job_description(job_data: Dict[str, str]) -> str:
    """
    Format normalized job data into a readable job description
    
    Args:
        job_data: Normalized job description data
        
    Returns:
        Formatted job description string
    """
    sections = []
    
    if job_data.get("title"):
        sections.append(f"# {job_data['title']}")
    
    if job_data.get("description"):
        sections.append(f"## Job Description\n{job_data['description']}")
    
    if job_data.get("requirements"):
        sections.append(f"## Requirements\n{job_data['requirements']}")
    
    if job_data.get("location"):
        sections.append(f"## Location\n{job_data['location']}")
    
    if job_data.get("salary_range"):
        sections.append(f"## Salary Range\n{job_data['salary_range']}")
    
    return "\n\n".join(sections)


# Test function
def test_jd_formatter():
    """Test the job description formatter with various inputs"""
    
    # Test case 1: Long job description text
    long_text = """
    Senior Python Developer
    
    We are looking for a Senior Python Developer to join our team.
    
    About the Role:
    You will be responsible for developing and maintaining our Python applications.
    You will work with Django, Flask, and other Python frameworks.
    
    Requirements:
    - 5+ years of Python experience
    - Experience with Django and Flask
    - Knowledge of SQL databases
    - Bachelor's degree in Computer Science
    
    Location: San Francisco, CA
    Salary: $120,000 - $150,000 per year
    """
    
    # Test case 2: Inconsistent field names
    inconsistent_input = {
        "job_title": "Data Scientist",
        "job_description": "Analyze data and build ML models",
        "must_have": "Python, SQL, Machine Learning",
        "city": "New York, NY",
        "compensation": "$100,000 - $130,000"
    }
    
    # Test case 3: Already normalized
    normalized_input = {
        "title": "Software Engineer",
        "description": "Develop software applications",
        "requirements": "Python, JavaScript, 3+ years experience",
        "location": "Remote",
        "salary_range": "$80,000 - $120,000"
    }
    
    print("🧪 Testing Job Description Formatter")
    print("=" * 50)
    
    # Test long text
    print("\n1. Testing long text input:")
    result1 = normalize_jd_input({"raw_text": long_text})
    print(f"   Title: {result1['title']}")
    print(f"   Description: {result1['description'][:100]}...")
    print(f"   Requirements: {result1['requirements'][:100]}...")
    print(f"   Location: {result1['location']}")
    print(f"   Salary: {result1['salary_range']}")
    
    # Test inconsistent fields
    print("\n2. Testing inconsistent field names:")
    result2 = normalize_jd_input(inconsistent_input)
    print(f"   Title: {result2['title']}")
    print(f"   Description: {result2['description']}")
    print(f"   Requirements: {result2['requirements']}")
    print(f"   Location: {result2['location']}")
    print(f"   Salary: {result2['salary_range']}")
    
    # Test normalized input
    print("\n3. Testing already normalized input:")
    result3 = normalize_jd_input(normalized_input)
    print(f"   Title: {result3['title']}")
    print(f"   Description: {result3['description']}")
    print(f"   Requirements: {result3['requirements']}")
    print(f"   Location: {result3['location']}")
    print(f"   Salary: {result3['salary_range']}")
    
    print("\n✅ Job Description Formatter tests completed!")
    return True


if __name__ == "__main__":
    test_jd_formatter()
