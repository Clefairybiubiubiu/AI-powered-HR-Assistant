"""
Candidate Data Formatter and Normalizer

Automatically extracts and maps candidate/resume data into a standardized schema
regardless of input format (parsed text, inconsistent fields, etc.)
"""
import re
from typing import Dict, Any, List, Optional
import json


def normalize_candidate_input(parsed_text: str, raw_input: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Normalize candidate/resume input into standardized schema
    
    Args:
        parsed_text: Text extracted from resume file or input
        raw_input: Optional raw candidate data (can be messy or inconsistent)
        
    Returns:
        Dict with standardized fields: name, email, phone, skills, experience_summary
    """
    
    # If raw_input is provided and already matches schema, return as-is
    if raw_input and _is_valid_candidate_schema(raw_input):
        return {
            "name": raw_input.get("name", ""),
            "email": raw_input.get("email", ""),
            "phone": raw_input.get("phone", ""),
            "skills": raw_input.get("skills", []),
            "experience_summary": raw_input.get("experience_summary", "")
        }
    
    # Extract each field using NLP/regex heuristics
    name = _extract_name(parsed_text, raw_input)
    email = _extract_email(parsed_text, raw_input)
    phone = _extract_phone(parsed_text, raw_input)
    skills = _extract_skills(parsed_text, raw_input)
    experience_summary = _extract_experience_summary(parsed_text, raw_input)
    
    return {
        "name": name,
        "email": email,
        "phone": phone,
        "skills": skills,
        "experience_summary": experience_summary
    }


def _is_valid_candidate_schema(data: Dict[str, Any]) -> bool:
    """Check if input already matches the expected candidate schema"""
    required_fields = {"name", "email", "phone", "skills", "experience_summary"}
    return all(field in data for field in required_fields)


def _extract_name(parsed_text: str, raw_input: Dict[str, Any] = None) -> str:
    """Extract candidate name using multiple heuristics"""
    
    # Check for explicit name fields first
    if raw_input:
        name_fields = ["name", "full_name", "candidate_name", "applicant_name"]
        for field in name_fields:
            if field in raw_input and raw_input[field]:
                return str(raw_input[field]).strip()
    
    # Look for name patterns in text (usually at the beginning)
    lines = parsed_text.split('\n')
    
    # Check first few lines for name patterns
    for line in lines[:5]:
        line = line.strip()
        if not line:
            continue
        
        # Pattern: First line that looks like a name (2-4 words, title case)
        name_pattern = r"^([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})$"
        match = re.search(name_pattern, line)
        if match:
            name = match.group(1).strip()
            # Exclude common non-name words
            if not any(word in name.lower() for word in ['resume', 'cv', 'curriculum', 'vitae', 'experience', 'skills']):
                return name
        
        # Pattern: Name followed by title/position
        name_title_pattern = r"^([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})\s*[-|]\s*[A-Z]"
        match = re.search(name_title_pattern, line)
        if match:
            return match.group(1).strip()
    
    # Look for "Name:" pattern
    name_label_pattern = r"(?:Name|Full Name|Candidate)[:.]?\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})"
    match = re.search(name_label_pattern, parsed_text, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    
    return ""


def _extract_email(parsed_text: str, raw_input: Dict[str, Any] = None) -> str:
    """Extract email address using regex"""
    
    # Check for explicit email fields first
    if raw_input:
        email_fields = ["email", "email_address", "contact_email", "e_mail"]
        for field in email_fields:
            if field in raw_input and raw_input[field]:
                email = str(raw_input[field]).strip()
                if _is_valid_email(email):
                    return email
    
    # Look for email patterns in text
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    emails = re.findall(email_pattern, parsed_text)
    
    if emails:
        # Return the first valid email found
        for email in emails:
            if _is_valid_email(email):
                return email
    
    return ""


def _extract_phone(parsed_text: str, raw_input: Dict[str, Any] = None) -> str:
    """Extract phone number using regex"""
    
    # Check for explicit phone fields first
    if raw_input:
        phone_fields = ["phone", "phone_number", "mobile", "telephone", "contact_number"]
        for field in phone_fields:
            if field in raw_input and raw_input[field]:
                phone = str(raw_input[field]).strip()
                if _is_valid_phone(phone):
                    return phone
    
    # Look for phone patterns in text
    phone_patterns = [
        r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',  # 123-456-7890, 123.456.7890, 1234567890
        r'\(\d{3}\)\s*\d{3}[-.]?\d{4}',    # (123) 456-7890
        r'\+\d{1,3}[-.\s]?\d{3,4}[-.\s]?\d{3,4}[-.\s]?\d{3,4}',  # International format
        r'\b\d{3}\s+\d{3}\s+\d{4}\b'       # 123 456 7890
    ]
    
    for pattern in phone_patterns:
        matches = re.findall(pattern, parsed_text)
        if matches:
            # Return the first valid phone number found
            for phone in matches:
                if _is_valid_phone(phone):
                    return phone
    
    return ""


def _extract_skills(parsed_text: str, raw_input: Dict[str, Any] = None) -> List[str]:
    """Extract skills using keyword detection and section analysis"""
    
    # Check for explicit skills fields first
    if raw_input:
        skills_fields = ["skills", "technical_skills", "competencies", "expertise"]
        for field in skills_fields:
            if field in raw_input and raw_input[field]:
                skills = raw_input[field]
                if isinstance(skills, list):
                    return [str(skill).strip() for skill in skills if str(skill).strip()]
                elif isinstance(skills, str):
                    return [skill.strip() for skill in skills.split(',') if skill.strip()]
    
    # Common technical skills database
    technical_skills = {
        'programming': [
            'Python', 'Java', 'JavaScript', 'TypeScript', 'C++', 'C#', 'Go', 'Rust',
            'PHP', 'Ruby', 'Swift', 'Kotlin', 'Scala', 'R', 'MATLAB', 'Perl', 'Shell'
        ],
        'web_development': [
            'HTML', 'CSS', 'React', 'Angular', 'Vue.js', 'Node.js', 'Django', 'Flask',
            'Express.js', 'Spring', 'ASP.NET', 'Laravel', 'Rails', 'jQuery', 'Bootstrap'
        ],
        'databases': [
            'SQL', 'MySQL', 'PostgreSQL', 'MongoDB', 'Redis', 'Oracle', 'SQLite',
            'Cassandra', 'DynamoDB', 'Elasticsearch', 'Neo4j'
        ],
        'cloud_platforms': [
            'AWS', 'Azure', 'Google Cloud', 'Docker', 'Kubernetes', 'Terraform',
            'Jenkins', 'CI/CD', 'DevOps'
        ],
        'data_science': [
            'Machine Learning', 'Deep Learning', 'TensorFlow', 'PyTorch', 'Scikit-learn',
            'Pandas', 'NumPy', 'Data Analysis', 'Statistics', 'R', 'Tableau', 'Power BI'
        ],
        'mobile_development': [
            'iOS', 'Android', 'React Native', 'Flutter', 'Xamarin', 'Swift', 'Kotlin'
        ],
        'other_technical': [
            'Git', 'Linux', 'Windows', 'macOS', 'REST API', 'GraphQL', 'Microservices',
            'Agile', 'Scrum', 'JIRA', 'Confluence'
        ]
    }
    
    # Flatten all skills into one list
    all_skills = []
    for category, skills in technical_skills.items():
        all_skills.extend(skills)
    
    # Look for skills in text
    found_skills = []
    text_lower = parsed_text.lower()
    
    for skill in all_skills:
        # Check for exact matches (case insensitive)
        if skill.lower() in text_lower:
            found_skills.append(skill)
        # Check for variations
        elif _skill_variation_match(skill, text_lower):
            found_skills.append(skill)
    
    # Look for skills sections
    skills_sections = [
        r'(?:Skills|Technical Skills|Competencies|Expertise)[:.]?\s*(.+?)(?=\n\s*(?:Experience|Education|Projects|$))',
        r'(?:Programming Languages|Technologies)[:.]?\s*(.+?)(?=\n\s*(?:Experience|Education|Projects|$))'
    ]
    
    for pattern in skills_sections:
        match = re.search(pattern, parsed_text, re.IGNORECASE | re.DOTALL)
        if match:
            skills_text = match.group(1)
            # Extract skills from the section
            section_skills = _extract_skills_from_section(skills_text, all_skills)
            found_skills.extend(section_skills)
    
    # Remove duplicates and return
    return list(set(found_skills))


def _extract_experience_summary(parsed_text: str, raw_input: Dict[str, Any] = None) -> str:
    """Extract experience summary from resume text"""
    
    # Check for explicit experience fields first
    if raw_input:
        exp_fields = ["experience", "experience_summary", "work_history", "career_summary"]
        for field in exp_fields:
            if field in raw_input and raw_input[field]:
                return str(raw_input[field]).strip()
    
    # Look for experience sections
    exp_sections = [
        r'(?:Experience|Work History|Career|Professional Experience)[:.]?\s*(.+?)(?=\n\s*(?:Education|Skills|Projects|$))',
        r'(?:Employment|Work Experience)[:.]?\s*(.+?)(?=\n\s*(?:Education|Skills|Projects|$))'
    ]
    
    for pattern in exp_sections:
        match = re.search(pattern, parsed_text, re.IGNORECASE | re.DOTALL)
        if match:
            exp_text = match.group(1).strip()
            if len(exp_text) > 50:  # Reasonable experience summary length
                return exp_text
    
    # Look for job titles and companies (experience indicators)
    experience_indicators = []
    lines = parsed_text.split('\n')
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # Look for job title patterns
        if any(title in line.lower() for title in ['engineer', 'developer', 'analyst', 'manager', 'director', 'specialist', 'coordinator']):
            experience_indicators.append(line)
        # Look for company patterns
        elif any(company in line.lower() for company in ['inc', 'corp', 'ltd', 'llc', 'company', 'technologies', 'solutions']):
            experience_indicators.append(line)
    
    if experience_indicators:
        return "\n".join(experience_indicators[:5])  # Limit to 5 items
    
    return ""


def _is_valid_email(email: str) -> bool:
    """Validate email format"""
    email_pattern = r'^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}$'
    return bool(re.match(email_pattern, email))


def _is_valid_phone(phone: str) -> bool:
    """Validate phone number format"""
    # Remove all non-digit characters for validation
    digits_only = re.sub(r'\D', '', phone)
    return len(digits_only) >= 10 and len(digits_only) <= 15


def _skill_variation_match(skill: str, text: str) -> bool:
    """Check for skill variations and abbreviations"""
    skill_lower = skill.lower()
    
    # Common variations
    variations = {
        'javascript': ['js', 'ecmascript'],
        'machine learning': ['ml', 'machine learning', 'ai'],
        'artificial intelligence': ['ai', 'artificial intelligence'],
        'data analysis': ['data analysis', 'analytics'],
        'web development': ['web dev', 'frontend', 'backend'],
        'mobile development': ['mobile dev', 'ios', 'android'],
        'cloud computing': ['cloud', 'aws', 'azure', 'gcp']
    }
    
    if skill_lower in variations:
        return any(var in text for var in variations[skill_lower])
    
    return False


def _extract_skills_from_section(skills_text: str, all_skills: List[str]) -> List[str]:
    """Extract skills from a skills section text"""
    found_skills = []
    skills_lower = skills_text.lower()
    
    for skill in all_skills:
        if skill.lower() in skills_lower:
            found_skills.append(skill)
    
    return found_skills


def format_candidate_data(candidate_data: Dict[str, Any]) -> str:
    """
    Format normalized candidate data into a readable summary
    
    Args:
        candidate_data: Normalized candidate data
        
    Returns:
        Formatted candidate summary string
    """
    sections = []
    
    if candidate_data.get("name"):
        sections.append(f"**Name:** {candidate_data['name']}")
    
    if candidate_data.get("email"):
        sections.append(f"**Email:** {candidate_data['email']}")
    
    if candidate_data.get("phone"):
        sections.append(f"**Phone:** {candidate_data['phone']}")
    
    if candidate_data.get("skills"):
        skills_str = ", ".join(candidate_data['skills'])
        sections.append(f"**Skills:** {skills_str}")
    
    if candidate_data.get("experience_summary"):
        sections.append(f"**Experience:** {candidate_data['experience_summary']}")
    
    return "\n\n".join(sections)


# Test function
def test_candidate_formatter():
    """Test the candidate formatter with various inputs"""
    
    # Test case 1: Parsed resume text
    resume_text = """
    John Smith
    Senior Software Engineer
    john.smith@email.com
    (555) 123-4567
    
    Skills:
    Python, JavaScript, React, Node.js, AWS, Docker
    
    Experience:
    Senior Software Engineer at TechCorp Inc. (2020-2023)
    - Led development of microservices architecture
    - Technologies: Python, Django, PostgreSQL, AWS
    
    Software Engineer at StartupXYZ (2018-2020)
    - Developed web applications using React and Node.js
    - Implemented CI/CD pipelines
    """
    
    # Test case 2: Inconsistent field names
    inconsistent_input = {
        "candidate_name": "Jane Doe",
        "email_address": "jane.doe@example.com",
        "mobile_number": "+1-555-987-6543",
        "technical_skills": ["Python", "SQL", "Data Analysis"],
        "work_history": "5+ years in data science"
    }
    
    # Test case 3: Already normalized
    normalized_input = {
        "name": "Bob Johnson",
        "email": "bob.johnson@company.com",
        "phone": "555-555-5555",
        "skills": ["Java", "Spring", "MySQL"],
        "experience_summary": "Senior Java Developer with 8 years experience"
    }
    
    print("🧪 Testing Candidate Formatter")
    print("=" * 50)
    
    # Test parsed text
    print("\n1. Testing parsed resume text:")
    result1 = normalize_candidate_input(resume_text)
    print(f"   Name: {result1['name']}")
    print(f"   Email: {result1['email']}")
    print(f"   Phone: {result1['phone']}")
    print(f"   Skills: {result1['skills']}")
    print(f"   Experience: {result1['experience_summary'][:100]}...")
    
    # Test inconsistent fields
    print("\n2. Testing inconsistent field names:")
    result2 = normalize_candidate_input("", inconsistent_input)
    print(f"   Name: {result2['name']}")
    print(f"   Email: {result2['email']}")
    print(f"   Phone: {result2['phone']}")
    print(f"   Skills: {result2['skills']}")
    print(f"   Experience: {result2['experience_summary']}")
    
    # Test normalized input
    print("\n3. Testing already normalized input:")
    result3 = normalize_candidate_input("", normalized_input)
    print(f"   Name: {result3['name']}")
    print(f"   Email: {result3['email']}")
    print(f"   Phone: {result3['phone']}")
    print(f"   Skills: {result3['skills']}")
    print(f"   Experience: {result3['experience_summary']}")
    
    print("\n✅ Candidate Formatter tests completed!")
    return True


if __name__ == "__main__":
    test_candidate_formatter()
