"""
Test script for the similarity API
"""
import requests
import json

# API endpoint
API_URL = "http://localhost:8000/similarity"

# Sample data
sample_resume = """
John Smith
Senior Software Engineer
Email: john.smith@email.com

SUMMARY
Experienced software engineer with 6+ years of experience in Python development, 
microservices architecture, and cloud technologies. Strong background in building 
scalable web applications and leading development teams.

TECHNICAL SKILLS
Programming Languages: Python, JavaScript, TypeScript, SQL
Frameworks: Django, Flask, React, Node.js
Databases: PostgreSQL, MongoDB, Redis
Cloud Platforms: AWS, Azure, Docker, Kubernetes
Tools: Git, Jenkins, Jira

EXPERIENCE
Senior Software Engineer | TechCorp Inc. | 2020 - Present
- Led development of microservices architecture serving 1M+ users
- Implemented CI/CD pipelines using Jenkins and Docker
- Technologies: Python, Django, PostgreSQL, AWS, Kubernetes

Software Engineer | StartupXYZ | 2018 - 2020
- Developed REST APIs using Flask and FastAPI
- Built data processing pipelines using Python and pandas
- Technologies: Python, Flask, MongoDB, Docker

EDUCATION
Bachelor of Science in Computer Science
University of California, Berkeley | 2016
"""

sample_job_desc = """
Senior Python Developer

We are looking for a Senior Python Developer to join our team. You will be responsible 
for developing and maintaining our backend services, working with microservices architecture, 
and collaborating with cross-functional teams.

Requirements:
- 5+ years of Python experience
- Experience with Django/Flask frameworks
- Knowledge of REST APIs
- Experience with databases (PostgreSQL, MongoDB)
- Experience with cloud platforms (AWS, Azure)
- Experience with Docker and Kubernetes
- Strong problem-solving skills
- Excellent communication skills

Education: Bachelor's degree in Computer Science or related field preferred
"""

def test_similarity_api():
    """Test the similarity API endpoint"""
    
    # Prepare request data
    request_data = {
        "resume": sample_resume,
        "job_desc": sample_job_desc
    }
    
    print("🧪 Testing Similarity API...")
    print("=" * 50)
    
    try:
        # Make API request
        response = requests.post(API_URL, json=request_data)
        
        if response.status_code == 200:
            result = response.json()
            
            print("✅ API Response:")
            print(f"📊 Overall Similarity Score: {result['similarity_score']:.3f}")
            print(f"🎯 Skill Match: {result['skill_match']:.3f}")
            print(f"💼 Experience Alignment: {result['experience_alignment']:.3f}")
            print(f"🎓 Education Match: {result['education_match']:.3f}")
            
            print("\n📋 Details:")
            details = result['details']
            print(f"🧠 Semantic Similarity: {details['semantic_similarity']:.3f}")
            
            print(f"\n🔧 Resume Skills Found: {len(details['resume_skills'])}")
            print(f"   {', '.join(details['resume_skills'][:10])}{'...' if len(details['resume_skills']) > 10 else ''}")
            
            print(f"\n💼 Job Skills Required: {len(details['job_skills'])}")
            print(f"   {', '.join(details['job_skills'][:10])}{'...' if len(details['job_skills']) > 10 else ''}")
            
            print(f"\n✅ Matched Skills: {len(details['matched_skills'])}")
            print(f"   {', '.join(details['matched_skills'])}")
            
            print(f"\n❌ Missing Skills: {len(details['missing_skills'])}")
            print(f"   {', '.join(details['missing_skills'])}")
            
            print(f"\n⚖️ Scoring Weights:")
            weights = details['weights']
            print(f"   α (skill_match): {weights['alpha']}")
            print(f"   β (experience_alignment): {weights['beta']}")
            print(f"   γ (education_match): {weights['gamma']}")
            
            # Calculate weighted score manually for verification
            manual_score = (weights['alpha'] * result['skill_match'] + 
                          weights['beta'] * result['experience_alignment'] + 
                          weights['gamma'] * result['education_match'])
            print(f"\n🧮 Manual Calculation: {manual_score:.3f}")
            
        else:
            print(f"❌ API Error: {response.status_code}")
            print(f"Response: {response.text}")
            
    except requests.exceptions.ConnectionError:
        print("❌ Connection Error: Make sure the API server is running on http://localhost:8000")
        print("   Start the server with: python similarity_app.py")
    except Exception as e:
        print(f"❌ Error: {e}")


def test_health_endpoint():
    """Test the health endpoint"""
    try:
        response = requests.get("http://localhost:8000/health")
        if response.status_code == 200:
            print("✅ Health Check: API is running")
            print(f"   Model: {response.json()['model']}")
        else:
            print(f"❌ Health Check Failed: {response.status_code}")
    except requests.exceptions.ConnectionError:
        print("❌ Health Check: API server not running")


if __name__ == "__main__":
    print("🚀 HR Assistant Similarity API Test")
    print("=" * 50)
    
    # Test health first
    test_health_endpoint()
    print()
    
    # Test similarity endpoint
    test_similarity_api()
