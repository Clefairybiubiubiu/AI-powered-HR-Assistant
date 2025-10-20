"""
Test script for Streamlit-FastAPI integration
"""
import requests
import json
import time
import subprocess
import sys
import os
from typing import Dict, Any

def test_api_connection():
    """Test if the FastAPI backend is running"""
    print("🧪 Testing API Connection...")
    
    try:
        # Test health endpoint
        health_response = requests.get("http://localhost:8000/health", timeout=5)
        if health_response.status_code == 200:
            print("✅ API Health Check: PASSED")
            return True
        else:
            print(f"❌ API Health Check: FAILED ({health_response.status_code})")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ API Connection: FAILED (Server not running)")
        return False
    except Exception as e:
        print(f"❌ API Connection: FAILED ({e})")
        return False

def test_similarity_endpoint():
    """Test the similarity endpoint with sample data"""
    print("\n🧪 Testing Similarity Endpoint...")
    
    sample_data = {
        "resume": """
        John Smith
        Senior Software Engineer
        Email: john.smith@email.com
        
        SUMMARY
        Experienced software engineer with 6+ years of experience in Python development, 
        microservices architecture, and cloud technologies.
        
        TECHNICAL SKILLS
        Programming Languages: Python, JavaScript, TypeScript, SQL
        Frameworks: Django, Flask, React, Node.js
        Databases: PostgreSQL, MongoDB, Redis
        Cloud Platforms: AWS, Azure, Docker, Kubernetes
        
        EXPERIENCE
        Senior Software Engineer | TechCorp Inc. | 2020 - Present
        - Led development of microservices architecture
        - Technologies: Python, Django, PostgreSQL, AWS
        
        EDUCATION
        Bachelor of Science in Computer Science
        University of California, Berkeley | 2016
        """,
        "job_desc": """
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
    }
    
    try:
        response = requests.post(
            "http://localhost:8000/similarity",
            json=sample_data,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Similarity Endpoint: PASSED")
            print(f"   Overall Score: {result.get('similarity_score', 0):.3f}")
            print(f"   Skill Match: {result.get('skill_match', 0):.3f}")
            print(f"   Experience Alignment: {result.get('experience_alignment', 0):.3f}")
            print(f"   Education Match: {result.get('education_match', 0):.3f}")
            return True
        else:
            print(f"❌ Similarity Endpoint: FAILED ({response.status_code})")
            print(f"   Response: {response.text}")
            return False
            
    except requests.exceptions.Timeout:
        print("❌ Similarity Endpoint: TIMEOUT (Request took too long)")
        return False
    except Exception as e:
        print(f"❌ Similarity Endpoint: FAILED ({e})")
        return False

def test_streamlit_import():
    """Test if Streamlit app can be imported"""
    print("\n🧪 Testing Streamlit App Import...")
    
    try:
        # Add frontend directory to path
        frontend_dir = os.path.join(os.path.dirname(__file__), 'frontend')
        sys.path.insert(0, frontend_dir)
        
        # Try to import the app
        import app
        print("✅ Streamlit App Import: PASSED")
        return True
        
    except ImportError as e:
        print(f"❌ Streamlit App Import: FAILED ({e})")
        return False
    except Exception as e:
        print(f"❌ Streamlit App Import: FAILED ({e})")
        return False

def test_streamlit_functionality():
    """Test Streamlit app functionality"""
    print("\n🧪 Testing Streamlit App Functionality...")
    
    try:
        # Test the functions from the app
        from app import check_api_health, send_similarity_request
        
        # Test API health check
        health_result = check_api_health()
        print(f"   API Health Check: {'PASSED' if health_result else 'FAILED'}")
        
        # Test similarity request
        sample_resume = "John Smith\nPython Developer\n5 years experience"
        sample_job = "Python Developer\n3+ years experience required"
        
        result = send_similarity_request(sample_resume, sample_job)
        if result:
            print("✅ Streamlit App Functionality: PASSED")
            print(f"   Similarity Score: {result.get('similarity_score', 0):.3f}")
            return True
        else:
            print("❌ Streamlit App Functionality: FAILED (No result returned)")
            return False
            
    except Exception as e:
        print(f"❌ Streamlit App Functionality: FAILED ({e})")
        return False

def run_comprehensive_test():
    """Run comprehensive integration test"""
    print("🚀 HR Assistant Streamlit-FastAPI Integration Test")
    print("=" * 60)
    
    # Test results
    tests = []
    
    # Test 1: API Connection
    tests.append(("API Connection", test_api_connection()))
    
    # Test 2: Similarity Endpoint
    tests.append(("Similarity Endpoint", test_similarity_endpoint()))
    
    # Test 3: Streamlit Import
    tests.append(("Streamlit Import", test_streamlit_import()))
    
    # Test 4: Streamlit Functionality
    tests.append(("Streamlit Functionality", test_streamlit_functionality()))
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 Test Results Summary")
    print("=" * 60)
    
    passed = 0
    total = len(tests)
    
    for test_name, result in tests:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name:20s}: {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The integration is working correctly.")
        print("\n📋 Next Steps:")
        print("1. Start the FastAPI backend: python run_similarity_app.py")
        print("2. Start the Streamlit frontend: python run_streamlit_app.py")
        print("3. Open http://localhost:8501 in your browser")
    else:
        print("⚠️  Some tests failed. Please check the issues above.")
        print("\n🔧 Troubleshooting:")
        print("1. Make sure the FastAPI backend is running on port 8000")
        print("2. Check that all dependencies are installed")
        print("3. Verify the API endpoints are working correctly")
    
    return passed == total

if __name__ == "__main__":
    success = run_comprehensive_test()
    sys.exit(0 if success else 1)
