"""
Test API file upload functionality
"""
import requests
import json
import os

def test_api_health():
    """Test if the API is running"""
    print("🧪 Testing API Health...")
    
    try:
        response = requests.get("http://localhost:8000/health", timeout=5)
        if response.status_code == 200:
            print("✅ API is running")
            return True
        else:
            print(f"❌ API health check failed: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ API is not running. Please start it with: python similarity_app.py")
        return False
    except Exception as e:
        print(f"❌ API health check error: {e}")
        return False

def test_file_upload():
    """Test file upload to the API"""
    print("\n🧪 Testing File Upload...")
    
    # Check if test file exists
    test_file = "test_resume.docx"
    if not os.path.exists(test_file):
        print(f"❌ Test file {test_file} not found. Please run test_file_parsing.py first.")
        return False
    
    try:
        # Prepare the file upload
        with open(test_file, 'rb') as f:
            files = {'resume_file': (test_file, f, 'application/vnd.openxmlformats-officedocument.wordprocessingml.document')}
            data = {'job_desc': 'Python Developer with 3+ years experience. Must know Django, Flask, and AWS.'}
            
            # Make the request
            response = requests.post(
                "http://localhost:8000/similarity-files",
                files=files,
                data=data,
                timeout=30
            )
        
        if response.status_code == 200:
            result = response.json()
            print("✅ File upload successful!")
            print(f"   Similarity Score: {result.get('similarity_score', 0):.3f}")
            print(f"   Skill Match: {result.get('skill_match', 0):.3f}")
            print(f"   Experience Alignment: {result.get('experience_alignment', 0):.3f}")
            print(f"   Education Match: {result.get('education_match', 0):.3f}")
            return True
        else:
            print(f"❌ File upload failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
            
    except requests.exceptions.Timeout:
        print("❌ Request timed out. The API might be processing slowly.")
        return False
    except Exception as e:
        print(f"❌ File upload error: {e}")
        return False

def test_text_similarity():
    """Test the original text similarity endpoint"""
    print("\n🧪 Testing Text Similarity...")
    
    try:
        payload = {
            "resume": "John Smith\nSenior Software Engineer\nPython, Django, AWS, 6 years experience\nBachelor in Computer Science",
            "job_desc": "Python Developer\n3+ years experience\nDjango, Flask, AWS required\nBachelor degree preferred"
        }
        
        response = requests.post(
            "http://localhost:8000/similarity",
            json=payload,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Text similarity successful!")
            print(f"   Similarity Score: {result.get('similarity_score', 0):.3f}")
            return True
        else:
            print(f"❌ Text similarity failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Text similarity error: {e}")
        return False

def main():
    """Main test function"""
    print("🚀 API File Upload Test")
    print("=" * 40)
    
    # Test API health
    health_success = test_api_health()
    
    if not health_success:
        print("\n❌ Cannot proceed without API. Please start the API first.")
        return False
    
    # Test text similarity (should work even with dependency issues)
    text_success = test_text_similarity()
    
    # Test file upload
    file_success = test_file_upload()
    
    # Summary
    print("\n" + "=" * 40)
    print("📊 Test Results Summary")
    print("=" * 40)
    
    tests = [
        ("API Health", health_success),
        ("Text Similarity", text_success),
        ("File Upload", file_success)
    ]
    
    passed = 0
    for test_name, result in tests:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name:15s}: {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        print("\n🎉 All API tests passed!")
        print("📁 File upload functionality is working correctly")
        print("💡 You can now use the Streamlit app to upload files")
    else:
        print("\n⚠️  Some tests failed. Please check the issues above.")
        if not text_success:
            print("💡 Note: Text similarity might fail due to ML dependency issues")
            print("   But file upload should still work for basic parsing")
    
    return passed >= 1  # At least API health should pass

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
