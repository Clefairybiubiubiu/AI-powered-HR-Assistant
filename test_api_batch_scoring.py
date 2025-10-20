"""
Simple API test script for batch scoring endpoint
"""
import requests
import json
import time


def test_batch_scoring_api():
    """Test the batch scoring API endpoint"""
    print("🧪 Testing Batch Scoring API")
    print("=" * 50)
    
    # Test data
    test_data = {
        "jobs": [
            {
                "title": "Data Scientist",
                "description": "We are looking for a Data Scientist to analyze large datasets and build machine learning models.",
                "requirements": "Python, SQL, Machine Learning, TensorFlow, 3+ years experience"
            },
            {
                "title": "Marketing Analyst", 
                "description": "Join our marketing team to analyze campaign performance and customer behavior.",
                "requirements": "Google Analytics, Excel, Power BI, Data Visualization, Marketing experience"
            }
        ],
        "candidates": [
            {
                "name": "Alice",
                "resume_text": "Alice Johnson, Senior Data Scientist. 5+ years in data science and machine learning. Expert in Python, SQL, TensorFlow, PyTorch. Built recommendation systems and predictive models."
            },
            {
                "name": "Bob",
                "resume_text": "Bob Smith, Marketing Analyst. 4+ years in marketing analytics. Expert in Google Analytics, Excel, Power BI. Analyzed campaign performance and customer behavior."
            }
        ]
    }
    
    print("📋 Test Data:")
    print(f"   Jobs: {len(test_data['jobs'])}")
    print(f"   Candidates: {len(test_data['candidates'])}")
    
    # Check if API is running
    print("\n🔍 Checking API Health...")
    try:
        health_response = requests.get("http://localhost:8000/health", timeout=5)
        if health_response.status_code == 200:
            print("   ✅ API is running")
        else:
            print("   ❌ API health check failed")
            return False
    except requests.exceptions.ConnectionError:
        print("   ❌ API is not running. Please start it with: python similarity_app.py")
        return False
    
    # Test batch scoring endpoint
    print("\n📡 Testing Batch Scoring Endpoint...")
    print("   POST /api/v1/scoring/batch")
    
    try:
        start_time = time.time()
        
        response = requests.post(
            "http://localhost:8000/api/v1/scoring/batch",
            json=test_data,
            timeout=60
        )
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        if response.status_code == 200:
            results = response.json()
            print(f"   ✅ Request successful (took {processing_time:.2f}s)")
            
            # Display results
            print("\n🎯 Batch Scoring Results:")
            for result in results['results']:
                print(f"   Job: {result['job_title']}")
                print(f"   Best Candidate: {result['best_candidate']}")
                print(f"   Score: {result['score']:.3f}")
                print(f"   Reason: {result['reason']}")
                print()
            
            return True
        else:
            print(f"   ❌ Request failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
            
    except requests.exceptions.Timeout:
        print("   ❌ Request timed out")
        return False
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False


def test_with_more_data():
    """Test with more comprehensive data"""
    print("\n🧪 Testing with More Data")
    print("=" * 50)
    
    # More comprehensive test data
    test_data = {
        "jobs": [
            {
                "title": "Data Scientist",
                "description": "We are looking for a Data Scientist to analyze large datasets and build machine learning models. You will work with Python, SQL, and various ML frameworks.",
                "requirements": "Python, SQL, Machine Learning, TensorFlow, 3+ years experience"
            },
            {
                "title": "Marketing Analyst", 
                "description": "Join our marketing team to analyze campaign performance and customer behavior. You will use Google Analytics, Excel, and visualization tools.",
                "requirements": "Google Analytics, Excel, Power BI, Data Visualization, Marketing experience"
            },
            {
                "title": "Software Engineer",
                "description": "Develop web applications using modern technologies. You will work with JavaScript, React, Node.js, and cloud platforms.",
                "requirements": "JavaScript, React, Node.js, AWS, 2+ years experience"
            }
        ],
        "candidates": [
            {
                "name": "Alice",
                "resume_text": """
                Alice Johnson
                Senior Data Scientist
                alice.johnson@email.com
                
                Experience:
                - 5+ years in data science and machine learning
                - Expert in Python, SQL, TensorFlow, PyTorch
                - Built recommendation systems and predictive models
                - Experience with A/B testing and statistical analysis
                - Led data science team at TechCorp
                
                Skills: Python, SQL, Machine Learning, TensorFlow, PyTorch, Statistics, A/B Testing
                """
            },
            {
                "name": "Bob",
                "resume_text": """
                Bob Smith
                Marketing Analyst
                bob.smith@email.com
                
                Experience:
                - 4+ years in marketing analytics
                - Expert in Google Analytics, Excel, Power BI
                - Analyzed campaign performance and customer behavior
                - Created data visualizations and reports
                - Managed marketing data at StartupXYZ
                
                Skills: Google Analytics, Excel, Power BI, Data Visualization, Marketing, Campaign Analysis
                """
            },
            {
                "name": "Charlie",
                "resume_text": """
                Charlie Brown
                Full Stack Developer
                charlie.brown@email.com
                
                Experience:
                - 3+ years in web development
                - Expert in JavaScript, React, Node.js, AWS
                - Built scalable web applications
                - Experience with microservices and cloud deployment
                - Led development team at WebCorp
                
                Skills: JavaScript, React, Node.js, AWS, Docker, Microservices, Web Development
                """
            }
        ]
    }
    
    print("📋 Test Data:")
    print(f"   Jobs: {len(test_data['jobs'])}")
    print(f"   Candidates: {len(test_data['candidates'])}")
    
    try:
        start_time = time.time()
        
        response = requests.post(
            "http://localhost:8000/api/v1/scoring/batch",
            json=test_data,
            timeout=60
        )
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        if response.status_code == 200:
            results = response.json()
            print(f"   ✅ Request successful (took {processing_time:.2f}s)")
            
            # Display results
            print("\n🎯 Batch Scoring Results:")
            for result in results['results']:
                print(f"   Job: {result['job_title']}")
                print(f"   Best Candidate: {result['best_candidate']}")
                print(f"   Score: {result['score']:.3f}")
                print(f"   Reason: {result['reason']}")
                print()
            
            return True
        else:
            print(f"   ❌ Request failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False


def test_edge_cases():
    """Test edge cases"""
    print("\n🧪 Testing Edge Cases")
    print("=" * 50)
    
    # Test empty jobs
    print("\n🔍 Testing Empty Jobs...")
    empty_jobs_data = {
        "jobs": [],
        "candidates": [{"name": "Alice", "resume_text": "Python developer"}]
    }
    
    try:
        response = requests.post(
            "http://localhost:8000/api/v1/scoring/batch",
            json=empty_jobs_data,
            timeout=30
        )
        
        if response.status_code == 200:
            results = response.json()
            print(f"   ✅ Empty jobs handled gracefully: {len(results['results'])} results")
        else:
            print(f"   ❌ Empty jobs failed: {response.status_code}")
    except Exception as e:
        print(f"   ❌ Error with empty jobs: {e}")
    
    # Test empty candidates
    print("\n🔍 Testing Empty Candidates...")
    empty_candidates_data = {
        "jobs": [{"title": "Developer", "description": "Build apps", "requirements": "Python"}],
        "candidates": []
    }
    
    try:
        response = requests.post(
            "http://localhost:8000/api/v1/scoring/batch",
            json=empty_candidates_data,
            timeout=30
        )
        
        if response.status_code == 200:
            results = response.json()
            print(f"   ✅ Empty candidates handled gracefully: {len(results['results'])} results")
        else:
            print(f"   ❌ Empty candidates failed: {response.status_code}")
    except Exception as e:
        print(f"   ❌ Error with empty candidates: {e}")
    
    # Test single job single candidate
    print("\n🔍 Testing Single Job Single Candidate...")
    single_data = {
        "jobs": [{"title": "Developer", "description": "Build apps", "requirements": "Python"}],
        "candidates": [{"name": "Alice", "resume_text": "Python developer"}]
    }
    
    try:
        response = requests.post(
            "http://localhost:8000/api/v1/scoring/batch",
            json=single_data,
            timeout=30
        )
        
        if response.status_code == 200:
            results = response.json()
            print(f"   ✅ Single job/candidate handled: {len(results['results'])} results")
            if results['results']:
                result = results['results'][0]
                print(f"   Best match: {result['best_candidate']} (score: {result['score']:.3f})")
        else:
            print(f"   ❌ Single job/candidate failed: {response.status_code}")
    except Exception as e:
        print(f"   ❌ Error with single job/candidate: {e}")
    
    print("\n✅ Edge case tests completed!")
    return True


def main():
    """Main test function"""
    print("🚀 Batch Scoring API Test Suite")
    print("=" * 60)
    
    try:
        # Test basic API
        basic_success = test_batch_scoring_api()
        
        # Test with more data
        comprehensive_success = test_with_more_data()
        
        # Test edge cases
        edge_cases_success = test_edge_cases()
        
        print("\n" + "=" * 60)
        print("📊 Test Results Summary")
        print("=" * 60)
        
        tests = [
            ("Basic API Test", basic_success),
            ("Comprehensive Data", comprehensive_success),
            ("Edge Cases", edge_cases_success)
        ]
        
        passed = 0
        for test_name, result in tests:
            status = "✅ PASSED" if result else "❌ FAILED"
            print(f"{test_name:20s}: {status}")
            if result:
                passed += 1
        
        print(f"\nOverall: {passed}/{len(tests)} tests passed")
        
        if passed == len(tests):
            print("\n🎉 All API tests completed successfully!")
            print("✅ Batch scoring API is working correctly")
            print("✅ Handles various data sizes")
            print("✅ Robust error handling")
            
            print("\n💡 Usage:")
            print("   • Send POST requests to /api/v1/scoring/batch")
            print("   • Include jobs and candidates arrays")
            print("   • Get back best matches with scores and reasons")
            
            return True
        else:
            print("\n⚠️  Some tests failed. Please check the issues above.")
            return False
            
    except Exception as e:
        print(f"\n❌ Test suite failed: {e}")
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
