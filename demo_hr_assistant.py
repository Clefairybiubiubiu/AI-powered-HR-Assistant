"""
Demo script for HR Assistant - Complete System
"""
import subprocess
import sys
import os
import time
import requests
import webbrowser
from threading import Thread

def check_port(port):
    """Check if a port is in use"""
    import socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    result = sock.connect_ex(('localhost', port))
    sock.close()
    return result == 0

def start_fastapi_backend():
    """Start the FastAPI backend"""
    print("🚀 Starting FastAPI Backend...")
    
    # Check if port 8000 is already in use
    if check_port(8000):
        print("⚠️  Port 8000 is already in use. Backend might already be running.")
        return True
    
    try:
        # Start the similarity app
        process = subprocess.Popen([
            sys.executable, 'similarity_app.py'
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Wait a moment for startup
        time.sleep(3)
        
        # Check if it's running
        if process.poll() is None:
            print("✅ FastAPI Backend started successfully")
            return True
        else:
            print("❌ Failed to start FastAPI Backend")
            return False
            
    except Exception as e:
        print(f"❌ Error starting FastAPI Backend: {e}")
        return False

def start_streamlit_frontend():
    """Start the Streamlit frontend"""
    print("🎨 Starting Streamlit Frontend...")
    
    # Check if port 8501 is already in use
    if check_port(8501):
        print("⚠️  Port 8501 is already in use. Frontend might already be running.")
        return True
    
    try:
        # Start the streamlit app
        process = subprocess.Popen([
            sys.executable, '-m', 'streamlit', 'run', 'frontend/app.py',
            '--server.port', '8501',
            '--server.address', '0.0.0.0',
            '--server.headless', 'true'
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Wait a moment for startup
        time.sleep(5)
        
        # Check if it's running
        if process.poll() is None:
            print("✅ Streamlit Frontend started successfully")
            return True
        else:
            print("❌ Failed to start Streamlit Frontend")
            return False
            
    except Exception as e:
        print(f"❌ Error starting Streamlit Frontend: {e}")
        return False

def test_api_connection():
    """Test if the API is working"""
    print("🧪 Testing API Connection...")
    
    max_retries = 10
    for i in range(max_retries):
        try:
            response = requests.get("http://localhost:8000/health", timeout=5)
            if response.status_code == 200:
                print("✅ API is responding")
                return True
        except requests.exceptions.RequestException:
            pass
        
        print(f"   Attempt {i+1}/{max_retries}...")
        time.sleep(2)
    
    print("❌ API is not responding")
    return False

def test_similarity_endpoint():
    """Test the similarity endpoint"""
    print("🧪 Testing Similarity Endpoint...")
    
    sample_data = {
        "resume": "John Smith\nPython Developer\n5 years experience\nBachelor in Computer Science",
        "job_desc": "Python Developer\n3+ years experience\nBachelor degree required"
    }
    
    try:
        response = requests.post(
            "http://localhost:8000/similarity",
            json=sample_data,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Similarity Endpoint working")
            print(f"   Score: {result.get('similarity_score', 0):.3f}")
            return True
        else:
            print(f"❌ Similarity Endpoint failed: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Similarity Endpoint error: {e}")
        return False

def open_browser():
    """Open browser to the Streamlit app"""
    print("🌐 Opening browser...")
    time.sleep(2)  # Give Streamlit time to start
    webbrowser.open("http://localhost:8501")

def main():
    """Main demo function"""
    print("🤖 HR Assistant - Complete System Demo")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not os.path.exists("similarity_app.py"):
        print("❌ Please run this script from the HR Assistant project directory")
        return
    
    print("📋 This demo will:")
    print("1. Start the FastAPI backend (similarity scoring)")
    print("2. Start the Streamlit frontend (user interface)")
    print("3. Test the API connection")
    print("4. Open the web interface in your browser")
    print()
    
    input("Press Enter to continue...")
    
    # Step 1: Start FastAPI backend
    print("\n" + "=" * 50)
    print("Step 1: Starting FastAPI Backend")
    print("=" * 50)
    
    if not start_fastapi_backend():
        print("❌ Cannot start FastAPI backend. Please check the error messages above.")
        return
    
    # Step 2: Test API
    print("\n" + "=" * 50)
    print("Step 2: Testing API Connection")
    print("=" * 50)
    
    if not test_api_connection():
        print("❌ API is not responding. Please check the backend logs.")
        return
    
    if not test_similarity_endpoint():
        print("❌ Similarity endpoint is not working. Please check the backend logs.")
        return
    
    # Step 3: Start Streamlit frontend
    print("\n" + "=" * 50)
    print("Step 3: Starting Streamlit Frontend")
    print("=" * 50)
    
    if not start_streamlit_frontend():
        print("❌ Cannot start Streamlit frontend. Please check the error messages above.")
        return
    
    # Step 4: Open browser
    print("\n" + "=" * 50)
    print("Step 4: Opening Web Interface")
    print("=" * 50)
    
    open_browser()
    
    # Success message
    print("\n🎉 HR Assistant is now running!")
    print("=" * 50)
    print("📊 FastAPI Backend: http://localhost:8000")
    print("🎨 Streamlit Frontend: http://localhost:8501")
    print("📖 API Documentation: http://localhost:8000/docs")
    print()
    print("💡 Usage Instructions:")
    print("1. Open http://localhost:8501 in your browser")
    print("2. Paste a resume in the left text area")
    print("3. Paste a job description in the right text area")
    print("4. Click 'Analyze Compatibility' to get the similarity score")
    print()
    print("🛑 To stop the services:")
    print("   - Press Ctrl+C in this terminal")
    print("   - Or close the terminal window")
    
    try:
        # Keep the script running
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n👋 Shutting down HR Assistant...")
        print("✅ Demo completed!")

if __name__ == "__main__":
    main()
