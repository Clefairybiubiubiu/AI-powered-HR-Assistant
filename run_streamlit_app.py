#!/usr/bin/env python3
"""
Startup script for HR Assistant Streamlit Frontend
"""
import subprocess
import sys
import os

def main():
    """Start the Streamlit frontend server"""
    print("🎨 Starting HR Assistant Streamlit Frontend...")
    print("📍 App will be available at: http://localhost:8501")
    print("🔗 Make sure the FastAPI backend is running on http://localhost:8000")
    print("=" * 60)
    
    # Change to frontend directory
    frontend_dir = os.path.join(os.path.dirname(__file__), 'frontend')
    os.chdir(frontend_dir)
    
    try:
        subprocess.run([
            sys.executable, '-m', 'streamlit', 
            'run', 'app.py',
            '--server.port', '8501',
            '--server.address', '0.0.0.0',
            '--server.headless', 'true'
        ], check=True)
    except KeyboardInterrupt:
        print("\n👋 Streamlit app stopped.")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error starting Streamlit: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
