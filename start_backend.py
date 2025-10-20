#!/usr/bin/env python3
"""
Startup script for HR Assistant Backend
"""
import subprocess
import sys
import os

def main():
    """Start the FastAPI backend server"""
    print("🚀 Starting HR Assistant Backend...")
    
    # Change to backend directory
    backend_dir = os.path.join(os.path.dirname(__file__), 'backend')
    os.chdir(backend_dir)
    
    # Start the server
    try:
        subprocess.run([
            sys.executable, '-m', 'uvicorn', 
            'main:app', 
            '--reload', 
            '--host', '0.0.0.0', 
            '--port', '8000'
        ], check=True)
    except KeyboardInterrupt:
        print("\n👋 Backend server stopped.")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error starting backend: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
