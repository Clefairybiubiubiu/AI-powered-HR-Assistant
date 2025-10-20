#!/usr/bin/env python3
"""
Startup script for the Similarity API
"""
import subprocess
import sys
import os

def main():
    """Start the similarity API server"""
    print("🚀 Starting Resume-Job Similarity API...")
    print("📍 Server will be available at: http://localhost:8000")
    print("📖 API Documentation: http://localhost:8000/docs")
    print("🔍 Health Check: http://localhost:8000/health")
    print("=" * 60)
    
    try:
        subprocess.run([
            sys.executable, '-m', 'uvicorn', 
            'similarity_app:app', 
            '--reload', 
            '--host', '0.0.0.0', 
            '--port', '8000'
        ], check=True)
    except KeyboardInterrupt:
        print("\n👋 Similarity API server stopped.")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error starting server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
