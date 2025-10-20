#!/usr/bin/env python3
"""
Startup script for HR Assistant Frontend
"""
import subprocess
import sys
import os

def main():
    """Start the Streamlit frontend server"""
    print("🎨 Starting HR Assistant Frontend...")
    
    # Change to frontend directory
    frontend_dir = os.path.join(os.path.dirname(__file__), 'frontend')
    os.chdir(frontend_dir)
    
    # Start the server
    try:
        subprocess.run([
            sys.executable, '-m', 'streamlit', 
            'run', 'app.py',
            '--server.port', '8501',
            '--server.address', '0.0.0.0'
        ], check=True)
    except KeyboardInterrupt:
        print("\n👋 Frontend server stopped.")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error starting frontend: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
