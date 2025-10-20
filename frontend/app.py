"""
Streamlit frontend for HR Assistant - Resume-Job Similarity Matching
"""
import streamlit as st
import requests
import json
import time
import io
import PyPDF2
import docx
from typing import Dict, Any, Optional

# Configure page
st.set_page_config(
    page_title="HR Assistant - Resume Matching",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API Configuration
API_BASE_URL = "http://localhost:8000"
SIMILARITY_ENDPOINT = f"{API_BASE_URL}/similarity"
SIMILARITY_FILES_ENDPOINT = f"{API_BASE_URL}/similarity-files"
HEALTH_ENDPOINT = f"{API_BASE_URL}/health"

def parse_pdf(file_content: bytes) -> str:
    """Parse PDF file content and extract text"""
    try:
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(file_content))
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        return text
    except Exception as e:
        st.error(f"Error parsing PDF: {str(e)}")
        return ""

def parse_docx(file_content: bytes) -> str:
    """Parse DOCX file content and extract text"""
    try:
        doc = docx.Document(io.BytesIO(file_content))
        text = ""
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
        return text
    except Exception as e:
        st.error(f"Error parsing DOCX: {str(e)}")
        return ""

def parse_uploaded_file(uploaded_file) -> str:
    """Parse uploaded file and extract text"""
    if uploaded_file is None:
        return ""
    
    file_type = uploaded_file.type
    file_content = uploaded_file.read()
    
    if file_type == "application/pdf":
        return parse_pdf(file_content)
    elif file_type in ["application/vnd.openxmlformats-officedocument.wordprocessingml.document", 
                       "application/msword"]:
        return parse_docx(file_content)
    elif file_type == "text/plain":
        return file_content.decode('utf-8')
    else:
        st.error(f"Unsupported file type: {file_type}")
        return ""

def check_api_health() -> bool:
    """Check if the FastAPI backend is running"""
    try:
        response = requests.get(HEALTH_ENDPOINT, timeout=5)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False

def send_similarity_request(resume_text: str, job_description: str) -> Optional[Dict[str, Any]]:
    """Send POST request to similarity endpoint"""
    try:
        payload = {
            "resume": resume_text,
            "job_desc": job_description
        }
        
        response = requests.post(
            SIMILARITY_ENDPOINT,
            json=payload,
            timeout=30
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API Error: {response.status_code} - {response.text}")
            return None
            
    except requests.exceptions.ConnectionError:
        st.error("❌ Cannot connect to API server. Please make sure the FastAPI backend is running on http://localhost:8000")
        return None
    except requests.exceptions.Timeout:
        st.error("⏱️ Request timed out. The analysis is taking longer than expected.")
        return None
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        return None

def send_file_upload_request(resume_file, job_description: str) -> Optional[Dict[str, Any]]:
    """Send file upload request to similarity endpoint"""
    try:
        # Prepare file upload
        files = {
            'resume_file': (resume_file.name, resume_file, resume_file.type)
        }
        data = {
            'job_desc': job_description
        }
        
        response = requests.post(
            SIMILARITY_FILES_ENDPOINT,
            files=files,
            data=data,
            timeout=30
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API Error: {response.status_code} - {response.text}")
            return None
            
    except requests.exceptions.ConnectionError:
        st.error("❌ Cannot connect to API server. Please make sure the FastAPI backend is running on http://localhost:8000")
        return None
    except requests.exceptions.Timeout:
        st.error("⏱️ Request timed out. The analysis is taking longer than expected.")
        return None
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        return None

def display_score_results(result: Dict[str, Any]):
    """Display the similarity score results"""
    
    # Main score display
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="🎯 Overall Score",
            value=f"{result['similarity_score']:.1%}",
            delta=None
        )
    
    with col2:
        st.metric(
            label="🔧 Skill Match",
            value=f"{result['skill_match']:.1%}",
            delta=None
        )
    
    with col3:
        st.metric(
            label="💼 Experience",
            value=f"{result['experience_alignment']:.1%}",
            delta=None
        )
    
    with col4:
        st.metric(
            label="🎓 Education",
            value=f"{result['education_match']:.1%}",
            delta=None
        )
    
    # Progress bars
    st.subheader("📊 Detailed Breakdown")
    
    # Overall score progress bar
    st.write("**Overall Similarity Score**")
    st.progress(result['similarity_score'])
    st.caption(f"Score: {result['similarity_score']:.3f}")
    
    # Component scores
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Skill Match**")
        st.progress(result['skill_match'])
        st.caption(f"Skills aligned: {result['skill_match']:.3f}")
        
        st.write("**Experience Alignment**")
        st.progress(result['experience_alignment'])
        st.caption(f"Experience match: {result['experience_alignment']:.3f}")
    
    with col2:
        st.write("**Education Match**")
        st.progress(result['education_match'])
        st.caption(f"Education match: {result['education_match']:.3f}")
        
        st.write("**Semantic Similarity**")
        st.progress(result['semantic_similarity'])
        st.caption(f"Text similarity: {result['semantic_similarity']:.3f}")

def display_detailed_analysis(result: Dict[str, Any]):
    """Display detailed analysis results"""
    
    if 'details' not in result:
        return
    
    details = result['details']
    
    # Create tabs for different analysis sections
    tab1, tab2, tab3, tab4 = st.tabs(["🔍 Skills Analysis", "📋 Matching Details", "⚖️ Scoring Weights", "📊 Additional Info"])
    
    with tab1:
        st.subheader("Skills Analysis")
        
        if 'resume_skills' in details:
            st.write("**Skills Found in Resume:**")
            resume_skills = details['resume_skills']
            if resume_skills:
                # Display skills in columns
                cols = st.columns(3)
                for i, skill in enumerate(resume_skills[:15]):  # Show first 15
                    with cols[i % 3]:
                        st.write(f"• {skill}")
                if len(resume_skills) > 15:
                    st.caption(f"... and {len(resume_skills) - 15} more skills")
            else:
                st.write("No skills detected")
        
        if 'job_skills' in details:
            st.write("**Skills Required for Job:**")
            job_skills = details['job_skills']
            if job_skills:
                cols = st.columns(3)
                for i, skill in enumerate(job_skills[:15]):  # Show first 15
                    with cols[i % 3]:
                        st.write(f"• {skill}")
                if len(job_skills) > 15:
                    st.caption(f"... and {len(job_skills) - 15} more skills")
            else:
                st.write("No skills detected")
    
    with tab2:
        st.subheader("Matching Details")
        
        if 'matched_skills' in details:
            matched = details['matched_skills']
            st.write(f"**✅ Matched Skills ({len(matched)}):**")
            if matched:
                cols = st.columns(3)
                for i, skill in enumerate(matched):
                    with cols[i % 3]:
                        st.success(f"✓ {skill}")
            else:
                st.write("No skills matched")
        
        if 'missing_skills' in details:
            missing = details['missing_skills']
            st.write(f"**❌ Missing Skills ({len(missing)}):**")
            if missing:
                cols = st.columns(3)
                for i, skill in enumerate(missing[:10]):  # Show first 10
                    with cols[i % 3]:
                        st.error(f"✗ {skill}")
                if len(missing) > 10:
                    st.caption(f"... and {len(missing) - 10} more missing skills")
            else:
                st.write("All required skills are present!")
    
    with tab3:
        st.subheader("Scoring Weights")
        
        if 'weights' in details:
            weights = details['weights']
            st.write("**Formula:** `score = α × skill_match + β × experience_alignment + γ × education_match`")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("α (Skill Match)", f"{weights['alpha']:.1f}")
            
            with col2:
                st.metric("β (Experience)", f"{weights['beta']:.1f}")
            
            with col3:
                st.metric("γ (Education)", f"{weights['gamma']:.1f}")
            
            # Show calculation
            st.write("**Calculation:**")
            st.code(f"""
α × skill_match = {weights['alpha']:.1f} × {result['skill_match']:.3f} = {weights['alpha'] * result['skill_match']:.3f}
β × experience = {weights['beta']:.1f} × {result['experience_alignment']:.3f} = {weights['beta'] * result['experience_alignment']:.3f}
γ × education = {weights['gamma']:.1f} × {result['education_match']:.3f} = {weights['gamma'] * result['education_match']:.3f}
Total = {result['similarity_score']:.3f}
            """)
    
    with tab4:
        st.subheader("Additional Information")
        
        # Show any additional details
        for key, value in details.items():
            if key not in ['resume_skills', 'job_skills', 'matched_skills', 'missing_skills', 'weights']:
                st.write(f"**{key.replace('_', ' ').title()}:**")
                if isinstance(value, (list, dict)):
                    st.json(value)
                else:
                    st.write(str(value))

def main():
    """Main Streamlit application"""
    
    # Header
    st.title("🤖 HR Assistant - Resume-Job Matching")
    st.markdown("Analyze the compatibility between resumes and job descriptions using AI-powered similarity scoring.")
    st.markdown("---")
    
    # Check API health
    with st.spinner("Checking API connection..."):
        api_healthy = check_api_health()
    
    if not api_healthy:
        st.error("""
        ❌ **API Server Not Available**
        
        Please make sure the FastAPI backend is running:
        
        1. Open a terminal and navigate to the project directory
        2. Run: `python similarity_app.py` or `python run_similarity_app.py`
        3. Wait for the server to start on http://localhost:8000
        4. Refresh this page
        """)
        return
    
    st.success("✅ Connected to API server")
    
    # Sidebar
    with st.sidebar:
        st.header("📋 Instructions")
        st.markdown("""
        **Option 1: Upload Files**
        1. Upload PDF/DOCX resume file
        2. Upload PDF/DOCX job description file
        3. Click Analyze
        
        **Option 2: Paste Text**
        1. Paste resume text in the text area
        2. Paste job description in the text area
        3. Click Analyze
        """)
        
        st.header("📁 Supported File Types")
        st.markdown("""
        - **PDF**: .pdf files
        - **Word**: .docx, .doc files
        - **Text**: .txt files
        """)
        
        st.header("⚙️ Settings")
        show_details = st.checkbox("Show Detailed Analysis", value=True)
        show_weights = st.checkbox("Show Scoring Weights", value=True)
    
    # Input method selection
    st.subheader("📝 Input Method")
    input_method = st.radio(
        "Choose how to provide the documents:",
        ["📁 Upload Files", "📝 Paste Text"],
        horizontal=True
    )
    
    st.markdown("---")
    
    # Main content based on input method
    if input_method == "📁 Upload Files":
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📄 Resume File")
            resume_file = st.file_uploader(
                "Upload resume file (PDF, DOCX, TXT):",
                type=['pdf', 'docx', 'doc', 'txt'],
                help="Supported formats: PDF, DOCX, DOC, TXT"
            )
            
            if resume_file is not None:
                st.success(f"✅ File uploaded: {resume_file.name}")
                with st.spinner("Parsing resume file..."):
                    resume_text = parse_uploaded_file(resume_file)
                    if resume_text:
                        st.text_area(
                            "Parsed Resume Text:",
                            value=resume_text,
                            height=200,
                            help="This is the extracted text from your uploaded file"
                        )
                    else:
                        st.error("❌ Failed to parse the resume file")
                        resume_text = ""
            else:
                resume_text = ""
        
        with col2:
            st.subheader("💼 Job Description File")
            job_file = st.file_uploader(
                "Upload job description file (PDF, DOCX, TXT):",
                type=['pdf', 'docx', 'doc', 'txt'],
                help="Supported formats: PDF, DOCX, DOC, TXT"
            )
            
            if job_file is not None:
                st.success(f"✅ File uploaded: {job_file.name}")
                with st.spinner("Parsing job description file..."):
                    job_description = parse_uploaded_file(job_file)
                    if job_description:
                        st.text_area(
                            "Parsed Job Description:",
                            value=job_description,
                            height=200,
                            help="This is the extracted text from your uploaded file"
                        )
                    else:
                        st.error("❌ Failed to parse the job description file")
                        job_description = ""
            else:
                job_description = ""
    
    else:  # Paste Text method
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📄 Resume Text")
            resume_text = st.text_area(
                "Paste the candidate's resume here:",
                height=300,
                placeholder="""John Smith
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
University of California, Berkeley | 2016"""
            )
        
        with col2:
            st.subheader("💼 Job Description")
            job_description = st.text_area(
                "Paste the job description here:",
                height=300,
                placeholder="""Senior Python Developer

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

Education: Bachelor's degree in Computer Science or related field preferred"""
            )
    
    # Analyze button
    st.markdown("---")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        analyze_button = st.button(
            "🔍 Analyze Compatibility",
            type="primary",
            use_container_width=True
        )
    
    # Process analysis
    if analyze_button:
        # Validate inputs based on input method
        if input_method == "📁 Upload Files":
            # Check if files are uploaded
            if 'resume_file' in locals() and resume_file is not None:
                if not job_description.strip():
                    st.error("❌ Please enter job description")
                    return
                
                # Show progress
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Simulate progress steps
                status_text.text("🔄 Parsing resume file...")
                progress_bar.progress(25)
                time.sleep(0.5)
                
                status_text.text("🔄 Analyzing job description...")
                progress_bar.progress(50)
                time.sleep(0.5)
                
                status_text.text("🔄 Computing similarity scores...")
                progress_bar.progress(75)
                time.sleep(0.5)
                
                # Send file upload request to API
                status_text.text("🔄 Sending file to API...")
                result = send_file_upload_request(resume_file, job_description)
                
            else:
                st.error("❌ Please upload a resume file")
                return
        else:
            # Text input method
            if not resume_text.strip():
                st.error("❌ Please enter resume text")
                return
            
            if not job_description.strip():
                st.error("❌ Please enter job description")
                return
            
            # Show progress
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Simulate progress steps
            status_text.text("🔄 Analyzing resume...")
            progress_bar.progress(25)
            time.sleep(0.5)
            
            status_text.text("🔄 Analyzing job description...")
            progress_bar.progress(50)
            time.sleep(0.5)
            
            status_text.text("🔄 Computing similarity scores...")
            progress_bar.progress(75)
            time.sleep(0.5)
            
            # Send text request to API
            status_text.text("🔄 Sending request to API...")
            result = send_similarity_request(resume_text, job_description)
        
        if result:
            progress_bar.progress(100)
            status_text.text("✅ Analysis complete!")
            time.sleep(1)
            
            # Clear progress indicators
            progress_bar.empty()
            status_text.empty()
            
            # Display results
            st.markdown("---")
            st.subheader("📊 Analysis Results")
            
            # Main score display
            display_score_results(result)
            
            # Detailed analysis
            if show_details:
                st.markdown("---")
                display_detailed_analysis(result)
            
            # Success message
            st.success("🎉 Analysis completed successfully!")
            
        else:
            progress_bar.empty()
            status_text.empty()
            st.error("❌ Failed to analyze. Please check the API connection and try again.")

if __name__ == "__main__":
    main()