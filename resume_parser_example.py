"""
Example usage of ResumeParser class
"""
import sys
import os

# Add the backend directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

from models.document_parser import ResumeParser

def main():
    """Demonstrate ResumeParser usage"""
    
    # Sample resume text
    resume_text = """
    Jane Doe
    Senior Data Scientist
    Email: jane.doe@email.com
    Phone: (555) 987-6543
    
    PROFESSIONAL SUMMARY
    Data scientist with 5+ years of experience in machine learning, statistical analysis, 
    and data visualization. Expert in Python, R, and SQL with strong background in 
    building predictive models and leading data science teams.
    
    TECHNICAL SKILLS
    Programming: Python, R, SQL, JavaScript
    Machine Learning: scikit-learn, TensorFlow, PyTorch, XGBoost
    Data Analysis: pandas, numpy, matplotlib, seaborn, plotly
    Databases: PostgreSQL, MySQL, MongoDB, Redis
    Cloud Platforms: AWS, Azure, Google Cloud Platform
    Tools: Jupyter, Git, Docker, Kubernetes, Tableau
    
    EXPERIENCE
    Senior Data Scientist | TechCorp | 2020 - Present
    - Built ML models for customer churn prediction with 90% accuracy
    - Led data science team of 5 engineers
    - Technologies: Python, TensorFlow, AWS, PostgreSQL
    
    Data Scientist | DataCorp | 2018 - 2020
    - Developed recommendation systems using collaborative filtering
    - Created interactive dashboards using Tableau and Plotly
    - Technologies: Python, R, SQL, MongoDB
    
    EDUCATION
    Master of Science in Data Science
    Stanford University | 2018
    
    Bachelor of Science in Computer Science
    University of California, Berkeley | 2016
    
    CERTIFICATIONS
    AWS Certified Machine Learning Specialty
    Google Data Analytics Professional Certificate
    """
    
    print("🚀 ResumeParser Example")
    print("=" * 50)
    
    # Initialize parser
    parser = ResumeParser()
    
    # Parse the resume
    result = parser.parse_resume(resume_text)
    
    print("📋 Parsed Results:")
    print(f"Skills: {result['skills']}")
    print(f"Education: {result['education']}")
    print(f"Experience: {result['experience']}")
    
    # Get additional statistics
    stats = parser.get_text_statistics(resume_text)
    print(f"\n📊 Text Statistics:")
    print(f"Word Count: {stats['word_count']}")
    print(f"Sentence Count: {stats['sentence_count']}")
    print(f"Unique Words: {stats['unique_words']}")
    
    # Get named entities
    entities = parser.extract_named_entities(resume_text)
    print(f"\n🏷️ Named Entities:")
    for entity, label in entities[:10]:
        print(f"  {entity} ({label})")
    
    # Get keyword frequencies
    freq = parser.get_keyword_frequency(resume_text)
    print(f"\n📈 Top Keywords:")
    for keyword, count in sorted(freq.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"  {keyword}: {count}")

if __name__ == "__main__":
    main()
