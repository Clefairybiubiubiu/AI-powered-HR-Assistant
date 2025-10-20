"""
Test script for ResumeParser class
"""
import sys
import os

# Add the backend directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

from models.document_parser import ResumeParser

def test_resume_parser():
    """Test the ResumeParser functionality"""
    
    # Sample resume text
    sample_resume = """
    John Smith
    Senior Software Engineer
    Email: john.smith@email.com
    Phone: (555) 123-4567
    
    SUMMARY
    Experienced software engineer with 6+ years of experience in Python development, 
    microservices architecture, and cloud technologies. Strong background in building 
    scalable web applications and leading development teams.
    
    TECHNICAL SKILLS
    Programming Languages: Python, JavaScript, TypeScript, SQL
    Frameworks: Django, Flask, React, Node.js
    Databases: PostgreSQL, MongoDB, Redis
    Cloud Platforms: AWS, Azure, Docker, Kubernetes
    Tools: Git, Jenkins, Jira, Confluence
    
    EXPERIENCE
    Senior Software Engineer | TechCorp Inc. | 2020 - Present
    - Led development of microservices architecture serving 1M+ users
    - Implemented CI/CD pipelines using Jenkins and Docker
    - Technologies: Python, Django, PostgreSQL, AWS, Kubernetes
    
    Software Engineer | StartupXYZ | 2018 - 2020
    - Developed REST APIs using Flask and FastAPI
    - Built data processing pipelines using Python and pandas
    - Technologies: Python, Flask, MongoDB, Docker
    
    Junior Developer | WebDev Co. | 2016 - 2018
    - Developed web applications using Django and React
    - Worked on database design and optimization
    - Technologies: Python, Django, JavaScript, PostgreSQL
    
    EDUCATION
    Bachelor of Science in Computer Science
    University of California, Berkeley | 2016
    
    Master of Science in Software Engineering
    Stanford University | 2018
    
    CERTIFICATIONS
    AWS Certified Solutions Architect
    Google Cloud Professional Developer
    """
    
    print("🧪 Testing ResumeParser...")
    print("=" * 60)
    
    try:
        # Initialize parser
        parser = ResumeParser()
        
        # Parse resume
        result = parser.parse_resume(sample_resume)
        
        print("✅ ResumeParser Results:")
        print(f"📊 Skills Found: {len(result['skills'])}")
        print(f"   {', '.join(result['skills'][:10])}{'...' if len(result['skills']) > 10 else ''}")
        
        print(f"\n🎓 Education Found: {len(result['education'])}")
        for edu in result['education']:
            print(f"   - {edu}")
        
        print(f"\n💼 Experience Found: {len(result['experience'])}")
        for exp in result['experience']:
            print(f"   - {exp}")
        
        # Test additional functionality
        print(f"\n📈 Text Statistics:")
        stats = parser.get_text_statistics(sample_resume)
        print(f"   Word Count: {stats.get('word_count', 0)}")
        print(f"   Sentence Count: {stats.get('sentence_count', 0)}")
        print(f"   Unique Words: {stats.get('unique_words', 0)}")
        print(f"   Named Entities: {stats.get('named_entities', 0)}")
        
        print(f"\n🏷️ Named Entities:")
        entities = parser.extract_named_entities(sample_resume)
        for entity, label in entities[:10]:  # Show first 10
            print(f"   {entity} ({label})")
        
        print(f"\n📊 Keyword Frequencies:")
        freq = parser.get_keyword_frequency(sample_resume)
        for keyword, count in sorted(freq.items(), key=lambda x: x[1], reverse=True)[:10]:
            print(f"   {keyword}: {count}")
        
        print(f"\n🎯 POS Tag Distribution:")
        pos_tags = stats.get('pos_tags', {})
        for pos, count in sorted(pos_tags.items(), key=lambda x: x[1], reverse=True)[:5]:
            print(f"   {pos}: {count}")
        
    except Exception as e:
        print(f"❌ Error testing ResumeParser: {e}")
        print("Make sure spaCy model is installed: python -m spacy download en_core_web_sm")


def test_with_different_resume():
    """Test with a different resume format"""
    
    data_scientist_resume = """
    Sarah Johnson
    Data Scientist
    Email: sarah.johnson@email.com
    
    PROFESSIONAL SUMMARY
    Data scientist with 4+ years of experience in machine learning, statistical analysis, 
    and data visualization. Passionate about turning data into actionable insights and 
    building predictive models that drive business value.
    
    TECHNICAL SKILLS
    Programming: Python, R, SQL
    Data Science: pandas, numpy, scikit-learn, tensorflow, pytorch
    Visualization: matplotlib, seaborn, plotly, tableau
    Databases: PostgreSQL, MySQL, MongoDB
    Cloud: AWS, Google Cloud Platform
    Tools: Jupyter, Git, Docker
    
    EXPERIENCE
    Senior Data Scientist | DataCorp | 2021 - Present
    - Built ML models for customer churn prediction with 85% accuracy
    - Led data science projects worth $2M+ in business impact
    - Technologies: Python, scikit-learn, AWS, PostgreSQL
    
    Data Scientist | Analytics Inc. | 2019 - 2021
    - Developed recommendation systems using collaborative filtering
    - Created interactive dashboards using Tableau and Plotly
    - Technologies: Python, R, SQL, Tableau
    
    EDUCATION
    Master of Science in Data Science
    Columbia University | 2018
    
    Bachelor of Science in Statistics
    University of Michigan | 2016
    """
    
    print("\n" + "=" * 60)
    print("🧪 Testing with Data Scientist Resume...")
    print("=" * 60)
    
    try:
        parser = ResumeParser()
        result = parser.parse_resume(data_scientist_resume)
        
        print("✅ Data Scientist Resume Results:")
        print(f"📊 Skills: {', '.join(result['skills'][:8])}")
        print(f"🎓 Education: {', '.join(result['education'][:5])}")
        print(f"💼 Experience: {', '.join(result['experience'][:5])}")
        
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    print("🚀 ResumeParser Test Suite")
    print("=" * 60)
    
    # Test main functionality
    test_resume_parser()
    
    # Test with different resume
    test_with_different_resume()
    
    print("\n✅ ResumeParser testing completed!")
