"""
Example usage of SimilarityScorer class
"""
import sys
import os

# Add the backend directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

from models.similarity_scorer import SimilarityScorer

def main():
    """Demonstrate SimilarityScorer usage"""
    
    # Sample resume
    resume_text = """
    Jane Doe
    Senior Data Scientist
    Email: jane.doe@email.com
    
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
    """
    
    # Sample job description
    job_description = """
    Senior Data Scientist
    
    We are looking for a Senior Data Scientist to join our team. You will be responsible 
    for building machine learning models, analyzing large datasets, and providing insights 
    to drive business decisions.
    
    Requirements:
    - 3+ years of data science experience
    - Strong Python skills
    - Experience with pandas, numpy, scikit-learn
    - Experience with SQL
    - Knowledge of machine learning algorithms
    - Experience with data visualization tools
    - PhD or Master's degree in related field preferred
    """
    
    print("🚀 SimilarityScorer Example")
    print("=" * 50)
    
    # Initialize scorer
    scorer = SimilarityScorer()
    
    # Compute fit score
    result = scorer.compute_fit_score(resume_text, job_description)
    
    print("📊 Fit Score Results:")
    print(f"Overall Score: {result['overall_score']:.3f}")
    print(f"Skill Match: {result['skill_match']:.3f}")
    print(f"Experience Alignment: {result['experience_alignment']:.3f}")
    print(f"Education Match: {result['education_match']:.3f}")
    print(f"Semantic Similarity: {result['semantic_similarity']:.3f}")
    
    print(f"\n⚖️ Scoring Formula:")
    weights = result['weights']
    print(f"score = α * skill_match + β * experience_alignment + γ * education_match")
    print(f"score = {weights['alpha']} * {result['skill_match']:.3f} + {weights['beta']} * {result['experience_alignment']:.3f} + {weights['gamma']} * {result['education_match']:.3f}")
    print(f"score = {weights['alpha'] * result['skill_match']:.3f} + {weights['beta'] * result['experience_alignment']:.3f} + {weights['gamma'] * result['education_match']:.3f}")
    print(f"score = {result['overall_score']:.3f}")
    
    print(f"\n📋 Resume Analysis:")
    resume_analysis = result['resume_analysis']
    print(f"Skills Found: {len(resume_analysis['skills_found'])}")
    print(f"Education Found: {len(resume_analysis['education_found'])}")
    print(f"Experience Found: {len(resume_analysis['experience_found'])}")
    
    print(f"\n💼 Job Analysis:")
    job_analysis = result['job_analysis']
    print(f"Skills Required: {len(job_analysis['skills_required'])}")
    print(f"Skill Categories: {list(job_analysis['skills_categories'].keys())}")
    
    print(f"\n🔍 Matching Details:")
    matching = result['matching_details']
    print(f"Matched Skills: {len(matching['matched_skills'])}")
    print(f"Missing Skills: {len(matching['missing_skills'])}")
    print(f"Extra Skills: {len(matching['extra_skills'])}")
    
    print(f"\n📝 Top Matched Skills:")
    for skill in matching['matched_skills'][:10]:
        print(f"  ✓ {skill}")
    
    print(f"\n❌ Missing Skills:")
    for skill in matching['missing_skills'][:10]:
        print(f"  ✗ {skill}")
    
    print(f"\n➕ Extra Skills:")
    for skill in matching['extra_skills'][:10]:
        print(f"  + {skill}")

def demonstrate_batch_scoring():
    """Demonstrate batch scoring functionality"""
    
    print("\n" + "=" * 50)
    print("📊 Batch Scoring Example")
    print("=" * 50)
    
    # Multiple candidates
    candidates = [
        {
            'name': 'Alice Johnson',
            'resume': """
            Alice Johnson
            Senior Python Developer
            Python, Django, Flask, PostgreSQL, AWS, Docker
            6+ years experience in software development
            Master of Science in Computer Science
            """
        },
        {
            'name': 'Bob Smith',
            'resume': """
            Bob Smith
            Data Scientist
            Python, R, SQL, pandas, numpy, scikit-learn
            4+ years experience in data science
            PhD in Statistics
            """
        },
        {
            'name': 'Carol Davis',
            'resume': """
            Carol Davis
            Frontend Developer
            JavaScript, React, HTML, CSS, Node.js
            3 years experience in web development
            Bachelor of Science in Information Technology
            """
        }
    ]
    
    job_description = """
    Senior Python Developer
    
    We need a Senior Python Developer with 5+ years of experience.
    Must have experience with Django, Flask, PostgreSQL, AWS, and Docker.
    Bachelor's degree in Computer Science preferred.
    """
    
    scorer = SimilarityScorer()
    results = scorer.batch_score_candidates(candidates, job_description)
    
    print("Rank | Name | Score | Skill Match | Experience | Education")
    print("-" * 70)
    
    for i, result in enumerate(results, 1):
        name = result.get('candidate_name', 'Unknown')
        score = result.get('overall_score', 0)
        skill_match = result.get('skill_match', 0)
        experience = result.get('experience_alignment', 0)
        education = result.get('education_match', 0)
        
        print(f"{i:4d} | {name:12s} | {score:.3f} | {skill_match:.3f} | {experience:.3f} | {education:.3f}")

def demonstrate_detailed_analysis():
    """Demonstrate detailed analysis functionality"""
    
    print("\n" + "=" * 50)
    print("🔍 Detailed Analysis Example")
    print("=" * 50)
    
    resume_text = """
    Michael Chen
    Software Engineer
    Python, Java, JavaScript, React, Node.js, MongoDB, AWS
    3 years experience in full-stack development
    Bachelor of Science in Computer Science
    """
    
    job_description = """
    Full Stack Developer
    
    We are looking for a Full Stack Developer with experience in:
    - Frontend: React, JavaScript, HTML, CSS
    - Backend: Python, Node.js, Django, Flask
    - Databases: MongoDB, PostgreSQL
    - Cloud: AWS, Docker
    - 2+ years experience required
    - Bachelor's degree preferred
    """
    
    scorer = SimilarityScorer()
    detailed_result = scorer.get_detailed_analysis(resume_text, job_description)
    
    print("📊 Detailed Analysis Results:")
    print(f"Overall Score: {detailed_result['overall_score']:.3f}")
    
    print(f"\n📈 Resume Statistics:")
    stats = detailed_result['resume_statistics']
    print(f"Word Count: {stats.get('word_count', 0)}")
    print(f"Sentence Count: {stats.get('sentence_count', 0)}")
    print(f"Unique Words: {stats.get('unique_words', 0)}")
    print(f"Named Entities: {stats.get('named_entities', 0)}")
    
    print(f"\n🏷️ Named Entities:")
    entities = detailed_result['resume_entities']
    for entity, label in entities[:10]:
        print(f"  {entity} ({label})")
    
    print(f"\n📊 Keyword Frequencies:")
    freq = detailed_result['resume_keyword_frequency']
    for keyword, count in sorted(freq.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"  {keyword}: {count}")

if __name__ == "__main__":
    # Basic usage
    main()
    
    # Batch scoring
    demonstrate_batch_scoring()
    
    # Detailed analysis
    demonstrate_detailed_analysis()
    
    print("\n✅ SimilarityScorer example completed!")
