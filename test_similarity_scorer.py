"""
Test script for SimilarityScorer functionality
"""
import sys
import os

# Add the backend directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

from models.similarity_scorer import SimilarityScorer

def test_similarity_scorer():
    """Test the SimilarityScorer functionality"""
    
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
    
    EDUCATION
    Bachelor of Science in Computer Science
    University of California, Berkeley | 2016
    
    Master of Science in Software Engineering
    Stanford University | 2018
    """
    
    # Sample job description
    sample_job = """
    Senior Python Developer
    
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
    
    Education: Bachelor's degree in Computer Science or related field preferred
    """
    
    print("🧪 Testing SimilarityScorer...")
    print("=" * 60)
    
    try:
        # Initialize scorer
        scorer = SimilarityScorer()
        
        # Test basic fit score calculation
        print("📊 Computing Fit Score...")
        result = scorer.compute_fit_score(sample_resume, sample_job)
        
        print("✅ Fit Score Results:")
        print(f"🎯 Overall Score: {result['overall_score']:.3f}")
        print(f"🔧 Skill Match: {result['skill_match']:.3f}")
        print(f"💼 Experience Alignment: {result['experience_alignment']:.3f}")
        print(f"🎓 Education Match: {result['education_match']:.3f}")
        print(f"🧠 Semantic Similarity: {result['semantic_similarity']:.3f}")
        
        print(f"\n⚖️ Scoring Weights:")
        weights = result['weights']
        print(f"   α (skill_match): {weights['alpha']}")
        print(f"   β (experience_alignment): {weights['beta']}")
        print(f"   γ (education_match): {weights['gamma']}")
        
        print(f"\n📋 Resume Analysis:")
        resume_analysis = result['resume_analysis']
        print(f"   Skills Found: {len(resume_analysis['skills_found'])}")
        print(f"   Education Found: {len(resume_analysis['education_found'])}")
        print(f"   Experience Found: {len(resume_analysis['experience_found'])}")
        
        print(f"\n💼 Job Analysis:")
        job_analysis = result['job_analysis']
        print(f"   Skills Required: {len(job_analysis['skills_required'])}")
        print(f"   Skill Categories: {list(job_analysis['skills_categories'].keys())}")
        
        print(f"\n🔍 Matching Details:")
        matching = result['matching_details']
        print(f"   Matched Skills: {len(matching['matched_skills'])}")
        print(f"   Missing Skills: {len(matching['missing_skills'])}")
        print(f"   Extra Skills: {len(matching['extra_skills'])}")
        
        # Show some examples
        print(f"\n📝 Examples:")
        print(f"   Matched Skills: {', '.join(matching['matched_skills'][:5])}")
        print(f"   Missing Skills: {', '.join(matching['missing_skills'][:5])}")
        print(f"   Resume Skills: {', '.join(resume_analysis['skills_found'][:5])}")
        
        return result
        
    except Exception as e:
        print(f"❌ Error testing SimilarityScorer: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_detailed_analysis():
    """Test detailed analysis functionality"""
    
    sample_resume = """
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
    
    EDUCATION
    Master of Science in Data Science
    Columbia University | 2018
    
    Bachelor of Science in Statistics
    University of Michigan | 2016
    """
    
    sample_job = """
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
    
    print("\n" + "=" * 60)
    print("🧪 Testing Detailed Analysis...")
    print("=" * 60)
    
    try:
        scorer = SimilarityScorer()
        
        # Get detailed analysis
        detailed_result = scorer.get_detailed_analysis(sample_resume, sample_job)
        
        print("✅ Detailed Analysis Results:")
        print(f"🎯 Overall Score: {detailed_result['overall_score']:.3f}")
        
        print(f"\n📊 Resume Statistics:")
        stats = detailed_result['resume_statistics']
        print(f"   Word Count: {stats.get('word_count', 0)}")
        print(f"   Sentence Count: {stats.get('sentence_count', 0)}")
        print(f"   Unique Words: {stats.get('unique_words', 0)}")
        print(f"   Named Entities: {stats.get('named_entities', 0)}")
        
        print(f"\n🏷️ Named Entities (first 10):")
        entities = detailed_result['resume_entities']
        for entity, label in entities[:10]:
            print(f"   {entity} ({label})")
        
        print(f"\n📈 Keyword Frequencies (top 10):")
        freq = detailed_result['resume_keyword_frequency']
        for keyword, count in sorted(freq.items(), key=lambda x: x[1], reverse=True)[:10]:
            print(f"   {keyword}: {count}")
        
        return detailed_result
        
    except Exception as e:
        print(f"❌ Error in detailed analysis: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_batch_scoring():
    """Test batch scoring functionality"""
    
    candidates = [
        {
            'name': 'John Smith',
            'resume': """
            John Smith
            Senior Software Engineer
            Python, Django, AWS, Docker, Kubernetes
            6+ years experience in software development
            Bachelor of Science in Computer Science
            """
        },
        {
            'name': 'Sarah Johnson',
            'resume': """
            Sarah Johnson
            Data Scientist
            Python, R, SQL, pandas, numpy, scikit-learn
            4+ years experience in data science
            Master of Science in Data Science
            """
        },
        {
            'name': 'Mike Wilson',
            'resume': """
            Mike Wilson
            Junior Developer
            JavaScript, React, Node.js, HTML, CSS
            2 years experience in web development
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
    
    print("\n" + "=" * 60)
    print("🧪 Testing Batch Scoring...")
    print("=" * 60)
    
    try:
        scorer = SimilarityScorer()
        
        # Score all candidates
        batch_results = scorer.batch_score_candidates(candidates, job_description)
        
        print("✅ Batch Scoring Results:")
        print("Rank | Name | Score | Skill Match | Experience | Education")
        print("-" * 70)
        
        for i, result in enumerate(batch_results, 1):
            name = result.get('candidate_name', 'Unknown')
            score = result.get('overall_score', 0)
            skill_match = result.get('skill_match', 0)
            experience = result.get('experience_alignment', 0)
            education = result.get('education_match', 0)
            
            print(f"{i:4d} | {name:12s} | {score:.3f} | {skill_match:.3f} | {experience:.3f} | {education:.3f}")
        
        return batch_results
        
    except Exception as e:
        print(f"❌ Error in batch scoring: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_formula_verification():
    """Test that the scoring formula is correctly applied"""
    
    print("\n" + "=" * 60)
    print("🧪 Testing Formula Verification...")
    print("=" * 60)
    
    try:
        scorer = SimilarityScorer()
        
        # Simple test case
        resume = "Python, Django, AWS, 5 years experience, Bachelor degree"
        job = "Python developer with Django, AWS, 3+ years experience, Bachelor required"
        
        result = scorer.compute_fit_score(resume, job)
        
        # Manual calculation
        alpha = result['weights']['alpha']
        beta = result['weights']['beta']
        gamma = result['weights']['gamma']
        
        manual_score = (alpha * result['skill_match'] + 
                       beta * result['experience_alignment'] + 
                       gamma * result['education_match'])
        
        print(f"✅ Formula Verification:")
        print(f"   Computed Score: {result['overall_score']:.6f}")
        print(f"   Manual Score: {manual_score:.6f}")
        print(f"   Match: {abs(result['overall_score'] - manual_score) < 0.0001}")
        
        print(f"\n📊 Component Breakdown:")
        print(f"   α * skill_match = {alpha:.1f} * {result['skill_match']:.3f} = {alpha * result['skill_match']:.3f}")
        print(f"   β * experience = {beta:.1f} * {result['experience_alignment']:.3f} = {beta * result['experience_alignment']:.3f}")
        print(f"   γ * education = {gamma:.1f} * {result['education_match']:.3f} = {gamma * result['education_match']:.3f}")
        print(f"   Total = {manual_score:.3f}")
        
        return result
        
    except Exception as e:
        print(f"❌ Error in formula verification: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    print("🚀 SimilarityScorer Test Suite")
    print("=" * 60)
    
    # Test basic functionality
    basic_result = test_similarity_scorer()
    
    # Test detailed analysis
    detailed_result = test_detailed_analysis()
    
    # Test batch scoring
    batch_result = test_batch_scoring()
    
    # Test formula verification
    formula_result = test_formula_verification()
    
    print("\n✅ All tests completed!")
    
    if basic_result:
        print(f"\n🎯 Final Summary:")
        print(f"   Overall Score: {basic_result['overall_score']:.3f}")
        print(f"   Formula: α={basic_result['weights']['alpha']} * {basic_result['skill_match']:.3f} + β={basic_result['weights']['beta']} * {basic_result['experience_alignment']:.3f} + γ={basic_result['weights']['gamma']} * {basic_result['education_match']:.3f}")
