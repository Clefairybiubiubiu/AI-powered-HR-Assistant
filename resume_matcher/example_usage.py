"""
Example usage of the refactored Resume-JD Matcher.
Demonstrates the improved structure and features.
"""
from pathlib import Path

from resume_matcher.config import config
from resume_matcher.logging_config import setup_logging
from resume_matcher.matchers import ResumeJDMatcher

# Set up logging
setup_logging(level="INFO")

# Configure data directory
data_dir = Path("/Users/junfeibai/Desktop/5560/test")
config.data_dir = data_dir

# Create matcher
print("Creating TF-IDF matcher...")
matcher = ResumeJDMatcher(str(data_dir))

# Load documents
print("Loading documents...")
try:
    matcher.load_documents()
    print(f"✅ Loaded {len(matcher.resumes)} resumes and {len(matcher.job_descriptions)} job descriptions")
except Exception as e:
    print(f"❌ Error loading documents: {e}")
    exit(1)

# Compute similarity
print("Computing similarity...")
try:
    similarity_matrix = matcher.compute_similarity()
    print(f"✅ Similarity matrix computed: {similarity_matrix.shape}")
except Exception as e:
    print(f"❌ Error computing similarity: {e}")
    exit(1)

# Get top matches
print("\nTop matches:")
top_matches = matcher.get_top_matches(top_k=3)
print(top_matches.to_string())

# Get directory info
print("\nDirectory information:")
dir_info = matcher.get_directory_info()
print(f"Resume files: {len(dir_info['resume_files'])}")
print(f"JD files: {len(dir_info['jd_files'])}")
print(f"Total files: {dir_info['total_files']}")

print("\n✅ Example completed successfully!")

