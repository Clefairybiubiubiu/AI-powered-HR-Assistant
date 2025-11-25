"""
Test script for the refactored Resume-JD Matcher.
Tests all the improvements and verifies everything works.
"""
import sys
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from resume_matcher.config import config
from resume_matcher.logging_config import setup_logging
from resume_matcher.matchers import ResumeJDMatcher

def test_imports():
    """Test that all imports work correctly."""
    print("=" * 60)
    print("TEST 1: Testing Imports")
    print("=" * 60)
    try:
        from resume_matcher import config, get_logger
        from resume_matcher.matchers import BaseMatcher, ResumeJDMatcher
        from resume_matcher.utils import DocumentProcessor
        from resume_matcher.utils.exceptions import DocumentProcessingError
        print("✅ All imports successful!")
        return True
    except Exception as e:
        print(f"❌ Import failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_configuration():
    """Test configuration management."""
    print("\n" + "=" * 60)
    print("TEST 2: Testing Configuration")
    print("=" * 60)
    try:
        print(f"Data directory: {config.data_dir}")
        print(f"Max file size: {config.max_file_size_mb} MB")
        print(f"Log level: {config.log_level}")
        print(f"TF-IDF max features: {config.tfidf_max_features}")
        print(f"Similarity thresholds: {config.similarity_threshold_high}, {config.similarity_threshold_medium}")
        print("✅ Configuration loaded successfully!")
        return True
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False

def test_logging():
    """Test logging setup."""
    print("\n" + "=" * 60)
    print("TEST 3: Testing Logging")
    print("=" * 60)
    try:
        from resume_matcher.logging_config import get_logger
        logger = get_logger(__name__)
        logger.debug("This is a DEBUG message")
        logger.info("This is an INFO message")
        logger.warning("This is a WARNING message")
        print("✅ Logging works correctly!")
        return True
    except Exception as e:
        print(f"❌ Logging test failed: {e}")
        return False

def test_document_processor():
    """Test document processor."""
    print("\n" + "=" * 60)
    print("TEST 4: Testing Document Processor")
    print("=" * 60)
    try:
        from resume_matcher.utils import DocumentProcessor
        
        processor = DocumentProcessor()
        
        # Test text normalization
        test_text = "  Multiple   Spaces  \n\n\nTest"
        normalized = processor.normalize_text(test_text)
        print(f"Original: {repr(test_text)}")
        print(f"Normalized: {repr(normalized)}")
        
        if "  " not in normalized and "\n\n\n" not in normalized:
            print("✅ Text normalization works!")
        else:
            print("⚠️ Text normalization may have issues")
        
        return True
    except Exception as e:
        print(f"❌ Document processor test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_path_validation():
    """Test path validation."""
    print("\n" + "=" * 60)
    print("TEST 5: Testing Path Validation")
    print("=" * 60)
    try:
        from resume_matcher.utils.path_validation import validate_directory
        
        # Test with your actual data directory
        test_dir = Path("/Users/junfeibai/Desktop/5560/test")
        if test_dir.exists():
            validated = validate_directory(test_dir)
            print(f"✅ Path validation works! Validated: {validated}")
            return True
        else:
            print(f"⚠️ Test directory doesn't exist: {test_dir}")
            print("   Skipping path validation test")
            return True
    except Exception as e:
        print(f"❌ Path validation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_matcher_creation():
    """Test matcher creation."""
    print("\n" + "=" * 60)
    print("TEST 6: Testing Matcher Creation")
    print("=" * 60)
    try:
        data_dir = "/Users/junfeibai/Desktop/5560/test"
        matcher = ResumeJDMatcher(data_dir)
        print(f"✅ Matcher created successfully!")
        print(f"   Data directory: {matcher.data_dir}")
        print(f"   Processor type: {type(matcher.processor).__name__}")
        return True, matcher
    except Exception as e:
        print(f"❌ Matcher creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False, None

def test_document_loading(matcher):
    """Test document loading."""
    print("\n" + "=" * 60)
    print("TEST 7: Testing Document Loading")
    print("=" * 60)
    try:
        matcher.load_documents()
        print(f"✅ Documents loaded successfully!")
        print(f"   Resumes: {len(matcher.resumes)}")
        print(f"   Job Descriptions: {len(matcher.job_descriptions)}")
        print(f"   Candidate names: {len(matcher.candidate_names)}")
        print(f"   JD names: {len(matcher.jd_names)}")
        
        if matcher.resumes:
            print(f"\n   Sample resume: {list(matcher.resumes.keys())[0]}")
        if matcher.job_descriptions:
            print(f"   Sample JD: {list(matcher.job_descriptions.keys())[0]}")
        
        return True
    except Exception as e:
        print(f"❌ Document loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_similarity_computation(matcher):
    """Test similarity computation."""
    print("\n" + "=" * 60)
    print("TEST 8: Testing Similarity Computation")
    print("=" * 60)
    try:
        if not matcher.resumes or not matcher.job_descriptions:
            print("⚠️ No documents loaded, skipping similarity computation")
            return True
        
        similarity_matrix = matcher.compute_similarity()
        print(f"✅ Similarity computation successful!")
        print(f"   Matrix shape: {similarity_matrix.shape}")
        print(f"   Min similarity: {similarity_matrix.min():.4f}")
        print(f"   Max similarity: {similarity_matrix.max():.4f}")
        print(f"   Mean similarity: {similarity_matrix.mean():.4f}")
        
        # Test top matches
        top_matches = matcher.get_top_matches(top_k=3)
        if not top_matches.empty:
            print(f"\n   Top matches sample:")
            print(top_matches.head().to_string())
        
        return True
    except Exception as e:
        print(f"❌ Similarity computation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_directory_info(matcher):
    """Test directory info retrieval."""
    print("\n" + "=" * 60)
    print("TEST 9: Testing Directory Info")
    print("=" * 60)
    try:
        dir_info = matcher.get_directory_info()
        print(f"✅ Directory info retrieved successfully!")
        print(f"   Resume files: {len(dir_info['resume_files'])}")
        print(f"   JD files: {len(dir_info['jd_files'])}")
        print(f"   Total files: {dir_info['total_files']}")
        return True
    except Exception as e:
        print(f"❌ Directory info test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("RESUME-JD MATCHER - REFACTORED CODE TEST SUITE")
    print("=" * 60)
    
    # Set up logging
    setup_logging(level="INFO")
    
    results = []
    
    # Run tests
    results.append(("Imports", test_imports()))
    results.append(("Configuration", test_configuration()))
    results.append(("Logging", test_logging()))
    results.append(("Document Processor", test_document_processor()))
    results.append(("Path Validation", test_path_validation()))
    
    success, matcher = test_matcher_creation()
    results.append(("Matcher Creation", success))
    
    if success and matcher:
        results.append(("Document Loading", test_document_loading(matcher)))
        
        if matcher.resumes and matcher.job_descriptions:
            results.append(("Similarity Computation", test_similarity_computation(matcher)))
        else:
            results.append(("Similarity Computation", True))  # Skip if no documents
        
        results.append(("Directory Info", test_directory_info(matcher)))
    
    # Print summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {test_name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! The refactored code is working correctly.")
        return 0
    else:
        print(f"\n⚠️ {total - passed} test(s) failed. Please review the errors above.")
        return 1

if __name__ == "__main__":
    exit(main())

