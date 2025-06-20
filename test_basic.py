#!/usr/bin/env python3
"""
Basic Test Suite for AI Co-Scientist Core Components

Tests the core functionality without requiring external API keys or dependencies
"""

import sys
import os
sys.path.insert(0, '.')

def test_data_structures():
    """Test core data structures"""
    print("🧪 Testing Core Data Structures...")
    
    try:
        from core.data_structures import (
            GenerationState, ReviewCriteria, HypothesisReview, 
            SupervisorConfig, EvolutionStrategy, create_timestamp,
            create_hypothesis_id, validate_score
        )
        
        # Test GenerationState
        gen_state = GenerationState(
            research_goal="Test automated hypothesis generation",
            constraints=["Must be testable", "Must be novel"]
        )
        assert gen_state.research_goal == "Test automated hypothesis generation"
        assert len(gen_state.constraints) == 2
        assert gen_state.status == "initialized"
        print("  ✅ GenerationState creation and attributes")
        
        # Test SupervisorConfig
        config = SupervisorConfig(
            research_goal="Test research coordination",
            max_cycles=5,
            evolution_every=2
        )
        assert config.research_goal == "Test research coordination"
        assert config.max_cycles == 5
        print("  ✅ SupervisorConfig creation and attributes")
        
        # Test utility functions
        timestamp = create_timestamp()
        assert isinstance(timestamp, str)
        assert len(timestamp) > 10  # ISO format should be longer
        print("  ✅ Timestamp creation")
        
        hyp_id = create_hypothesis_id("test")
        assert hyp_id.startswith("test_")
        print("  ✅ Hypothesis ID generation")
        
        # Test score validation
        assert validate_score(5.0) == 5.0
        assert validate_score(-1.0) == 1.0  # Should clamp to min
        assert validate_score(15.0) == 10.0  # Should clamp to max
        print("  ✅ Score validation")
        
        print("✅ All data structure tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Data structure test failed: {e}")
        return False

def test_config_structure():
    """Test configuration structure without external dependencies"""
    print("\n🧪 Testing Configuration Structure...")
    
    try:
        import os
        
        # Set minimal environment for testing
        os.environ['PUBMED_EMAIL'] = 'test@example.com'
        os.environ['GOOGLE_API_KEY'] = 'test_key_for_structure_validation'
        
        from core.config import (
            DatabaseConfig, ModelConfig, AgentConfig, SystemConfig,
            get_env, get_int_env, get_float_env, get_bool_env
        )
        
        # Test individual config classes
        db_config = DatabaseConfig(
            pubmed_email="test@example.com",
            rate_limit_delay=1.0,
            max_results_per_source=3
        )
        assert db_config.pubmed_email == "test@example.com"
        assert db_config.rate_limit_delay == 1.0
        print("  ✅ DatabaseConfig creation")
        
        model_config = ModelConfig(
            gemini_api_key="test_key",
            gemini_model="gemini-2.0-flash-exp"
        )
        assert model_config.gemini_model == "gemini-2.0-flash-exp"
        print("  ✅ ModelConfig creation")
        
        # Test environment variable helpers
        test_val = get_env('PUBMED_EMAIL', 'default')
        assert test_val == 'test@example.com'
        print("  ✅ Environment variable retrieval")
        
        test_int = get_int_env('NONEXISTENT_INT', 42)
        assert test_int == 42
        print("  ✅ Integer environment variable with default")
        
        test_float = get_float_env('NONEXISTENT_FLOAT', 3.14)
        assert test_float == 3.14
        print("  ✅ Float environment variable with default")
        
        test_bool = get_bool_env('NONEXISTENT_BOOL', True)
        assert test_bool == True
        print("  ✅ Boolean environment variable with default")
        
        print("✅ All configuration tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False

def test_text_processing_basics():
    """Test basic text processing without ML dependencies"""
    print("\n🧪 Testing Basic Text Processing...")
    
    try:
        # Import only the analyzer which doesn't need ML libs
        from utils.text_processing import TextAnalyzer
        
        analyzer = TextAnalyzer()
        
        # Test keyword extraction
        test_texts = [
            "Machine learning approaches for drug discovery in cancer research",
            "Deep learning models for predicting protein-drug interactions", 
            "Artificial intelligence in personalized medicine and therapy"
        ]
        
        keywords = analyzer.extract_keywords(test_texts, top_k=5)
        assert isinstance(keywords, list)
        assert len(keywords) > 0
        print(f"  ✅ Keyword extraction: found {len(keywords)} keywords")
        
        # Test text similarity
        similarity = analyzer.compute_text_similarity(test_texts[0], test_texts[1])
        assert 0 <= similarity <= 1
        print(f"  ✅ Text similarity: {similarity:.3f}")
        
        # Test sentence extraction
        long_text = "This is the first sentence. This is the second sentence with more words. This is a very short one."
        sentences = analyzer.extract_sentences(long_text)
        assert isinstance(sentences, list)
        assert len(sentences) >= 2
        print(f"  ✅ Sentence extraction: found {len(sentences)} sentences")
        
        print("✅ Basic text processing tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Text processing test failed: {e}")
        return False

def test_file_integrity():
    """Test file integrity and content validation"""
    print("\n🧪 Testing File Integrity...")
    
    try:
        # Check requirements.txt
        with open('requirements.txt', 'r') as f:
            requirements = f.read()
            assert 'langchain-community' in requirements
            assert 'sentence-transformers' in requirements
            assert 'scikit-learn' in requirements
        print("  ✅ requirements.txt contains expected dependencies")
        
        # Check .env.example
        with open('.env.example', 'r') as f:
            env_example = f.read()
            assert 'GOOGLE_API_KEY' in env_example
            assert 'PUBMED_EMAIL' in env_example
            assert 'GEMINI_MODEL' in env_example
        print("  ✅ .env.example contains required environment variables")
        
        # Check .gitignore
        with open('.gitignore', 'r') as f:
            gitignore = f.read()
            assert '.env' in gitignore
            assert '__pycache__' in gitignore
            assert '*.log' in gitignore
        print("  ✅ .gitignore contains essential exclusions")
        
        # Check core module files exist and have content
        core_files = ['core/config.py', 'core/data_structures.py']
        for file_path in core_files:
            with open(file_path, 'r') as f:
                content = f.read()
                assert len(content) > 1000  # Should have substantial content
                assert 'class' in content  # Should contain class definitions
        print("  ✅ Core module files have substantial content")
        
        print("✅ All file integrity tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ File integrity test failed: {e}")
        return False

def run_all_tests():
    """Run all basic tests"""
    print("🚀 AI CO-SCIENTIST BASIC TEST SUITE")
    print("=" * 60)
    
    tests = [
        test_data_structures,
        test_config_structure,
        test_text_processing_basics,
        test_file_integrity
    ]
    
    passed = 0
    total = len(tests)
    
    for test_func in tests:
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"❌ Test {test_func.__name__} crashed: {e}")
    
    print("\n" + "=" * 60)
    print(f"📊 TEST RESULTS: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED! Core functionality is working correctly.")
        print("\n💡 Next steps:")
        print("   1. Install dependencies: pip install -r requirements.txt")
        print("   2. Copy .env.example to .env and add your API keys")
        print("   3. Run the full system integration tests")
        return True
    else:
        print("⚠️  Some tests failed. Please check the implementation.")
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)