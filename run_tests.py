#!/usr/bin/env python3
"""
Test Runner for AI Co-Scientist

This script provides a convenient way to run all tests for the AI Co-Scientist system.
"""

import sys
import subprocess
import argparse
from pathlib import Path

def run_basic_tests():
    """Run basic tests that don't require external dependencies."""
    print("🧪 Running Basic Tests...")
    print("=" * 50)
    
    try:
        result = subprocess.run([sys.executable, "test_basic.py"], 
                              capture_output=True, text=True, timeout=60)
        
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        
        return result.returncode == 0
        
    except subprocess.TimeoutExpired:
        print("❌ Basic tests timed out")
        return False
    except Exception as e:
        print(f"❌ Error running basic tests: {e}")
        return False

def run_unit_tests():
    """Run unit tests using pytest."""
    print("\n🔬 Running Unit Tests...")
    print("=" * 50)
    
    try:
        result = subprocess.run([
            sys.executable, "-m", "pytest", 
            "tests/unit/", 
            "-v", 
            "--tb=short",
            "--timeout=300"  # 5 minute timeout per test
        ], capture_output=True, text=True, timeout=1800)  # 30 minute total timeout
        
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        
        return result.returncode == 0
        
    except subprocess.TimeoutExpired:
        print("❌ Unit tests timed out")
        return False
    except Exception as e:
        print(f"❌ Error running unit tests: {e}")
        return False

def run_integration_tests():
    """Run integration tests using pytest."""
    print("\n🔗 Running Integration Tests...")
    print("=" * 50)
    
    try:
        result = subprocess.run([
            sys.executable, "-m", "pytest", 
            "tests/integration/", 
            "-v", 
            "--tb=short",
            "--timeout=600"  # 10 minute timeout per test
        ], capture_output=True, text=True, timeout=3600)  # 1 hour total timeout
        
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        
        return result.returncode == 0
        
    except subprocess.TimeoutExpired:
        print("❌ Integration tests timed out")
        return False
    except Exception as e:
        print(f"❌ Error running integration tests: {e}")
        return False

def run_specific_test(test_path):
    """Run a specific test file."""
    print(f"\n🎯 Running Specific Test: {test_path}")
    print("=" * 50)
    
    try:
        if test_path.endswith('.py') and not test_path.startswith('test_'):
            # Direct Python execution
            result = subprocess.run([sys.executable, test_path], 
                                  capture_output=True, text=True, timeout=300)
        else:
            # Pytest execution
            result = subprocess.run([
                sys.executable, "-m", "pytest", 
                test_path, 
                "-v", 
                "--tb=short"
            ], capture_output=True, text=True, timeout=600)
        
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        
        return result.returncode == 0
        
    except subprocess.TimeoutExpired:
        print(f"❌ Test {test_path} timed out")
        return False
    except Exception as e:
        print(f"❌ Error running test {test_path}: {e}")
        return False

def check_dependencies():
    """Check if required dependencies are available."""
    print("🔍 Checking Dependencies...")
    
    required_packages = [
        "pytest",
        "langchain-community", 
        "sentence-transformers",
        "scikit-learn",
        "numpy"
    ]
    
    missing = []
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            print(f"  ✅ {package}")
        except ImportError:
            print(f"  ❌ {package} (missing)")
            missing.append(package)
    
    if missing:
        print(f"\n⚠️  Missing packages: {', '.join(missing)}")
        print("   Install with: pip install -r requirements.txt")
        return False
    
    print("✅ All dependencies available")
    return True

def main():
    """Main test runner function."""
    parser = argparse.ArgumentParser(
        description="Test runner for AI Co-Scientist system",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_tests.py                    # Run all tests
  python run_tests.py --basic           # Run only basic tests
  python run_tests.py --unit            # Run only unit tests
  python run_tests.py --integration     # Run only integration tests
  python run_tests.py --test test_basic.py  # Run specific test
        """
    )
    
    parser.add_argument("--basic", action="store_true", 
                       help="Run only basic tests (no external dependencies)")
    parser.add_argument("--unit", action="store_true",
                       help="Run only unit tests")
    parser.add_argument("--integration", action="store_true",
                       help="Run only integration tests") 
    parser.add_argument("--test", type=str,
                       help="Run specific test file")
    parser.add_argument("--skip-deps", action="store_true",
                       help="Skip dependency checking")
    
    args = parser.parse_args()
    
    print("🚀 AI CO-SCIENTIST TEST RUNNER")
    print("=" * 60)
    
    # Check dependencies unless skipped
    if not args.skip_deps and not args.basic:
        if not check_dependencies():
            print("\n💡 Try running with --basic for tests without external dependencies")
            sys.exit(1)
    
    results = []
    
    # Run specific test
    if args.test:
        success = run_specific_test(args.test)
        results.append(("Specific Test", success))
    
    # Run basic tests
    elif args.basic or (not args.unit and not args.integration and not args.test):
        success = run_basic_tests()
        results.append(("Basic Tests", success))
        
        if not args.basic:  # If running all tests, continue with others
            if not args.skip_deps and check_dependencies():
                success = run_unit_tests()
                results.append(("Unit Tests", success))
                
                success = run_integration_tests()
                results.append(("Integration Tests", success))
    
    # Run unit tests only
    elif args.unit:
        success = run_unit_tests()
        results.append(("Unit Tests", success))
    
    # Run integration tests only
    elif args.integration:
        success = run_integration_tests()
        results.append(("Integration Tests", success))
    
    # Print summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    
    total_passed = 0
    total_tests = len(results)
    
    for test_name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{test_name:.<50} {status}")
        if success:
            total_passed += 1
    
    print("-" * 60)
    print(f"TOTAL: {total_passed}/{total_tests} test suites passed")
    
    if total_passed == total_tests:
        print("\n🎉 ALL TESTS PASSED!")
        sys.exit(0)
    else:
        print(f"\n⚠️  {total_tests - total_passed} test suite(s) failed")
        sys.exit(1)

if __name__ == "__main__":
    main()