#!/usr/bin/env python
"""
Comprehensive Test Runner for Qdrant RAG System

This script runs all tests and generates a test report with coverage information.
It's designed to be used as part of the CI/CD pipeline or for local testing.
"""
import os
import sys
import subprocess
import datetime
import json
import argparse

def setup_venv():
    """Set up virtual environment if it doesn't exist"""
    if not os.path.exists("test_venv"):
        print("Creating virtual environment...")
        subprocess.run([sys.executable, "-m", "venv", "test_venv"], check=True)
    
    # Determine the activate script path based on OS
    if sys.platform == "win32":
        activate_script = os.path.join("test_venv", "Scripts", "activate")
    else:
        activate_script = os.path.join("test_venv", "bin", "activate")
    
    return activate_script

def install_dependencies(activate_script):
    """Install required dependencies"""
    print("Installing dependencies...")
    
    if sys.platform == "win32":
        # Install only test dependencies (skip problematic requirements.txt)
        test_deps = [
            "pytest", "pytest-asyncio", "pytest-mock", "pytest-cov",
            "httpx", "requests", "PyJWT", "pyyaml", "slowapi"
        ]
        test_deps_cmd = f"{activate_script} && pip install {' '.join(test_deps)}"
        subprocess.run(test_deps_cmd, shell=True, check=True)
    else:
        # For non-Windows platforms
        test_deps = [
            "pytest", "pytest-asyncio", "pytest-mock", "pytest-cov",
            "httpx", "requests", "PyJWT", "pyyaml", "slowapi"
        ]
        test_deps_cmd = f"source {activate_script} && pip install {' '.join(test_deps)}"
        subprocess.run(test_deps_cmd, shell=True, check=True, executable='/bin/bash')

def run_tests(activate_script, test_path, coverage=False, verbose=True):
    """Run tests and return the result"""
    print(f"Running tests in {test_path}...")
    
    cmd_parts = ["python", "-m", "pytest"]
    
    if verbose:
        cmd_parts.append("-v")
    
    if coverage:
        cmd_parts.extend(["--cov=Scripts", "--cov-report=xml", "--cov-report=term"])
    
    cmd_parts.append(test_path)
    
    if sys.platform == "win32":
        cmd = f"{activate_script} && {' '.join(cmd_parts)}"
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    else:
        cmd = f"source {activate_script} && {' '.join(cmd_parts)}"
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, executable='/bin/bash')
    
    print(result.stdout)
    
    if result.stderr:
        print("ERRORS:")
        print(result.stderr)
    
    return result.returncode == 0

def generate_test_report(test_results, output_path):
    """Generate a test report"""
    print("Generating test report...")
    
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Read code coverage if available
    coverage_data = None
    if os.path.exists("coverage.xml"):
        import xml.etree.ElementTree as ET
        try:
            tree = ET.parse("coverage.xml")
            root = tree.getroot()
            coverage_data = {
                "line_rate": float(root.get("line-rate", 0)) * 100,
                "branch_rate": float(root.get("branch-rate", 0)) * 100,
                "complexity": root.get("complexity", "N/A")
            }
        except Exception as e:
            print(f"Error parsing coverage data: {e}")
    
    # Create report content
    report = {
        "title": "Automated Test Report",
        "timestamp": timestamp,
        "test_results": test_results,
        "coverage": coverage_data
    }
    
    # Write the report to a JSON file
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)
    
    print(f"Test report generated: {output_path}")
      # Create a Markdown version of the report
    md_path = os.path.splitext(output_path)[0] + ".md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(f"# Automated Test Report\n\n")
        f.write(f"Generated: {timestamp}\n\n")
        
        f.write("## Test Results\n\n")
        f.write("| Test Category | Status | Details |\n")
        f.write("|--------------|--------|--------|\n")
        for category, result in test_results.items():
            status = "PASSED" if result["passed"] else "FAILED"
            f.write(f"| {category} | {status} | {result.get('details', '')} |\n")
        
        if coverage_data:
            f.write("\n## Code Coverage\n\n")
            f.write(f"- Line Coverage: {coverage_data['line_rate']:.2f}%\n")
            f.write(f"- Branch Coverage: {coverage_data['branch_rate']:.2f}%\n")
            f.write(f"- Complexity: {coverage_data['complexity']}\n")
    
    print(f"Markdown report generated: {md_path}")

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Run tests for Qdrant RAG System")
    parser.add_argument("--no-coverage", action="store_true", help="Disable coverage reporting")
    parser.add_argument("--no-verbose", action="store_true", help="Disable verbose output")
    parser.add_argument("--output", help="Output file for test report")
    parser.add_argument("--unit-only", action="store_true", help="Run only unit tests")
    parser.add_argument("--integration-only", action="store_true", help="Run only integration tests")
    parser.add_argument("--e2e-only", action="store_true", help="Run only end-to-end tests")
    parser.add_argument("--performance-only", action="store_true", help="Run only performance tests")
    parser.add_argument("--vector-search-only", action="store_true", help="Run only vector search tests")
    return parser.parse_args()

def main():
    """Main entry point"""
    args = parse_args()
    
    # Create reports directory if it doesn't exist
    os.makedirs("reports", exist_ok=True)
    output_path = os.path.join("reports", args.output) if args.output else "reports/test_report.json"
    
    coverage = not args.no_coverage
    verbose = not args.no_verbose
    
    # Setup test environment
    activate_script = setup_venv()
    install_dependencies(activate_script)
    
    # Define test categories and paths
    test_categories = {
        "API Hub Unit Tests": "tests/simplified_api_hub_test.py",
        "API Hub Integration Tests": "tests/integration_api_hub_test.py",
        "End-to-End Tests": "tests/e2e_api_hub_test.py",
        "Vector Search Tests": "tests/test_vector_search.py",
        "Performance Tests": "tests/test_performance.py"
    }
    
    # Filter test categories based on command line arguments
    if args.unit_only:
        test_categories = {k: v for k, v in test_categories.items() if "Unit" in k}
    elif args.integration_only:
        test_categories = {k: v for k, v in test_categories.items() if "Integration" in k}
    elif args.e2e_only:
        test_categories = {k: v for k, v in test_categories.items() if "End-to-End" in k}
    elif args.performance_only:
        test_categories = {k: v for k, v in test_categories.items() if "Performance" in k}
    elif args.vector_search_only:
        test_categories = {k: v for k, v in test_categories.items() if "Vector Search" in k}
    
    # Run tests and collect results
    test_results = {}
    all_passed = True
    
    for category, path in test_categories.items():
        print(f"\n{'=' * 80}\n{category}\n{'=' * 80}\n")
        passed = run_tests(activate_script, path, coverage=(coverage and "Performance" not in category), verbose=verbose)
        test_results[category] = {"passed": passed}
        if not passed:
            all_passed = False
    
    # Generate test report
    generate_test_report(test_results, output_path)
    
    print("\nSummary:")
    for category, result in test_results.items():
        status = "PASSED" if result["passed"] else "FAILED"
        print(f"  {category}: {status}")
    
    if all_passed:
        print("\nAll tests passed! 🎉")
        return 0
    else:
        print("\nSome tests failed. Check the report for details.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
