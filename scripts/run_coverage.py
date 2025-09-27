#!/usr/bin/env python3
"""
Script to run tests with coverage and generate coverage report.
"""

import subprocess
import sys
import os
from pathlib import Path


def run_coverage():
    """Run tests with coverage reporting."""
    print("Running tests with coverage...")

    # Change to project root
    project_root = Path(__file__).parent.parent
    os.chdir(project_root)

    # Run tests with coverage
    cmd = [
        "python",
        "-m",
        "pytest",
        "--cov=.",
        "--cov-report=term-missing",
        "--cov-report=html:htmlcov",
        "--cov-report=xml:coverage.xml",
        "tests/test_chains.py",
        "tests/test_knowledge_bases.py",
        "tests/test_workflows.py",
        "-v",
    ]

    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("✅ Tests completed successfully!")
        print("\nCoverage report generated:")
        print("- HTML report: htmlcov/index.html")
        print("- XML report: coverage.xml")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Tests failed with exit code {e.returncode}")
        print("STDOUT:", e.stdout)
        print("STDERR:", e.stderr)
        return False


def generate_coverage_badge():
    """Generate a simple coverage badge."""
    try:
        # Read coverage from XML file
        import xml.etree.ElementTree as ET

        tree = ET.parse("coverage.xml")
        root = tree.getroot()

        # Get coverage percentage
        coverage = float(root.get("line-rate", "0")) * 100

        # Generate badge
        badge_color = (
            "green" if coverage >= 80 else "yellow" if coverage >= 60 else "red"
        )
        badge_url = (
            f"https://img.shields.io/badge/coverage-{coverage:.1f}%25-{badge_color}"
        )

        print(f"\n📊 Coverage: {coverage:.1f}%")
        print(f"🛡️ Badge URL: {badge_url}")

        return coverage
    except Exception as e:
        print(f"❌ Failed to generate coverage badge: {e}")
        return None


if __name__ == "__main__":
    print("🧪 KERAG Test Coverage Runner")
    print("=" * 40)

    success = run_coverage()

    if success:
        coverage = generate_coverage_badge()
        if coverage is not None:
            print(f"\n✅ Coverage analysis complete: {coverage:.1f}%")
        else:
            print("\n⚠️ Coverage analysis completed but badge generation failed")
    else:
        print("\n❌ Coverage analysis failed")
        sys.exit(1)

