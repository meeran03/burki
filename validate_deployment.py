#!/usr/bin/env python3
"""
Deployment Validation Script for AWS App Runner
Checks if all necessary files and configurations are in place.
"""

import os
import sys
from pathlib import Path

def check_file_exists(file_path, description):
    """Check if a file exists and print the result."""
    if Path(file_path).exists():
        print(f"✅ {description}: {file_path}")
        return True
    else:
        print(f"❌ {description}: {file_path} - NOT FOUND")
        return False

def check_dockerfile():
    """Validate Dockerfile contents."""
    if not Path("Dockerfile").exists():
        return False
    
    with open("Dockerfile", "r") as f:
        content = f.read()
        
    required_elements = [
        "FROM python:",
        "COPY requirements.txt",
        "RUN pip install",
        "COPY . .",
        "EXPOSE",
        "CMD"
    ]
    
    missing = []
    for element in required_elements:
        if element not in content:
            missing.append(element)
    
    if missing:
        print(f"❌ Dockerfile missing elements: {', '.join(missing)}")
        return False
    else:
        print("✅ Dockerfile validation passed")
        return True

def check_requirements():
    """Check if requirements.txt has essential packages."""
    if not Path("requirements.txt").exists():
        return False
    
    with open("requirements.txt", "r") as f:
        content = f.read()
    
    required_packages = ["fastapi", "uvicorn", "gunicorn", "psycopg2"]
    missing = []
    
    for package in required_packages:
        if package not in content.lower():
            missing.append(package)
    
    if missing:
        print(f"❌ requirements.txt missing packages: {', '.join(missing)}")
        return False
    else:
        print("✅ requirements.txt validation passed")
        return True

def main():
    """Main validation function."""
    print("🔍 Validating AWS App Runner deployment readiness...\n")
    
    all_good = True
    
    # Check essential files
    files_to_check = [
        ("Dockerfile", "Docker configuration"),
        ("apprunner.yaml", "App Runner configuration"),
        ("start.sh", "Startup script"),
        (".dockerignore", "Docker ignore file"),
        ("requirements.txt", "Python dependencies"),
        ("app/main.py", "Main application file"),
        (".env.example", "Environment variables example"),
        (".env.production", "Production environment template"),
        ("AWS_APP_RUNNER_DEPLOYMENT.md", "Deployment guide")
    ]
    
    for file_path, description in files_to_check:
        if not check_file_exists(file_path, description):
            all_good = False
    
    print()
    
    # Validate file contents
    if not check_dockerfile():
        all_good = False
    
    if not check_requirements():
        all_good = False
    
    # Check if start.sh is executable
    if Path("start.sh").exists():
        start_sh_path = Path("start.sh")
        if not os.access(start_sh_path, os.X_OK):
            print("⚠️  start.sh is not executable (this will be fixed by Docker)")
        else:
            print("✅ start.sh is executable")
    
    print("\n" + "="*60)
    
    if all_good:
        print("🎉 All checks passed! Your app is ready for AWS App Runner deployment.")
        print("\nNext steps:")
        print("1. Push your code to GitHub")
        print("2. Set up AWS RDS database")
        print("3. Configure environment variables in AWS App Runner")
        print("4. Deploy using the AWS Console or CLI")
        print("5. Follow the AWS_APP_RUNNER_DEPLOYMENT.md guide")
    else:
        print("⚠️  Some issues found. Please fix them before deployment.")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 