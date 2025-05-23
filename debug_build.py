#!/usr/bin/env python3
"""
Debug script to test application startup without running the full server.
This helps identify import and initialization issues before deployment.
"""

import sys
import os
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

def test_imports():
    """Test critical imports that might fail during build."""
    try:
        logger.info("Testing imports...")
        
        # Test dotenv loading
        from dotenv import load_dotenv
        load_dotenv()
        logger.info("✓ dotenv loaded successfully")
        
        # Test database imports
        from app.db.database import Base, engine, SessionLocal
        logger.info("✓ Database imports successful")
        
        # Test FastAPI imports
        from fastapi import FastAPI
        logger.info("✓ FastAPI imports successful")
        
        # Test service imports
        from app.services.deepgram_service import DeepgramService
        from app.services.tts_service import TTSService
        logger.info("✓ Service imports successful")
        
        # Test main app import
        from app.main import app
        logger.info("✓ Main app import successful")
        
        return True
        
    except Exception as e:
        logger.error(f"Import failed: {e}", exc_info=True)
        return False

def test_config():
    """Test configuration loading."""
    try:
        logger.info("Testing configuration...")
        
        # Check critical environment variables
        required_vars = [
            "TWILIO_ACCOUNT_SID",
            "TWILIO_AUTH_TOKEN", 
            "DEEPGRAM_API_KEY",
            "OPENAI_API_KEY",
            "ELEVENLABS_API_KEY"
        ]
        
        missing_vars = []
        for var in required_vars:
            if not os.getenv(var):
                missing_vars.append(var)
                
        if missing_vars:
            logger.warning(f"Missing environment variables: {missing_vars}")
        else:
            logger.info("✓ All required environment variables present")
            
        # Test database configuration
        from app.db.database import DATABASE_URL
        logger.info(f"✓ Database URL configured: {DATABASE_URL.replace(os.getenv('DB_PASSWORD', ''), '***')}")
        
        return True
        
    except Exception as e:
        logger.error(f"Configuration test failed: {e}", exc_info=True)
        return False

def main():
    """Run all tests."""
    logger.info("Starting build debug tests...")
    
    tests = [
        ("Imports", test_imports),
        ("Configuration", test_config),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        logger.info(f"\n--- Testing {test_name} ---")
        if test_func():
            logger.info(f"✓ {test_name} test PASSED")
            passed += 1
        else:
            logger.error(f"✗ {test_name} test FAILED")
    
    logger.info(f"\n--- Results ---")
    logger.info(f"Passed: {passed}/{total}")
    
    if passed == total:
        logger.info("🎉 All tests passed! Application should build successfully.")
        sys.exit(0)
    else:
        logger.error("❌ Some tests failed. Fix these issues before deploying.")
        sys.exit(1)

if __name__ == "__main__":
    main() 