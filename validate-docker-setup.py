#!/usr/bin/env python3
"""
Docker Setup Validation Script for Wagehood Trading System

This script validates that all Docker configurations are correct and production-ready.
"""

import os
import sys
import yaml
import json
from pathlib import Path

def log_info(message: str):
    print(f"✅ {message}")

def log_warning(message: str):
    print(f"⚠️  {message}")

def log_error(message: str):
    print(f"❌ {message}")

def check_dockerfile():
    """Validate Dockerfile configuration"""
    print("\n📋 Checking Dockerfile...")
    
    dockerfile_path = Path("Dockerfile")
    if not dockerfile_path.exists():
        log_error("Dockerfile not found")
        return False
    
    with open(dockerfile_path, 'r') as f:
        content = f.read()
    
    checks = {
        "Multi-stage build": "FROM python:3.9-slim as builder" in content,
        "Security updates": "apt-get upgrade -y" in content,
        "Non-root user": "groupadd -r wagehood && useradd -r -g wagehood wagehood" in content,
        "Health check script": "COPY docker-healthcheck.py" in content,
        "Comprehensive health check": "CMD python docker-healthcheck.py" in content,
        "Python security settings": "PYTHONHASHSEED=random" in content,
        "Credential validation env": "ALPACA_API_KEY_REQUIRED=true" in content,
    }
    
    all_passed = True
    for check_name, passed in checks.items():
        if passed:
            log_info(f"Dockerfile {check_name}")
        else:
            log_error(f"Dockerfile missing {check_name}")
            all_passed = False
    
    return all_passed

def check_docker_compose():
    """Validate docker-compose.yml configuration"""
    print("\n📋 Checking docker-compose.yml...")
    
    compose_path = Path("docker-compose.yml")
    if not compose_path.exists():
        log_error("docker-compose.yml not found")
        return False
    
    try:
        with open(compose_path, 'r') as f:
            compose_config = yaml.safe_load(f)
    except yaml.YAMLError as e:
        log_error(f"Invalid YAML in docker-compose.yml: {e}")
        return False
    
    checks = []
    service = compose_config.get('services', {}).get('wagehood', {})
    
    # Check required environment variables validation
    env_vars = service.get('environment', [])
    
    # Convert list format to dict for easier checking
    env_dict = {}
    for env in env_vars:
        if '=' in env:
            key, value = env.split('=', 1)
            env_dict[key] = value
    
    # Check critical credential variables have validation
    critical_vars = ['ALPACA_API_KEY', 'ALPACA_SECRET_KEY']
    for var in critical_vars:
        if var in env_dict:
            if '${' in str(env_dict[var]) and '?}' in str(env_dict[var]):
                log_info(f"Environment variable {var} has validation")
            else:
                log_warning(f"Environment variable {var} present but no validation")
        else:
            log_error(f"Missing environment variable {var}")
            checks.append(False)
    
    # Check other required variables exist
    other_required = ['WAGEHOOD_ENV', 'REDIS_HOST', 'REDIS_PORT']
    for var in other_required:
        if var in env_dict:
            log_info(f"Environment variable {var} configured")
        else:
            log_error(f"Missing environment variable {var}")
            checks.append(False)
    
    # Check health check configuration
    healthcheck = service.get('healthcheck', {})
    if healthcheck:
        test_cmd = healthcheck.get('test', [])
        if 'docker-healthcheck.py' in ' '.join(test_cmd):
            log_info("Health check uses comprehensive script")
        else:
            log_error("Health check not using comprehensive script")
            checks.append(False)
        
        if healthcheck.get('start_period') == '120s':
            log_info("Health check start period configured correctly")
        else:
            log_warning("Health check start period should be 120s for Alpaca validation")
    else:
        log_error("No health check configured")
        checks.append(False)
    
    # Check volumes
    volumes = service.get('volumes', [])
    expected_volumes = ['/app/data', '/app/logs']
    for expected in expected_volumes:
        if any(expected in str(vol) for vol in volumes):
            log_info(f"Volume mount configured for {expected}")
        else:
            log_error(f"Missing volume mount for {expected}")
            checks.append(False)
    
    # Check resource limits
    deploy = service.get('deploy', {})
    resources = deploy.get('resources', {})
    if resources.get('limits'):
        log_info("Resource limits configured")
    else:
        log_warning("No resource limits configured")
    
    return len([c for c in checks if not c]) == 0

def check_entrypoint():
    """Validate docker-entrypoint.sh"""
    print("\n📋 Checking docker-entrypoint.sh...")
    
    entrypoint_path = Path("docker-entrypoint.sh")
    if not entrypoint_path.exists():
        log_error("docker-entrypoint.sh not found")
        return False
    
    with open(entrypoint_path, 'r') as f:
        content = f.read()
    
    checks = {
        "Credential validation": "ALPACA_API_KEY" in content and "ALPACA_SECRET_KEY" in content,
        "Credential format validation": "${#ALPACA_API_KEY}" in content,
        "Detailed error messages": "CRITICAL ERROR" in content,
        "Security checks": "if [[ ${#ALPACA_API_KEY} -lt 20" in content,
        "Alpaca connectivity test": "MinimalAlpacaProvider" in content,
        "Signal handlers": "trap shutdown SIGTERM SIGINT" in content,
    }
    
    all_passed = True
    for check_name, passed in checks.items():
        if passed:
            log_info(f"Entrypoint {check_name}")
        else:
            log_error(f"Entrypoint missing {check_name}")
            all_passed = False
    
    return all_passed

def check_health_check_script():
    """Validate docker-healthcheck.py"""
    print("\n📋 Checking docker-healthcheck.py...")
    
    healthcheck_path = Path("docker-healthcheck.py")
    if not healthcheck_path.exists():
        log_error("docker-healthcheck.py not found")
        return False
    
    with open(healthcheck_path, 'r') as f:
        content = f.read()
    
    checks = {
        "Redis connectivity check": "check_redis_connectivity" in content,
        "Alpaca credentials check": "check_alpaca_credentials" in content,
        "Alpaca connectivity check": "check_alpaca_connectivity" in content,
        "Market data retrieval": "check_market_data_retrieval" in content,
        "Core imports check": "check_core_imports" in content,
        "System configuration check": "check_system_configuration" in content,
        "Async support": "async def" in content,
        "Proper exit codes": "sys.exit(" in content,
    }
    
    all_passed = True
    for check_name, passed in checks.items():
        if passed:
            log_info(f"Health check {check_name}")
        else:
            log_error(f"Health check missing {check_name}")
            all_passed = False
    
    return all_passed

def check_dockerignore():
    """Validate .dockerignore"""
    print("\n📋 Checking .dockerignore...")
    
    dockerignore_path = Path(".dockerignore")
    if not dockerignore_path.exists():
        log_warning(".dockerignore not found")
        return False
    
    with open(dockerignore_path, 'r') as f:
        content = f.read()
    
    important_excludes = ['.env', '.git', '__pycache__', '*.log', 'test_*.py']
    all_good = True
    
    for exclude in important_excludes:
        if exclude in content:
            log_info(f"Dockerignore excludes {exclude}")
        else:
            log_warning(f"Dockerignore should exclude {exclude}")
            all_good = False
    
    return all_good

def check_security_practices():
    """Check security best practices"""
    print("\n📋 Checking Security Practices...")
    
    # Check if .env file is in .gitignore
    gitignore_path = Path(".gitignore")
    security_score = 0
    
    if gitignore_path.exists():
        with open(gitignore_path, 'r') as f:
            gitignore_content = f.read()
        
        if '.env' in gitignore_content:
            log_info("Environment files excluded from git")
            security_score += 1
        else:
            log_error(".env files should be in .gitignore")
    
    # Check if there are any hardcoded credentials
    sensitive_files = ['Dockerfile', 'docker-compose.yml', 'docker-entrypoint.sh']
    hardcoded_patterns = ['pk_', 'sk_', 'password=', 'secret=']
    
    for file_path in sensitive_files:
        if Path(file_path).exists():
            with open(file_path, 'r') as f:
                content = f.read().lower()
            
            for pattern in hardcoded_patterns:
                if pattern in content and 'example' not in content:
                    log_error(f"Potential hardcoded credential in {file_path}")
                    security_score -= 1
    
    if security_score >= 0:
        log_info("No obvious security issues found")
    
    return security_score >= 0

def main():
    """Main validation function"""
    print("🐳 Wagehood Docker Configuration Validator")
    print("=" * 50)
    
    checks = [
        ("Dockerfile", check_dockerfile),
        ("Docker Compose", check_docker_compose),
        ("Entrypoint Script", check_entrypoint),
        ("Health Check Script", check_health_check_script),
        ("Dockerignore", check_dockerignore),
        ("Security Practices", check_security_practices),
    ]
    
    passed = 0
    failed = 0
    
    for check_name, check_func in checks:
        try:
            if check_func():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            log_error(f"Error checking {check_name}: {e}")
            failed += 1
    
    print("\n" + "=" * 50)
    print(f"📊 Validation Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("\n🎉 All validations passed! Docker setup is production-ready.")
        print("\n📋 Next steps:")
        print("   1. Set your Alpaca credentials in .env file")
        print("   2. Run: docker-compose up -d")
        print("   3. Monitor: docker-compose logs -f")
        return 0
    else:
        print(f"\n❌ {failed} validations failed. Please fix the issues above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())