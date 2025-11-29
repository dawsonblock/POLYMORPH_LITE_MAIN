import pytest
import os
import re

SUSPICIOUS_PATTERNS = [
    r"AWS_ACCESS_KEY_ID\s*=\s*['\"]?AKIA[0-9A-Z]{16}['\"]?",
    r"AWS_SECRET_ACCESS_KEY\s*=\s*['\"]?[A-Za-z0-9/+=]{40}['\"]?",
    r"password\s*=\s*['\"](?!(\$\{?|\{\{)).+['\"]",  # password = "literal" (not variable)
    r"secret_key\s*=\s*['\"](?!(\$\{?|\{\{)).+['\"]",
]

def scan_file(filepath):
    """Scan a file for suspicious patterns."""
    issues = []
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
            for pattern in SUSPICIOUS_PATTERNS:
                if re.search(pattern, content, re.IGNORECASE):
                    # Exclude known test files or CI configs with dummy values
                    if "test" in filepath or "ci.yml" in filepath:
                        continue
                    issues.append(f"Found suspicious pattern '{pattern}' in {filepath}")
    except UnicodeDecodeError:
        pass # Skip binary files
    return issues

def test_no_hardcoded_secrets():
    """Scan infra and github dirs for secrets."""
    root_dirs = ["infra", ".github"]
    all_issues = []
    
    base_path = os.getcwd()
    
    for d in root_dirs:
        path = os.path.join(base_path, d)
        if not os.path.exists(path):
            continue
            
        for root, _, files in os.walk(path):
            for file in files:
                if file.endswith(('.yml', '.yaml', '.tf', '.py', '.sh')):
                    filepath = os.path.join(root, file)
                    all_issues.extend(scan_file(filepath))
                    
    assert len(all_issues) == 0, f"Found potential hardcoded secrets: {all_issues}"
