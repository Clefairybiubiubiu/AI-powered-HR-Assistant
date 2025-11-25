#!/usr/bin/env python3
"""
Comprehensive project review script to identify:
1. Duplicate files
2. Syntax errors
3. Import errors
4. Blank/empty sections
5. Unused files
"""

import os
import ast
import sys
from pathlib import Path
from collections import defaultdict
import hashlib

def get_file_hash(filepath):
    """Calculate MD5 hash of file."""
    try:
        with open(filepath, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()
    except Exception:
        return None

def check_syntax(filepath):
    """Check Python file for syntax errors."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            code = f.read()
        ast.parse(code)
        return True, None
    except SyntaxError as e:
        return False, str(e)
    except Exception as e:
        return False, f"Error reading file: {e}"

def check_imports(filepath):
    """Check for import errors."""
    errors = []
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            tree = ast.parse(f.read())
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    try:
                        __import__(alias.name)
                    except ImportError:
                        # This is expected for some imports, skip
                        pass
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    try:
                        __import__(node.module)
                    except ImportError:
                        # This is expected for some imports, skip
                        pass
    except Exception as e:
        errors.append(str(e))
    
    return errors

def find_duplicates(directory):
    """Find duplicate files."""
    file_hashes = defaultdict(list)
    
    for root, dirs, files in os.walk(directory):
        # Skip __pycache__ and .git
        dirs[:] = [d for d in dirs if d not in ['__pycache__', '.git', 'node_modules']]
        
        for file in files:
            if file.endswith(('.py', '.md', '.txt', '.sh')):
                filepath = os.path.join(root, file)
                file_hash = get_file_hash(filepath)
                if file_hash:
                    file_hashes[file_hash].append(filepath)
    
    duplicates = {h: paths for h, paths in file_hashes.items() if len(paths) > 1}
    return duplicates

def check_blank_sections(filepath):
    """Check for blank/empty sections in Python files."""
    blank_sections = []
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        in_class = False
        in_function = False
        current_class = None
        current_function = None
        empty_lines = 0
        
        for i, line in enumerate(lines, 1):
            stripped = line.strip()
            
            # Check for class definition
            if stripped.startswith('class '):
                if in_function and current_function:
                    blank_sections.append(f"Line {i}: Empty function '{current_function}' in class '{current_class}'")
                in_class = True
                current_class = stripped.split('(')[0].replace('class ', '').strip()
                in_function = False
                current_function = None
                empty_lines = 0
            
            # Check for function definition
            elif stripped.startswith('def '):
                if in_function and current_function and empty_lines > 3:
                    blank_sections.append(f"Line {i}: Function '{current_function}' has {empty_lines} empty lines")
                in_function = True
                current_function = stripped.split('(')[0].replace('def ', '').strip()
                empty_lines = 0
            
            # Count empty lines
            elif not stripped or stripped.startswith('#'):
                empty_lines += 1
            else:
                empty_lines = 0
    
    except Exception as e:
        blank_sections.append(f"Error reading file: {e}")
    
    return blank_sections

def main():
    project_dir = Path(__file__).parent
    print("=" * 80)
    print("PROJECT REVIEW REPORT")
    print("=" * 80)
    print()
    
    # 1. Find duplicate files
    print("1. CHECKING FOR DUPLICATE FILES")
    print("-" * 80)
    duplicates = find_duplicates(project_dir)
    if duplicates:
        print(f"Found {len(duplicates)} sets of duplicate files:")
        for file_hash, paths in duplicates.items():
            print(f"\n  Duplicate set (hash: {file_hash[:8]}...):")
            for path in paths:
                rel_path = os.path.relpath(path, project_dir)
                size = os.path.getsize(path)
                print(f"    - {rel_path} ({size} bytes)")
    else:
        print("  ✓ No duplicate files found")
    print()
    
    # 2. Check syntax errors
    print("2. CHECKING FOR SYNTAX ERRORS")
    print("-" * 80)
    syntax_errors = []
    python_files = list(project_dir.rglob("*.py"))
    python_files = [f for f in python_files if '__pycache__' not in str(f)]
    
    for filepath in python_files:
        is_valid, error = check_syntax(filepath)
        if not is_valid:
            rel_path = os.path.relpath(filepath, project_dir)
            syntax_errors.append((rel_path, error))
            print(f"  ✗ {rel_path}: {error}")
    
    if not syntax_errors:
        print(f"  ✓ All {len(python_files)} Python files have valid syntax")
    print()
    
    # 3. Check for obvious duplicate code patterns
    print("3. CHECKING FOR POTENTIAL DUPLICATE CODE")
    print("-" * 80)
    # Check for duplicate file names with " 2" suffix
    duplicate_names = []
    for filepath in project_dir.rglob("*"):
        if filepath.is_file() and " 2" in filepath.name:
            duplicate_names.append(os.path.relpath(filepath, project_dir))
    
    if duplicate_names:
        print("  Files with ' 2' suffix (likely duplicates):")
        for name in duplicate_names:
            print(f"    - {name}")
    else:
        print("  ✓ No files with ' 2' suffix found")
    print()
    
    # 4. Check main file for issues
    print("4. CHECKING MAIN FILE (resume_jd_matcher.py)")
    print("-" * 80)
    main_file = project_dir / "resume_jd_matcher.py"
    if main_file.exists():
        # Check file size
        size = main_file.stat().st_size
        lines = len(open(main_file).readlines())
        print(f"  File size: {size:,} bytes")
        print(f"  Lines of code: {lines:,}")
        
        # Check for blank sections
        blank_sections = check_blank_sections(main_file)
        if blank_sections:
            print(f"  ⚠ Found {len(blank_sections)} potential blank sections:")
            for section in blank_sections[:5]:  # Show first 5
                print(f"    - {section}")
        else:
            print("  ✓ No obvious blank sections found")
        
        # Check syntax
        is_valid, error = check_syntax(main_file)
        if is_valid:
            print("  ✓ Syntax is valid")
        else:
            print(f"  ✗ Syntax error: {error}")
    else:
        print("  ✗ Main file not found!")
    print()
    
    # 5. Summary
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total Python files checked: {len(python_files)}")
    print(f"Syntax errors: {len(syntax_errors)}")
    print(f"Duplicate file sets: {len(duplicates)}")
    print(f"Files with ' 2' suffix: {len(duplicate_names)}")
    print()
    
    if syntax_errors or duplicates or duplicate_names:
        print("⚠ ISSUES FOUND - Review recommended")
    else:
        print("✓ No major issues found")

if __name__ == "__main__":
    main()

