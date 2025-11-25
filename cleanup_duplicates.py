#!/usr/bin/env python3
"""
Cleanup script to identify and optionally remove duplicate files.

This script helps clean up the project by identifying duplicate files.
"""

import os
from pathlib import Path

def find_duplicates():
    """Find duplicate files in the project."""
    project_dir = Path(__file__).parent
    
    # Known duplicates
    duplicates = [
        "resume_jd_matcher 2.py",
        "RESUME_MATCHER_README 2.md",
        "resume_matcher_requirements 2.txt"
    ]
    
    print("🔍 Checking for duplicate files...\n")
    
    found_duplicates = []
    for dup_file in duplicates:
        file_path = project_dir / dup_file
        if file_path.exists():
            found_duplicates.append(file_path)
            print(f"⚠️  Found duplicate: {file_path.name}")
            print(f"   Size: {file_path.stat().st_size:,} bytes")
            print(f"   Path: {file_path}\n")
        else:
            print(f"✅ No duplicate: {dup_file}\n")
    
    if found_duplicates:
        print(f"\n📊 Summary: Found {len(found_duplicates)} duplicate file(s)")
        print("\n💡 Recommendation:")
        print("   These are likely backup/alternative versions.")
        print("   You can safely delete them if you're using the main versions.")
        print("\n   To delete them, run:")
        print("   python cleanup_duplicates.py --delete")
    else:
        print("✅ No duplicate files found!")
    
    return found_duplicates

def delete_duplicates():
    """Delete duplicate files (with confirmation)."""
    duplicates = find_duplicates()
    
    if not duplicates:
        print("No duplicates to delete.")
        return
    
    print(f"\n⚠️  WARNING: This will delete {len(duplicates)} file(s)")
    response = input("Are you sure? (yes/no): ")
    
    if response.lower() == 'yes':
        for dup_file in duplicates:
            try:
                dup_file.unlink()
                print(f"✅ Deleted: {dup_file.name}")
            except Exception as e:
                print(f"❌ Error deleting {dup_file.name}: {e}")
    else:
        print("Cancelled.")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--delete":
        delete_duplicates()
    else:
        find_duplicates()
        print("\n💡 Tip: Run with --delete flag to remove duplicates")

