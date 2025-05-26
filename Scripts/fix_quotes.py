#!/usr/bin/env python3
"""
Script to fix escaped quotes in enhanced_api.py
"""

def fix_escaped_quotes():
    file_path = 'Scripts/enhanced_api.py'
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Replace escaped quotes with regular quotes
        content = content.replace('\\"', '"')
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print(f"Successfully fixed escaped quotes in {file_path}")
        
    except Exception as e:
        print(f"Error fixing quotes: {e}")

if __name__ == "__main__":
    fix_escaped_quotes()
