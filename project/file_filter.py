"""
File Filter Module
This module accepts the base directory path and filters files based on their extensions.
The filtered files' paths are saved to a specified output file.
"""

import os
import json
EXTENSION_PARSERS = {
    ".py": "tree-sitter-python",
    ".js": "tree-sitter-javascript",
    ".java": "tree-sitter-java",
    ".cpp": "tree-sitter-cpp",
    ".c": "tree-sitter-c",
}

def get_file_extension_mapping():
    """
    Returns a mapping of programming languages to their file extensions.
    """
    return {
        'python': ['.py'],
        'c': ['.c', '.h'],
        'javascript': ['.js', '.jsx'],
        'java': ['.java']
    }

EXCLUDED_DIRS = {
    '__pycache__', 'tree_sitter_libs', '.git', 'node_modules', 'venv', 'env',
    'build', 'dist', '.idea', '.vscode', '__test__', 'tests'
}
EXCLUDED_EXTENSIONS = {'.pyc', '.pyo', '.lock'}
MAX_FILE_SIZE_MB = 5  

def should_exclude_dir(dir_path, ignored_items=None):
    """Checks if a directory should be excluded."""
    if ignored_items is None:
        ignored_items = set()
    dir_name = os.path.basename(dir_path)
    return any(excluded in dir_path.split(os.sep) for excluded in EXCLUDED_DIRS) or dir_name in ignored_items

def should_exclude_file(file_name, file_ext, file_size, ignored_items=None):
    """Checks if a file should be excluded."""
    if ignored_items is None:
        ignored_items = set()
    return (
        file_ext.lower() in EXCLUDED_EXTENSIONS or
        file_name.startswith('.') or
        'test' in file_name.lower() or
        file_size > MAX_FILE_SIZE_MB * 1024 * 1024 or
        file_name in ignored_items
    )


def filter_files_by_extension(base_dir, output_file=None, ignored_items=None):
    """
    Filters relevant code files from a directory and associates them with tree-sitter parsers.
    
    Args:
        base_dir (str): Base directory to scan
        output_file (str, optional): Path to output JSON file
        ignored_items (list, optional): List of file or directory names to ignore
    """
    print(f"Scanning: {base_dir}...")
    
    if ignored_items is None:
        ignored_items = set()
    else:
        ignored_items = set(ignored_items)
        
    extension_mapping = get_file_extension_mapping()
    filtered_files = {lang: {"parser": EXTENSION_PARSERS.get(extensions[0], "Unknown"), "files": []} for lang, extensions in extension_mapping.items()}
    excluded_dirs_count, excluded_files_count = 0, 0

    for root, dirs, files in os.walk(base_dir):
        if should_exclude_dir(root, ignored_items):
            excluded_dirs_count += 1
            continue

        for file in files:
            file_path = os.path.join(root, file)
            file_ext = os.path.splitext(file)[1]
            file_size = os.path.getsize(file_path)

            if should_exclude_file(file, file_ext, file_size, ignored_items):
                excluded_files_count += 1
                continue

            for lang, extensions in extension_mapping.items():
                if file_ext in extensions:
                    filtered_files[lang]["files"].append(file_path)
                    break

    total_files = sum(len(data["files"]) for data in filtered_files.values())
    print(f"Found {total_files} files (Skipped {excluded_dirs_count} directories, {excluded_files_count} files).")

    for lang, data in filtered_files.items():
        print(f"  - {lang} ({data['parser']}): {len(data['files'])} files")

    if output_file:
        with open(output_file, 'w') as f:
            json.dump(filtered_files, f, indent=2)
        print(f"Saved to: {output_file}")

    return filtered_files

if __name__ == "__main__":
    try:
        from . import BASE_DIR
    except ImportError:
        import sys
        sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        from modules.code_extraction import BASE_DIR

    filter_files_by_extension(BASE_DIR, "code_files.json")
