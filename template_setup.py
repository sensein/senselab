import argparse
import os
import sys
from typing import Dict


def replace_in_file(file_path: str, replacements: Dict[str, str]) -> None:
    """Replace placeholders in the content of a file with given values, skipping binary files.

    Args:
        file_path: The path to the file where replacements need to be made.
        replacements: A dictionary mapping placeholder strings to their replacement values.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()

        original_content = content
        for old, new in replacements.items():
            content = content.replace(old, new)

        if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as file:
                file.write(content)

    except UnicodeDecodeError:
        print(f"Skipping binary file: {file_path}")


def replace_in_filename(file_path: str, replacements: Dict[str, str]) -> str:
    """Replace placeholders in the name of a file and return the new file path.

    Args:
        file_path: The path to the file whose name needs to be modified.
        replacements: A dictionary mapping placeholder strings to their values.

    Returns:
        The new file path after renaming.
    """
    directory, filename = os.path.split(file_path)
    new_filename = filename
    for old, new in replacements.items():
        new_filename = new_filename.replace(old, new)

    new_file_path = os.path.join(directory, new_filename)
    if new_file_path != file_path:
        os.rename(file_path, new_file_path)

    return new_file_path

def replace_in_folder_name(dir_path: str, replacements: Dict[str, str]) -> str:
    """Replace placeholders in the name of a folder and return the new folder path.

    Args:
        dir_path: The path to the directory whose name needs to be modified.
        replacements: A dictionary mapping placeholder strings to their replacement values.

    Returns:
        The new directory path after renaming.
    """
    parent_dir, dir_name = os.path.split(dir_path)
    new_dir_name = dir_name
    for old, new in replacements.items():
        new_dir_name = new_dir_name.replace(old, new)

    new_dir_path = os.path.join(parent_dir, new_dir_name)
    if new_dir_path != dir_path:
        os.rename(dir_path, new_dir_path)
        return new_dir_path

    return dir_path

def process_directory(replacements: Dict[str, str]) -> None:
    """Recursively process a directory, replacing placeholders.

    Args:
        replacements: A dictionary mapping placeholder strings to their values.
    """
    for dirpath, dirnames, filenames in os.walk(".", topdown=False):
        # Replace placeholders in the files' content and names
        for filename in filenames:
            file_path = os.path.join(dirpath, filename)
            file_path = replace_in_filename(file_path, replacements)
            replace_in_file(file_path, replacements)
        
        # Replace placeholders in the folder names
        for dirname in dirnames:
            dir_path = os.path.join(dirpath, dirname)
            new_dir_path = replace_in_folder_name(dir_path, replacements)
            # Update dirnames list to reflect any changes
            index = dirnames.index(dirname)
            dirnames[index] = os.path.basename(new_dir_path)

def delete_current_script():
    """Deletes the currently executing script file."""
    script_file = sys.argv[0]  # Get the name of the current file
    try:
        os.remove(script_file)  # Attempt to delete the file
    except Exception as e:
        print(f"Error occurred while trying to delete the script: {e}")

def rename_github_directory():
    """Renames a directory from 'github' to '.github'."""
    current_name, new_name = "github", ".github"
    try:
        # Check if the current directory exists
        if os.path.exists(current_name):
            # Rename the directory
            os.rename(current_name, new_name)
            print(f"Directory renamed from '{current_name}' to '{new_name}'")
        else:
            print(f"The directory '{current_name}' does not exist.")
    except Exception as e:
        print(f"Error renaming directory: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Replace placeholders.")
    parser.add_argument("--package-name", required=True, help="Package name.")
    parser.add_argument("--package-repo-without-git-extension", 
                        required=True, 
                        help="Package repository URL, without .git extension.")
    parser.add_argument("--github-nickname", required=True, help="GitHub nickname.")
    parser.add_argument("--codecov-token", required=True, help="Codecov graphics token.")
    parser.add_argument("--email", required=True, help="Email address.")

    args = parser.parse_args()

    replacements = {
        "YOUR_PACKAGE_NAME": args.package_name,
        "YOUR_PACKAGE_REPO": args.package_repo_without_git_extension,
        "YOUR_GITHUB_NICKNAME": args.github_nickname,
        "CODECOV_GRAPHICS_TOKEN": args.codecov_token,
        "YOUR_EMAIL": args.email,
    }

    process_directory(replacements)
    rename_github_directory()
    delete_current_script()

