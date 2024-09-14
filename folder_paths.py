# folder_paths.py
# Utility functions to test the output of the save_image_node.py

import os

def get_output_directory():
    return os.path.abspath("output")

def get_save_image_path(filename_prefix, output_dir, width, height):
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Define filename pattern
    filename = f"{filename_prefix}_%batch_num%_image"

    # Initialize counter
    counter = 0

    # Define subfolder (optional)
    subfolder = ""

    return output_dir, filename, counter, subfolder, filename_prefix

def get_input_directory():
    return os.path.abspath("input")
