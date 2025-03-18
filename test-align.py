#!/usr/bin/env python
"""
Simplified version of the align-training-dataset.py script
to identify issues with file creation and face detection.
"""

import os
import sys
import cv2
import numpy as np
import argparse
from utils.utils_file import read_mesh
from detector.retinaface import RetinaFace

def simple_process_face(input_path, output_path, debug=True):
    """A simplified version that just tries to create files"""
    print(f"Processing: {input_path} -> {output_path}")
    
    # Create output directory
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        print(f"Created output directory: {os.path.dirname(output_path)}")
    except Exception as e:
        print(f"Failed to create output directory: {str(e)}")
        return False
    
    # Create a placeholder file regardless
    try:
        with open(output_path, 'w') as f:
            f.write(f"# Placeholder for {input_path}\n")
        print(f"Successfully created file: {output_path}")
        return True
    except Exception as e:
        print(f"Failed to create file: {str(e)}")
        return False

def process_sample(input_root, output_root, limit=5):
    """Process just a few files as a test"""
    count = 0
    success_count = 0
    
    print(f"Processing up to {limit} files from {input_root} to {output_root}")
    
    for person_id in os.listdir(input_root):
        if person_id.startswith('.'):
            continue
        
        person_dir = os.path.join(input_root, person_id)
        if os.path.isdir(person_dir):
            output_person_dir = os.path.join(output_root, person_id)
            
            for file in os.listdir(person_dir):
                if file.endswith('.wrl') and not file.startswith('.'):
                    input_path = os.path.join(person_dir, file)
                    output_path = os.path.join(output_person_dir, file)
                    
                    success = simple_process_face(input_path, output_path)
                    count += 1
                    if success:
                        success_count += 1
                    
                    if count >= limit:
                        break
        
        if count >= limit:
            break
    
    print(f"Processed {count} files, {success_count} succeeded")
    return success_count

def main():
    parser = argparse.ArgumentParser(description="Simplified testing script")
    parser.add_argument("--input", type=str, required=True, help="Input dataset root directory")
    parser.add_argument("--output", type=str, required=True, help="Output directory for processed dataset")
    parser.add_argument("--limit", type=int, default=5, help="Maximum number of files to process")
    args = parser.parse_args()
    
    print("=== Starting simplified processing ===")
    process_sample(args.input, args.output, args.limit)
    print("=== Completed simplified processing ===")

if __name__ == "__main__":
    main()