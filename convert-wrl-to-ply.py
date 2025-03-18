#!/usr/bin/env python
"""
WRL to PLY Converter Script using PyMeshLab

Converts all WRL files in the input directory to PLY format in the output directory,
preserving color information by using the same engine as MeshLab.
"""

import os
import sys
import argparse
from tqdm import tqdm
import pymeshlab

def convert_wrl_to_ply(input_path, output_path, verbose=False):
    """
    Convert a WRL file to PLY format using PyMeshLab
    
    Parameters:
    input_path (str): Path to input WRL file
    output_path (str): Path to output PLY file
    verbose (bool): Whether to print detailed debug info
    
    Returns:
    bool: True if successful, False otherwise
    """
    try:
        # Create a new MeshSet
        ms = pymeshlab.MeshSet()
        
        # Load the mesh
        ms.load_new_mesh(input_path)
        
        if verbose:
            # Get the current mesh
            mesh = ms.current_mesh()
            print(f"File: {input_path}")
            print(f"Vertices: {mesh.vertex_number()}")
            print(f"Faces: {mesh.face_number()}")
            print(f"Has vertex colors: {mesh.has_vertex_color()}")
            print(f"Has face colors: {mesh.has_face_color()}")
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save as PLY with colors - IMPORTANT: use binary=False to create ASCII PLY files
        ms.save_current_mesh(output_path, save_vertex_color=True, save_face_color=True, binary=False)
        
        if verbose:
            print(f"  Saved mesh to {output_path}")
        
        return True
    
    except Exception as e:
        print(f"Error converting {input_path}: {str(e)}")
        return False

def process_directory(input_dir, output_dir, recursive=True, verbose=False):
    """
    Process all WRL files in a directory
    
    Parameters:
    input_dir (str): Input directory path
    output_dir (str): Output directory path
    recursive (bool): Whether to process subdirectories recursively
    verbose (bool): Whether to print detailed debug info
    
    Returns:
    tuple: (total_files, success_count, failed_files)
    """
    total_files = 0
    success_count = 0
    failed_files = []
    
    # Count total files first for progress bar
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith('.wrl') and not file.startswith('.'):
                total_files += 1
        
        if not recursive:
            break
    
    # Process files with progress bar
    with tqdm(total=total_files, desc="Converting WRL to PLY") as pbar:
        for root, dirs, files in os.walk(input_dir):
            # Create relative path for output
            rel_path = os.path.relpath(root, input_dir)
            if rel_path == '.':
                rel_path = ''
            
            for file in files:
                if file.lower().endswith('.wrl') and not file.startswith('.'):
                    # Construct input and output paths
                    input_path = os.path.join(root, file)
                    output_file = os.path.splitext(file)[0] + '.ply'
                    output_path = os.path.join(output_dir, rel_path, output_file)
                    
                    # Convert file
                    success = convert_wrl_to_ply(input_path, output_path, verbose)
                    
                    if success:
                        success_count += 1
                    else:
                        failed_files.append(input_path)
                    
                    pbar.update(1)
            
            if not recursive:
                break
    
    return total_files, success_count, failed_files

def main():
    parser = argparse.ArgumentParser(description="Convert WRL files to PLY format with color preservation using PyMeshLab")
    parser.add_argument("--input", type=str, required=True, help="Input directory containing WRL files")
    parser.add_argument("--output", type=str, required=True, help="Output directory for PLY files")
    parser.add_argument("--no-recursive", action="store_true", help="Do not process subdirectories recursively")
    parser.add_argument("--verbose", action="store_true", help="Print detailed debug information")
    args = parser.parse_args()
    
    print(f"Converting WRL files from {args.input} to PLY files in {args.output}")
    print("Using PyMeshLab to preserve color information")
    
    # Process files
    total, success, failed = process_directory(
        args.input, 
        args.output, 
        recursive=not args.no_recursive,
        verbose=args.verbose
    )
    
    # Print summary
    print(f"\nConversion complete:")
    print(f"  - Total WRL files: {total}")
    print(f"  - Successfully converted: {success}")
    print(f"  - Failed: {len(failed)}")
    
    if len(failed) > 0:
        print("\nFailed files:")
        for file in failed[:10]:  # Show first 10 failed files
            print(f"  - {file}")
        
        if len(failed) > 10:
            print(f"  - (and {len(failed) - 10} more)")

if __name__ == "__main__":
    main()