#!/usr/bin/env python
"""
Standalone script to render PLY files to PNG images for inspection.
This helps to check why face detection might be failing.
"""

import os
import sys
import argparse
import numpy as np
import cv2
from tqdm import tqdm
import open3d as o3d

def read_ply(file_path):
    """Read a PLY file using Open3D (handles both ASCII and binary)"""
    try:
        mesh = o3d.io.read_triangle_mesh(file_path)
        vertices = np.asarray(mesh.vertices)
        triangles = np.asarray(mesh.triangles)
        
        if mesh.has_vertex_colors():
            colors = np.asarray(mesh.vertex_colors)
        else:
            colors = None
            
        return vertices, triangles, colors
    except Exception as e:
        raise RuntimeError(f"Error reading PLY file {file_path}: {str(e)}")

def render_mesh_to_image(vertices, triangles, colors=None, img_size=(1024, 1024)):
    """Render a 3D mesh to a 2D image (simplified version)"""
    # Create mesh
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(triangles)
    
    if colors is not None:
        mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
    else:
        # Use default color if no colors provided
        mesh.paint_uniform_color([0.7, 0.7, 0.7])
    
    # Compute normals for better visualization
    mesh.compute_vertex_normals()
    
    # Set up a simple renderer
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False, width=img_size[0], height=img_size[1])
    vis.add_geometry(mesh)
    
    # Center and auto-scale the view
    vis.get_view_control().rotate(0.0, 180.0)  # Rotate to front view
    vis.get_view_control().set_zoom(1.0)
    
    # Optional: add light to improve face visibility
    opt = vis.get_render_option()
    opt.background_color = np.array([0, 0, 0])  # Black background
    opt.point_size = 1.0
    
    # Render
    vis.poll_events()
    vis.update_renderer()
    img = vis.capture_screen_float_buffer(do_render=True)
    
    # Convert to numpy array and back to regular image format
    img_np = np.asarray(img) * 255
    img_np = img_np.astype(np.uint8)
    
    # Convert RGB to BGR (for OpenCV)
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    
    # Close visualizer
    vis.destroy_window()
    
    return img_bgr

def process_ply_file(input_path, output_dir):
    """Process a single PLY file and save rendered PNGs"""
    try:
        # Extract file information for output path
        filename = os.path.basename(input_path)
        filebase = os.path.splitext(filename)[0]
        
        # Create subdirectory in output folder if needed
        rel_dir = os.path.dirname(os.path.relpath(input_path, args.input))
        out_dir = os.path.join(output_dir, rel_dir)
        os.makedirs(out_dir, exist_ok=True)
        
        # Load PLY file
        vertices, triangles, colors = read_ply(input_path)
        
        # Render the mesh to image
        img = render_mesh_to_image(vertices, triangles, colors)
        
        # Also create a normalized version for better visibility
        img_norm = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
        
        # Save images
        cv2.imwrite(os.path.join(out_dir, f"{filebase}_render.png"), img)
        cv2.imwrite(os.path.join(out_dir, f"{filebase}_normalized.png"), img_norm)
        
        return True
    except Exception as e:
        print(f"Error processing {input_path}: {str(e)}")
        return False

def process_directory(input_dir, output_dir):
    """Process all PLY files in a directory recursively"""
    # Count files for progress bar
    total_files = 0
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.endswith('.ply') and not file.startswith('.'):
                total_files += 1
    
    # Process files
    success_count = 0
    failed_count = 0
    
    with tqdm(total=total_files, desc="Rendering PLY files") as pbar:
        for root, dirs, files in os.walk(input_dir):
            for file in files:
                if file.endswith('.ply') and not file.startswith('.'):
                    input_path = os.path.join(root, file)
                    
                    # Process file
                    success = process_ply_file(input_path, output_dir)
                    
                    if success:
                        success_count += 1
                    else:
                        failed_count += 1
                    
                    pbar.update(1)
    
    # Print summary
    print(f"\nRendering complete:")
    print(f"  - Total PLY files: {total_files}")
    print(f"  - Successfully rendered: {success_count}")
    print(f"  - Failed: {failed_count}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Render PLY files to PNG images for inspection")
    parser.add_argument("--input", type=str, required=True, help="Input directory containing PLY files")
    parser.add_argument("--output", type=str, required=True, help="Output directory for rendered images")
    args = parser.parse_args()
    
    print(f"Rendering PLY files from {args.input} to PNG images in {args.output}")
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output, exist_ok=True)
    
    # Process the directory
    process_directory(args.input, args.output)