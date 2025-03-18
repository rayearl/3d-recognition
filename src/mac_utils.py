import numpy as np
import os
import sys
from pathlib import Path
import shutil

def convert_ply_to_numpy(ply_file_path, output_dir=None):
    """
    Convert a PLY file to a .npy file with points and colors separately
    This is useful if you have issues with PLY loading libraries
    
    Args:
        ply_file_path: Path to the PLY file
        output_dir: Directory to save the numpy files (default is same as PLY)
        
    Returns:
        Tuple of paths to the saved points and colors files
    """
    try:
        # Try to use trimesh first
        import trimesh
        mesh = trimesh.load(ply_file_path)
        
        if hasattr(mesh, 'vertices'):
            # Extract points
            points = np.array(mesh.vertices)
            
            # Extract colors
            if hasattr(mesh.visual, 'vertex_colors'):
                # Convert from RGBA uint8 to RGB float
                colors = mesh.visual.vertex_colors[:, :3].astype(np.float32) / 255.0
            else:
                colors = np.ones((points.shape[0], 3), dtype=np.float32) * 0.5
        else:
            # For point cloud files
            points = np.array(mesh.vertices)
            if hasattr(mesh, 'colors'):
                colors = np.array(mesh.colors[:, :3]).astype(np.float32) / 255.0
            else:
                colors = np.ones((points.shape[0], 3), dtype=np.float32) * 0.5
    
    except Exception as e:
        print(f"Trimesh failed: {e}")
        try:
            # Fallback to direct PLY parsing if trimesh fails
            from plyfile import PlyData
            plydata = PlyData.read(ply_file_path)
            vertex = plydata['vertex']
            
            # Extract points
            points = np.vstack([vertex['x'], vertex['y'], vertex['z']]).T
            
            # Extract colors if available
            if 'red' in vertex and 'green' in vertex and 'blue' in vertex:
                r = vertex['red']
                g = vertex['green']
                b = vertex['blue']
                # Normalize colors to [0, 1]
                if r.max() > 1.0:  # Check if colors are in [0, 255]
                    r = r / 255.0
                    g = g / 255.0
                    b = b / 255.0
                colors = np.vstack([r, g, b]).T
            else:
                colors = np.ones((points.shape[0], 3), dtype=np.float32) * 0.5
                
        except Exception as e2:
            print(f"Direct PLY parsing failed: {e2}")
            raise RuntimeError(f"Failed to load PLY file {ply_file_path}: {e}, {e2}")
    
    # Determine output paths
    if output_dir is None:
        output_dir = os.path.dirname(ply_file_path)
    
    base_name = Path(ply_file_path).stem
    points_path = os.path.join(output_dir, f"{base_name}_points.npy")
    colors_path = os.path.join(output_dir, f"{base_name}_colors.npy")
    
    # Save as numpy files
    np.save(points_path, points)
    np.save(colors_path, colors)
    
    return points_path, colors_path

def load_numpy_point_cloud(points_path, colors_path=None):
    """
    Load a point cloud from numpy files
    
    Args:
        points_path: Path to the points numpy file
        colors_path: Path to the colors numpy file (optional)
        
    Returns:
        Tuple of points and colors numpy arrays
    """
    points = np.load(points_path)
    
    if colors_path and os.path.exists(colors_path):
        colors = np.load(colors_path)
    else:
        # Generate default colors
        colors = np.ones((points.shape[0], 3), dtype=np.float32) * 0.5
    
    return points, colors

def preprocess_dataset(data_dir, output_dir=None, min_samples_per_identity=4):
    """
    Preprocess an entire dataset of PLY files and convert to numpy format
    
    Args:
        data_dir: Directory containing identity folders with PLY files
        output_dir: Directory to save the preprocessed dataset (default creates a 'preprocessed' subdirectory)
        min_samples_per_identity: Minimum number of samples required per identity
        
    Returns:
        Path to the output directory
    """
    if output_dir is None:
        output_dir = os.path.join(data_dir, "preprocessed")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Process each identity
    identities_processed = 0
    files_processed = 0
    
    for identity in os.listdir(data_dir):
        # Skip hidden files and the preprocessed directory
        if identity.startswith('.') or identity == "preprocessed":
            continue
        
        identity_dir = os.path.join(data_dir, identity)
        if not os.path.isdir(identity_dir):
            continue
        
        # Get all PLY files for this identity
        ply_files = []
        for file_name in os.listdir(identity_dir):
            if file_name.startswith('.'):
                continue
            if file_name.lower().endswith('.ply'):
                ply_files.append(os.path.join(identity_dir, file_name))
        
        # Skip if not enough samples
        if len(ply_files) < min_samples_per_identity:
            print(f"Skipping identity {identity}: only {len(ply_files)} samples (need {min_samples_per_identity})")
            continue
        
        # Create output directory for this identity
        identity_output_dir = os.path.join(output_dir, identity)
        os.makedirs(identity_output_dir, exist_ok=True)
        
        # Process each PLY file
        for ply_file in ply_files:
            try:
                print(f"Processing {ply_file}...")
                points_path, colors_path = convert_ply_to_numpy(ply_file, identity_output_dir)
                files_processed += 1
            except Exception as e:
                print(f"Error processing {ply_file}: {e}")
        
        identities_processed += 1
    
    print(f"Preprocessing complete. Processed {files_processed} files across {identities_processed} identities.")
    return output_dir

def create_data_symlinks(preprocessed_dir, output_dir, train_ratio=0.8, seed=42):
    """
    Create train/val split using symlinks to the preprocessed files
    
    Args:
        preprocessed_dir: Directory containing preprocessed numpy files
        output_dir: Directory to create train/val split directories
        train_ratio: Ratio of data for training (0.0-1.0)
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (train_dir, val_dir) paths
    """
    import random
    random.seed(seed)
    
    # Create output directories
    train_dir = os.path.join(output_dir, "train")
    val_dir = os.path.join(output_dir, "val")
    
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    
    # Process each identity
    for identity in os.listdir(preprocessed_dir):
        if identity.startswith('.'):
            continue
        
        identity_dir = os.path.join(preprocessed_dir, identity)
        if not os.path.isdir(identity_dir):
            continue
        
        # Get all point cloud files (look for *_points.npy files)
        sample_files = []
        for file_name in os.listdir(identity_dir):
            if file_name.endswith("_points.npy"):
                sample_base = file_name[:-11]  # Remove "_points.npy"
                points_file = file_name
                colors_file = f"{sample_base}_colors.npy"
                
                # Only include if both files exist
                if os.path.exists(os.path.join(identity_dir, colors_file)):
                    sample_files.append((points_file, colors_file))
        
        # Skip if no valid samples
        if not sample_files:
            continue
        
        # Create identity directories in train/val
        train_identity_dir = os.path.join(train_dir, identity)
        val_identity_dir = os.path.join(val_dir, identity)
        
        os.makedirs(train_identity_dir, exist_ok=True)
        os.makedirs(val_identity_dir, exist_ok=True)
        
        # Split files into train/val
        random.shuffle(sample_files)
        split_idx = int(len(sample_files) * train_ratio)
        
        train_samples = sample_files[:split_idx]
        val_samples = sample_files[split_idx:]
        
        # Create symlinks
        for points_file, colors_file in train_samples:
            src_points = os.path.join(identity_dir, points_file)
            dst_points = os.path.join(train_identity_dir, points_file)
            src_colors = os.path.join(identity_dir, colors_file)
            dst_colors = os.path.join(train_identity_dir, colors_file)
            
            # Use copy instead of symlink for cross-platform compatibility
            shutil.copy2(src_points, dst_points)
            shutil.copy2(src_colors, dst_colors)
        
        for points_file, colors_file in val_samples:
            src_points = os.path.join(identity_dir, points_file)
            dst_points = os.path.join(val_identity_dir, points_file)
            src_colors = os.path.join(identity_dir, colors_file)
            dst_colors = os.path.join(val_identity_dir, colors_file)
            
            # Use copy instead of symlink for cross-platform compatibility
            shutil.copy2(src_points, dst_points)
            shutil.copy2(src_colors, dst_colors)
    
    return train_dir, val_dir

# Example usage
if __name__ == "__main__":
    # Check if data directory is provided
    if len(sys.argv) < 2:
        print("Usage: python utils.py <data_directory> [output_directory]")
        sys.exit(1)
    
    data_dir = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else None
    
    # Preprocess the dataset
    preprocessed_dir = preprocess_dataset(data_dir, output_dir)
    
    # Create train/val split
    split_dir = os.path.join(preprocessed_dir, "split")
    train_dir, val_dir = create_data_symlinks(preprocessed_dir, split_dir)
    
    print(f"Preprocessing complete!")
    print(f"Preprocessed data: {preprocessed_dir}")
    print(f"Train data: {train_dir}")
    print(f"Validation data: {val_dir}")