import os
import numpy as np
import open3d as o3d
import cv2
from tqdm import tqdm
import argparse
from utils.utils_file import read_ply, crete_mesh
from utils.utils_3d import render_mesh_to_image, crop_sphere, unproject_2d_to_3d, snap_to_mesh_surface, align_3d_face
from detector.retinaface import RetinaFace

def process_face(input_ply_path, output_ply_path, face_detector, debug=False):
    try:
        vertices, triangles, colors = read_ply(input_ply_path)
        
        img_2d, depth_map, intrinsic, extrinsic = render_mesh_to_image(vertices, triangles, colors)
        
        if debug:
            os.makedirs('runs', exist_ok=True)
            cv2.imwrite('runs/img_2d_ori.png', img_2d)
        
        faces, kpss = face_detector.detect(img_2d, input_size=(640, 640))
        
        if len(faces) == 0:
            print(f"Face not detected in {input_ply_path}")
            return None
        
        if debug and len(faces) > 0:
            img_debug = img_2d.copy()
            for box, kps in zip(faces, kpss):
                x1, y1, x2, y2, _ = box.astype(int)
                cv2.rectangle(img_debug, (x1, y1), (x2, y2), (0, 255, 0), 2)
                for kp in kps:
                    cv2.circle(img_debug, tuple(kp.astype(int)), 2, (0, 0, 255), -1)
            cv2.imwrite('runs/img_2d.png', img_debug)
        
        kps = kpss[0].astype(int)
        
        keypoints_3d = unproject_2d_to_3d(kps, depth_map, intrinsic, extrinsic)
        
        mesh_original = crete_mesh(vertices, triangles)
        keypoints_3d = snap_to_mesh_surface(keypoints_3d, mesh_original, intrinsic, extrinsic)
        keypoints_3d = np.array(keypoints_3d)
        
        if debug:
            show_mesh(vertices, triangles, keypoints_3d)
        
        vertices_aligned = align_3d_face(vertices, keypoints_3d)
        
        vertices_cropped, mask = crop_sphere(vertices, keypoints_3d[2], radius=90)
        
        old_to_new = np.cumsum(mask) - 1
        
        valid_triangles = []
        for tri in triangles:
            if mask[tri[0]] and mask[tri[1]] and mask[tri[2]]:
                new_tri = [old_to_new[i] for i in tri]
                valid_triangles.append(new_tri)
        
        if colors is not None and len(colors) > 0:
            cropped_colors = [colors[i] for i, include in enumerate(mask) if include]
            
            processed_mesh = o3d.geometry.TriangleMesh()
            processed_mesh.vertices = o3d.utility.Vector3dVector(vertices_cropped)
            processed_mesh.triangles = o3d.utility.Vector3iVector(valid_triangles)
            processed_mesh.vertex_colors = o3d.utility.Vector3dVector(cropped_colors)
        else:
            processed_mesh = o3d.geometry.TriangleMesh()
            processed_mesh.vertices = o3d.utility.Vector3dVector(vertices_cropped)
            processed_mesh.triangles = o3d.utility.Vector3iVector(valid_triangles)
        
        processed_mesh.compute_vertex_normals()
        
        os.makedirs(os.path.dirname(output_ply_path), exist_ok=True)
        o3d.io.write_triangle_mesh(output_ply_path, processed_mesh)
        
        if debug:
            o3d.visualization.draw_geometries([processed_mesh])
        
        return processed_mesh
    
    except Exception as e:
        print(f"Error processing {input_ply_path}: {str(e)}")
        return None


def process_dataset(input_root, output_root, face_detector, verbose=False):
    """Process the entire dataset"""
    # Count total files
    total_files = 0
    for person_id in os.listdir(input_root):
        if person_id.startswith('.'):
            continue
        person_dir = os.path.join(input_root, person_id)
        if os.path.isdir(person_dir):
            ply_files = [f for f in os.listdir(person_dir) if f.endswith('.ply') and not f.startswith('.')]
            total_files += len(ply_files)
    
    # Process each file
    processed_count = 0
    failed_count = 0
    
    pbar = tqdm(total=total_files, desc="Processing faces")
    
    for person_id in os.listdir(input_root):
        if person_id.startswith('.'):
            continue
        person_dir = os.path.join(input_root, person_id)
        if os.path.isdir(person_dir):
            # Create output directory for this person
            output_person_dir = os.path.join(output_root, person_id)
            os.makedirs(output_person_dir, exist_ok=True)
            
            # Process each PLY file
            for ply_file in os.listdir(person_dir):
                if ply_file.startswith('.'):
                    continue
                if ply_file.endswith('.ply'):
                    input_path = os.path.join(person_dir, ply_file)
                    output_path = os.path.join(output_person_dir, ply_file)
                    
                    success = process_face(input_path, output_path, face_detector, verbose)
                    
                    if success:
                        processed_count += 1
                    else:
                        failed_count += 1
                    
                    pbar.update(1)
    
    pbar.close()
    
    print(f"Dataset processing complete:")
    print(f"  - Total files: {total_files}")
    print(f"  - Successfully processed: {processed_count}")
    print(f"  - Failed: {failed_count}")

def main():
    parser = argparse.ArgumentParser(description="Align and crop 3D face dataset")
    parser.add_argument("--input", type=str, required=True, help="Input dataset root directory")
    parser.add_argument("--output", type=str, required=True, help="Output directory for processed dataset")
    parser.add_argument("--detector", type=str, default="weights/det_500m.onnx", help="Path to RetinaFace weights")
    parser.add_argument("--verbose", action="store_true", help="Print detailed progress")
    args = parser.parse_args()
    
    # Initialize face detector
    face_detector = RetinaFace(args.detector)
    face_detector.det_thresh = 0.3
    
    # Process the entire dataset
    process_dataset(args.input, args.output, face_detector, args.verbose)

if __name__ == "__main__":
    main()
    