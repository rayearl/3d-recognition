import numpy as np
import open3d as o3d
import cv2
import os
from utils.utils_file import read_ply, show_mesh, crete_mesh
from utils.utils_3d import render_mesh_to_image, crop_sphere, unproject_2d_to_3d, snap_to_mesh_surface, align_3d_face, filter_points_by_z_distance
from detector.retinaface import RetinaFace
from scipy.spatial import Delaunay

def show_face(ptc):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(ptc)
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(pcd)
    view_control = vis.get_view_control()
    view_control.set_front([0, 0, -1])
    view_control.set_up([0, -1, 0])
    vis.run()
    vis.destroy_window()

def process_face(input_ply_path, output_ply_path, face_detector, debug=False):
    try:
        vertices, triangles, colors = read_ply(input_ply_path)

        tri = Delaunay(vertices[:, :2])
        triangles = tri.simplices

        img_2d, depth_map, intrinsic, extrinsic = render_mesh_to_image(vertices, triangles, colors, img_size=(1024, 1024))
        
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
        
        vertices_cropped, mask = crop_sphere(vertices, keypoints_3d[2], radius=0.9)
        
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

def main():
    face_detector = RetinaFace("weights/det_500m.onnx")
    face_detector.det_thresh = 0.3
    
    input_ply = r"sample_output_aligned/102/instance_00.ply"
    output_ply = os.path.join("runs", 'processed_face.ply')
    
    processed_mesh = process_face(input_ply, output_ply, face_detector, debug=True)
    
    if processed_mesh is not None:
        print("Face processed successfully!")
    else:
        print("Face processing failed!")

if __name__ == "__main__":
    main()
