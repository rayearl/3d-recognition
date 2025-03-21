import numpy as np
import open3d as o3d
import cv2
import os
import copy
from utils.utils_file import read_ply, show_mesh, crete_mesh
from utils.utils_3d import render_mesh_to_image, crop_sphere, unproject_2d_to_3d, snap_to_mesh_surface, align_3d_face, filter_points_by_z_distance
from detector.retinaface import RetinaFace
from scipy.spatial import Delaunay
from utils.face2d import calc_error
from utils.face3d import *
from mpl_toolkits.mplot3d import Axes3D
import onnxruntime as ort



def process_face(input_ply_path, output_ply_path, face_detector, debug=False):
    vertices, triangles, colors = read_ply(input_ply_path)

    img_2d, depth_map, intrinsic, extrinsic = render_mesh_to_image(vertices, triangles, None, img_size=(2048, 2048))
    
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
    
    vertices_cropped, mask = crop_sphere(vertices, keypoints_3d[2], radius=90)
    if len(vertices_cropped) < 0:
        return None

    center, axes, extents = compute_oriented_bounding_box(vertices_cropped)

    distance_nose_to_center = calculate_distance(keypoints_3d[2], center)
    rotate_angle = calculate_obb_angles(axes)
    print(rotate_angle)
    if rotate_angle > 30:
        return None

    vertices_aligned, keypoints_3d_aligned = transform_to_new_coordinate(
        vertices_cropped, keypoints_3d, center, axes
    )
    print(len(vertices_aligned))
    vertices_aligned, keypoints_3d_aligned = rotate_to_eyes_up(
        vertices_aligned, keypoints_3d_aligned
    )
    
    if debug:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(vertices_aligned)
        pcd.paint_uniform_color([0.8, 0.8, 0.8])  #
        o3d.visualization.draw_geometries([pcd])
        visualize_obb_with_center_and_axes(vertices_cropped, center, axes, extents)

    old_to_new = np.cumsum(mask) - 1
    
    valid_triangles = []
    for tri in triangles:
        if mask[tri[0]] and mask[tri[1]] and mask[tri[2]]:
            new_tri = [old_to_new[i] for i in tri]
            valid_triangles.append(new_tri)
    
    if colors is not None and len(colors) > 0:
        cropped_colors = [colors[i] for i, include in enumerate(mask) if include]
        
        processed_mesh = o3d.geometry.TriangleMesh()
        processed_mesh.vertices = o3d.utility.Vector3dVector(vertices_aligned)
        processed_mesh.triangles = o3d.utility.Vector3iVector(valid_triangles)
        processed_mesh.vertex_colors = o3d.utility.Vector3dVector(cropped_colors)
    else:
        processed_mesh = o3d.geometry.TriangleMesh()
        processed_mesh.vertices = o3d.utility.Vector3dVector(vertices_aligned)
        processed_mesh.triangles = o3d.utility.Vector3iVector(valid_triangles)
    
    processed_mesh.compute_vertex_normals()
    if debug:
        o3d.visualization.draw_geometries([processed_mesh], window_name="Aligned Face")
    
    print(output_ply_path)
    os.makedirs(os.path.dirname(output_ply_path), exist_ok=True)
    o3d.io.write_triangle_mesh(output_ply_path, processed_mesh)
    
    return processed_mesh
    

def main():
    face_detector = RetinaFace("weights/det_500m.onnx")
    face_detector.det_thresh = 0.2
    
    root = r'D:\Projects\01.appliedrecog\3D-Face-PLY-61-90\WRL61-90\061'
    out = r'D:\Projects\01.appliedrecog\3D-Face-PLY-61-90-Aligned\061'

    for i in os.listdir(root):
        if not i.endswith(".ply"):
            continue

        input_ply = fr"{root}/{i}"

        output_ply = fr"{out}/{i}"

        print(input_ply)
        try:
            processed_mesh = process_face(input_ply, output_ply, face_detector, debug=True)
            
            if processed_mesh is not None:
                print("Face processed successfully!")
            else:
                print("Face processing failed!")
        except:
            pass
        break
if __name__ == "__main__":
    main()
