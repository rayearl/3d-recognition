import numpy as np
import open3d as o3d
import cv2
import os
import copy
from utils.utils_file import read_ply, show_mesh, crete_mesh
from utils.utils_3d import render_mesh_to_image, crop_sphere, unproject_2d_to_3d, snap_to_mesh_surface
from detector.retinaface import RetinaFace
from scipy.spatial import Delaunay
from utils.face2d import calc_error
from utils.face3d import *
from mpl_toolkits.mplot3d import Axes3D
import onnxruntime as ort
import numpy as np
from scipy.spatial.transform import Rotation


def rotation_matrix_from_axis_angle(axis, angle):
    K = np.array([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0]
    ])
    
    R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
    
    return R

def compute_rotation_matrix_to_align_vectors(source, target):
    source = source / np.linalg.norm(source)
    target = target / np.linalg.norm(target)
    
    dot_product = np.dot(source, target)
    
    if np.abs(dot_product - 1.0) < 1e-10:
        return np.eye(3)
    
    if np.abs(dot_product + 1.0) < 1e-10:
        perpendicular = np.array([1.0, 0.0, 0.0])
        if np.abs(np.dot(perpendicular, source)) > 0.9:
            perpendicular = np.array([0.0, 1.0, 0.0])
        
        perpendicular = perpendicular - np.dot(perpendicular, source) * source
        perpendicular = perpendicular / np.linalg.norm(perpendicular)
        
        R = rotation_matrix_from_axis_angle(perpendicular, np.pi)
        return R
    
    cross_product = np.cross(source, target)
    cross_product_norm = np.linalg.norm(cross_product)
    
    axis = cross_product / cross_product_norm
    
    angle = np.arctan2(cross_product_norm, dot_product)
    
    R = rotation_matrix_from_axis_angle(axis, angle)
    
    return R


def rotate_face_to_frontal(vertices, landmarks):
    left_eye = landmarks[0]
    right_eye = landmarks[1]
    nose = landmarks[2]
    
    eye_line = right_eye - left_eye
    eye_center = (left_eye + right_eye) / 2
    nose_line = nose - eye_center
    
    has_mouth = len(landmarks) >= 5
    if has_mouth:
        left_mouth = landmarks[3]
        right_mouth = landmarks[4]
        mouth_center = (left_mouth + right_mouth) / 2
    
    face_normal = np.cross(eye_line, nose_line)
    face_normal = face_normal / np.linalg.norm(face_normal)
    
    if face_normal[2] < 0:
        face_normal = -face_normal
    
    target_normal = np.array([0, 0, 1])
    
    center = nose.copy()
    
    R1 = compute_rotation_matrix_to_align_vectors(face_normal, target_normal)
    
    vertices_step1 = np.zeros_like(vertices)
    for i in range(len(vertices)):
        vertices_step1[i] = R1 @ (vertices[i] - center)
    
    landmarks_step1 = np.zeros_like(landmarks)
    for i in range(len(landmarks)):
        landmarks_step1[i] = R1 @ (landmarks[i] - center)

    eye_line_step1 = landmarks_step1[1] - landmarks_step1[0]
    
    eye_line_xy = np.array([eye_line_step1[0], eye_line_step1[1], 0])
    
    if np.linalg.norm(eye_line_xy) < 1e-6:
        R2 = np.eye(3)
    else:
        eye_line_xy = eye_line_xy / np.linalg.norm(eye_line_xy)
        
        target_eye_line = np.array([1, 0, 0])
        R2 = compute_rotation_matrix_to_align_vectors(eye_line_xy, target_eye_line)
    
    vertices_step2 = np.zeros_like(vertices_step1)
    for i in range(len(vertices_step1)):
        vertices_step2[i] = R2 @ vertices_step1[i]
    
    landmarks_step2 = np.zeros_like(landmarks_step1)
    for i in range(len(landmarks_step1)):
        landmarks_step2[i] = R2 @ landmarks_step1[i]
    
    if has_mouth:
        eye_center_step2 = (landmarks_step2[0] + landmarks_step2[1]) / 2
        eye_to_mouth = landmarks_step2[4] - eye_center_step2
        
        eye_to_mouth_yz = np.array([0, eye_to_mouth[1], eye_to_mouth[2]])
        
        if np.linalg.norm(eye_to_mouth_yz) > 1e-6:
            eye_to_mouth_yz = eye_to_mouth_yz / np.linalg.norm(eye_to_mouth_yz)
            
            target_vector = np.array([0, -1, 0])
            
            R3 = compute_rotation_matrix_to_align_vectors(eye_to_mouth_yz, target_vector)
        else:
            R3 = np.eye(3)
    else:
        eye_center_step2 = (landmarks_step2[0] + landmarks_step2[1]) / 2
        nose_pos = landmarks_step2[2]
        
        if nose_pos[1] > eye_center_step2[1]:
            R3 = np.array([
                [1, 0, 0],
                [0, -1, 0],
                [0, 0, -1]
            ])
        else:
            nose_yz = np.array([0, nose_pos[1] - eye_center_step2[1], nose_pos[2]])
            if np.linalg.norm(nose_yz) > 1e-6:
                nose_yz = nose_yz / np.linalg.norm(nose_yz)
                target_vector = np.array([0, -1, 0])
                R3 = compute_rotation_matrix_to_align_vectors(nose_yz, target_vector)
            else:
                R3 = np.eye(3)
    
    vertices_frontal = np.zeros_like(vertices_step2)
    for i in range(len(vertices_step2)):
        vertices_frontal[i] = R3 @ vertices_step2[i]
    
    landmarks_frontal = np.zeros_like(landmarks_step2)
    for i in range(len(landmarks_step2)):
        landmarks_frontal[i] = R3 @ landmarks_step2[i]
    
    R_combined = R3 @ R2 @ R1
    
    try:
        r = Rotation.from_matrix(R_combined)
        angles_rad = r.as_euler('xyz', degrees=False)
        
        angles_deg = np.degrees(angles_rad)
        rotation_angles = {
            "roll": float(angles_deg[0]),
            "pitch": float(angles_deg[1]),
            "yaw": float(angles_deg[2])
        }
    except Exception as e:
        print(f"Error calculating Euler angles: {str(e)}")
        rotation_angles = {"roll": 0.0, "pitch": 0.0, "yaw": 0.0}
    
    return vertices_frontal, landmarks_frontal, rotation_angles


def process_face(input_ply_path, output_ply_path, face_detector, debug=False):
    vertices, triangles, colors = read_ply(input_ply_path)

    img_2d, depth_map, intrinsic, extrinsic = render_mesh_to_image(vertices, triangles, None, img_size=(2048, 2048))
    
    if debug:
        os.makedirs('runs', exist_ok=True)
        cv2.imwrite('runs/img_2d_ori.png', img_2d)
    
    # Phát hiện khuôn mặt và các điểm mốc
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
    
    # Lấy điểm mốc của khuôn mặt đầu tiên được phát hiện
    kps = kpss[0].astype(int)
    
    # Chuyển đổi điểm mốc 2D thành 3D
    keypoints_3d = unproject_2d_to_3d(kps, depth_map, intrinsic, extrinsic)
    
    # Snap các điểm mốc vào bề mặt mesh
    mesh_original = crete_mesh(vertices, triangles)
    keypoints_3d = snap_to_mesh_surface(keypoints_3d, mesh_original, intrinsic, extrinsic)
    keypoints_3d = np.array(keypoints_3d)

    if debug:
        show_mesh(vertices, triangles, keypoints_3d)
    
    # Cắt khuôn mặt từ mesh (lấy phần hình cầu xung quanh mũi)
    vertices_cropped, mask = crop_sphere(vertices, keypoints_3d[2], radius=90)
    if len(vertices_cropped) < 100:
        print(f"Not enough points after cropping in {input_ply_path}")
        return None

    # Xoay khuôn mặt về chính diện
    vertices_frontal, landmarks_frontal, rotation_angles = rotate_face_to_frontal(vertices_cropped, keypoints_3d)
    
    if debug:
        # Hiển thị kết quả trước và sau khi xoay
        print(rotation_angles)
        pcd_before = o3d.geometry.PointCloud()
        pcd_before.points = o3d.utility.Vector3dVector(vertices_cropped)
        pcd_before.paint_uniform_color([0.8, 0.2, 0.2])  # Đỏ
        
        pcd_after = o3d.geometry.PointCloud()
        pcd_after.points = o3d.utility.Vector3dVector(vertices_frontal)
        pcd_after.paint_uniform_color([0.2, 0.8, 0.2])
        
        landmarks_before_pcd = o3d.geometry.PointCloud()
        landmarks_before_pcd.points = o3d.utility.Vector3dVector(keypoints_3d)
        landmarks_before_pcd.paint_uniform_color([1.0, 0.0, 0.0])
        
        landmarks_after_pcd = o3d.geometry.PointCloud()
        landmarks_after_pcd.points = o3d.utility.Vector3dVector(landmarks_frontal)
        landmarks_after_pcd.paint_uniform_color([0.0, 1.0, 0.0]) 
        
        o3d.visualization.draw_geometries([pcd_before, landmarks_before_pcd], 
                                          window_name="Before Rotation")
        
        o3d.visualization.draw_geometries([pcd_after, landmarks_after_pcd],
                                          window_name="After Rotation")

    old_to_new = np.cumsum(mask) - 1
    
    valid_triangles = []
    for tri in triangles:
        if mask[tri[0]] and mask[tri[1]] and mask[tri[2]]:
            new_tri = [old_to_new[i] for i in tri]
            valid_triangles.append(new_tri)
    
    if colors is not None and len(colors) > 0:
        cropped_colors = [colors[i] for i, include in enumerate(mask) if include]
        
        processed_mesh = o3d.geometry.TriangleMesh()
        processed_mesh.vertices = o3d.utility.Vector3dVector(vertices_frontal)
        processed_mesh.triangles = o3d.utility.Vector3iVector(valid_triangles)
        processed_mesh.vertex_colors = o3d.utility.Vector3dVector(cropped_colors)
    else:
        processed_mesh = o3d.geometry.TriangleMesh()
        processed_mesh.vertices = o3d.utility.Vector3dVector(vertices_frontal)
        processed_mesh.triangles = o3d.utility.Vector3iVector(valid_triangles)
    
    processed_mesh.compute_vertex_normals()
    if debug:
        o3d.visualization.draw_geometries([processed_mesh], window_name="Frontal Face Mesh")
        print(f"Saving frontal face to {output_ply_path}")

    os.makedirs(os.path.dirname(output_ply_path), exist_ok=True)
    o3d.io.write_triangle_mesh(output_ply_path, processed_mesh)
    
    return processed_mesh
    

def main():
    face_detector = RetinaFace("weights/det_500m.onnx")
    face_detector.det_thresh = 0.2
    
    root = r'D:\Projects\01.appliedrecog\iPhone12'
    out = r'runs'

    processed_mesh = process_face(r"D:\Projects\01.appliedrecog\Casia-3D-Face-ply-aligned-60-subjects\002\002-001.ply", 
            "runs/abc.ply", 
            face_detector, debug=True)

    return
    for i in os.listdir(root):
        if not i.endswith(".ply"):
            continue

        input_ply = fr"{root}/{i}"
        output_ply = fr"{out}/{i}"

        print(f"Processing: {input_ply}")
        try:
            processed_mesh = process_face(input_ply, output_ply, face_detector, debug=True)
            
            if processed_mesh is not None:
                print("Face rotated to frontal view successfully!")
            else:
                print("Face processing failed!")
        except Exception as e:
            print(f"Error processing face: {str(e)}")
        # break
        
if __name__ == "__main__":
    main()