import numpy as np
from scipy.linalg import svd
import open3d as o3d
import math
from scipy.spatial.transform import Rotation
from mpl_toolkits.mplot3d import Axes3D


def calculate_distance(point1, point2):
    return np.linalg.norm(point1 - point2)


def compute_oriented_bounding_box(points):
    center = np.mean(points, axis=0)
    
    points_centered = points - center
    
    cov_matrix = np.cov(points_centered, rowvar=False)
    
    _, _, vh = svd(cov_matrix)
    axes = vh
    
    points_aligned = np.dot(points_centered, axes.T)
    
    min_vals = np.min(points_aligned, axis=0)
    max_vals = np.max(points_aligned, axis=0)
    extents = (max_vals - min_vals) / 2
    
    center_offset = (min_vals + max_vals) / 2
    center = center + np.dot(center_offset, axes)
    
    return center, axes, extents


def get_box_vertices(center, axes, extents):
    sign_combinations = np.array([
        [-1, -1, -1],
        [-1, -1,  1],
        [-1,  1, -1],
        [-1,  1,  1],
        [ 1, -1, -1],
        [ 1, -1,  1],
        [ 1,  1, -1],
        [ 1,  1,  1]
    ])
    
    vertices_aligned = sign_combinations * extents
    vertices = np.dot(vertices_aligned, axes) + center
    
    return vertices

def visualize_obb_with_center_and_axes(vertices, center, axes, extents):
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(vertices)
    point_cloud.paint_uniform_color([0.8, 0.8, 0.8])
    
    box_vertices = get_box_vertices(center, axes, extents)
    
    box_lines = o3d.geometry.LineSet()
    box_lines.points = o3d.utility.Vector3dVector(box_vertices)
    
    box_edges = [
        [0, 1], [0, 2], [1, 3], [2, 3],
        [4, 5], [4, 6], [5, 7], [6, 7],
        [0, 4], [1, 5], [2, 6], [3, 7]
    ]
    box_lines.lines = o3d.utility.Vector2iVector(box_edges)
    box_lines.colors = o3d.utility.Vector3dVector([[0, 1, 0] for _ in range(len(box_edges))])
    
    center_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=5.0)
    center_sphere.translate(center)
    center_sphere.paint_uniform_color([1, 0, 0])
    
    axis_lines = o3d.geometry.LineSet()
    axis_points = np.vstack([
        center,
        center + axes[0] * extents[0] * 1.5,  # X
        center + axes[1] * extents[1] * 1.5,  # Y
        center + axes[2] * extents[2] * 1.5   # Z
    ])
    axis_lines.points = o3d.utility.Vector3dVector(axis_points)
    
    axis_edges = [[0, 1], [0, 2], [0, 3]]
    axis_lines.lines = o3d.utility.Vector2iVector(axis_edges)
    
    axis_colors = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    axis_lines.colors = o3d.utility.Vector3dVector(axis_colors)
    
    geometries = [point_cloud, box_lines, center_sphere, axis_lines]
    
    o3d.visualization.draw_geometries(geometries)


def transform_to_new_coordinate(vertices, keypoints, center, axes):
    vertices_centered = vertices - center
    keypoints_centered = keypoints - center
    
    transform_matrix = axes.T
    vertices_new = np.dot(vertices_centered, transform_matrix)
    keypoints_new = np.dot(keypoints_centered, transform_matrix)

    return vertices_new, keypoints_new


def rotate_to_eyes_up(vertices_aligned, keypoints_aligned):
    left_eye = keypoints_aligned[0]
    right_eye = keypoints_aligned[1]
    eye_vector = right_eye[:2] - left_eye[:2]
    angle = math.atan2(eye_vector[1], eye_vector[0])
    rotation_angle = -angle + np.pi/2 + np.pi/2
    
    cos_theta = math.cos(rotation_angle)
    sin_theta = math.sin(rotation_angle)
    rotation_matrix = np.array([
        [cos_theta, -sin_theta, 0],
        [sin_theta, cos_theta, 0],
        [0, 0, 1]
    ])
    
    vertices_rotated = np.dot(vertices_aligned, rotation_matrix)
    keypoints_rotated = np.dot(keypoints_aligned, rotation_matrix)
    
    return vertices_rotated, keypoints_rotated


def calculate_obb_angles(axes):
    standard_axes = np.eye(3)
    
    angles = np.zeros((3, 3))
    for i in range(3):  # OBB axes
        for j in range(3):  # Standard axes
            dot_product = np.dot(axes[i], standard_axes[j])
            cos_angle = np.clip(dot_product, -1.0, 1.0)
            angle_rad = np.arccos(cos_angle)
            angles[i, j] = np.degrees(angle_rad)

    min_angles = np.min(angles, axis=1)
    
    max_deviation = np.max(min_angles)
    
    min_angle_x = np.min(angles[0, :])
    min_angle_y = np.min(angles[1, :])
    min_angle_z = np.min(angles[2, :])
    
    max_angle = np.max([min_angle_x, min_angle_y, min_angle_z])
    
    avg_angle = np.mean([min_angle_x, min_angle_y, min_angle_z])
    
    return min_angle_z


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
