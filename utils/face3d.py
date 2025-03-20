import numpy as np
from scipy.linalg import svd
import open3d as o3d
import math


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
