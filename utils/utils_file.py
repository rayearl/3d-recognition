import re
import numpy as np
import open3d as o3d
from scipy.spatial import Delaunay
from matplotlib import cm

def read_ply(filename):
    vertices = []
    colors = []
    triangles = []
    
    with open(filename, 'r') as f:
        lines = f.readlines()
        
    reading_vertices = True
    
    for line in lines:
        values = line.strip().split()
        
        if not values:
            continue
            
        if values[0] == '3':
            reading_vertices = False
            triangles.append([int(values[1]), int(values[2]), int(values[3])])
        elif reading_vertices and len(values) == 7:
            x, y, z = float(values[0]), float(values[1]), float(values[2])
            r, g, b = int(values[3])/255.0, int(values[4])/255.0, int(values[5])/255.0
            
            vertices.append([x, y, z])
            colors.append([r, g, b])
    
    return np.array(vertices), np.array(triangles), np.array(colors)


def read_wrl(filename):
    with open(filename, "r", encoding="utf-8") as file:
        content = file.read()

    vertex_pattern = re.search(r'point\s*\[\s*([\d\.\-\s,\n]+?)\s*\]', content, re.DOTALL)
    vertices = []
    if vertex_pattern:
        points = vertex_pattern.group(1).strip().split(",")
        vertices = [list(map(float, p.split())) for p in points if p.strip()]
    
    face_pattern = re.search(r'coordIndex\s*\[(.*?)\]', content, re.DOTALL)
    triangles = []
    if face_pattern:
        indices = face_pattern.group(1).strip().replace("\n", " ").split(",")
        indices = [int(i) for i in indices if i.strip().isdigit()]
        triangles = [indices[i:i+3] for i in range(0, len(indices)-2, 3) if -1 not in indices[i:i+3]]

    colors = []
    color_pattern = re.search(r'color\s*\[\s*([\d\.\s,\n]+?)\s*\]', content, re.DOTALL)
    if color_pattern:
        color_data = color_pattern.group(1).strip().split(",")
        colors = [list(map(float, c.split())) for c in color_data if c.strip()]
    
    color_index = []
    color_index_pattern = re.search(r'colorIndex\s*\[(.*?)\]', content, re.DOTALL)
    if color_index_pattern:
        indices = color_index_pattern.group(1).strip().replace("\n", " ").split(",")
        color_index = [int(i) for i in indices if i.strip().isdigit() and i.strip() != "-1"]
    
    return np.array(vertices, dtype=np.float64), np.array(triangles), np.array(colors, dtype=np.float64)


def sort_vertices_into_grid(vertices, grid_shape):
    num_rows, num_cols = grid_shape
    if len(vertices) != num_rows * num_cols:
        raise ValueError(f"Số lượng điểm ({len(vertices)}) không khớp với lưới ({num_rows}x{num_cols})")

    # Sắp xếp theo y trước, rồi x
    sorted_indices = np.lexsort((vertices[:, 0], vertices[:, 1]))
    sorted_vertices = vertices[sorted_indices]
    
    # Tạo ánh xạ từ chỉ số cũ sang chỉ số mới
    index_map = np.zeros(len(vertices), dtype=int)
    for new_idx, old_idx in enumerate(sorted_indices):
        index_map[old_idx] = new_idx
    
    return sorted_vertices, index_map

def remap_triangles(triangles, index_map):
    return np.vectorize(lambda i: index_map[i])(triangles)


def crete_mesh(vertices, triangles):
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(triangles)
    mesh.compute_vertex_normals()
    return mesh

def show_mesh(vertices, triangles, kps3D=None):
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(triangles)
    mesh.compute_vertex_normals()

    geometries = [mesh]

    if kps3D is not None and len(kps3D) > 0:
        keypoint_cloud = o3d.geometry.PointCloud()
        keypoint_cloud.points = o3d.utility.Vector3dVector(kps3D)

        keypoint_cloud.paint_uniform_color([1, 0, 0])

        geometries.append(keypoint_cloud)

    o3d.visualization.draw_geometries(geometries)
