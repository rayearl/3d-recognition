import re
import numpy as np
import open3d as o3d
from scipy.spatial import Delaunay


def read_ply(filename):
    point_cloud = o3d.io.read_point_cloud(filename)
    
    vertices = np.asarray(point_cloud.points)
    
    colors = np.asarray(point_cloud.colors)

    if len(colors) == 0:
        colors = np.ones_like(vertices) * 0.5

    tri = Delaunay(vertices[:, :2])
    triangles = tri.simplices
    
    return vertices, triangles, colors

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

    return np.array(vertices, dtype=np.float64), np.array(triangles)


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
