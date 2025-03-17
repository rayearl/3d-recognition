import numpy as np
import open3d as o3d
import cv2


def filter_points_by_z_distance(vertices, colors, max_distance_cm=100.0):
    min_z = np.min(vertices[:, 2])
    print(min_z)
    print(np.max(vertices[:, 2])
    )
    max_allowed_z = min_z + max_distance_cm
    
    z_mask = vertices[:, 2] <= max_allowed_z
    
    filtered_vertices = vertices[z_mask]
    filtered_colors = colors[z_mask]
    
    return filtered_vertices, filtered_colors


def render_mesh_to_image(vertices, triangles, colors=None, img_size=(512, 512)):
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(triangles)
    
    if colors is not None:
        mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
    
    mesh.compute_vertex_normals()
    
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=img_size[0], height=img_size[1], visible=False)
    vis.add_geometry(mesh)
    
    ctr = vis.get_view_control()
    ctr.set_front([0, 0, 1])
    ctr.set_lookat([0, 0, 0])
    ctr.set_up([0, 1, 0])
    ctr.set_zoom(0.8)
    
    vis.update_geometry(mesh)
    vis.poll_events()
    vis.update_renderer()
    
    img = np.asarray(vis.capture_screen_float_buffer())
    
    cam_params = ctr.convert_to_pinhole_camera_parameters()
    intrinsic = cam_params.intrinsic.intrinsic_matrix
    extrinsic = cam_params.extrinsic
    
    depth_image = np.asarray(vis.capture_depth_float_buffer())
    
    vis.destroy_window()
    
    img = (img * 255).astype(np.uint8)
    return img, depth_image, intrinsic, extrinsic


def align_3d_face(vertices, keypoints_3d):
    nose_tip = keypoints_3d[2]
    center = np.mean(keypoints_3d[1:], axis=0)
    direction = center - nose_tip
    rotation_matrix = o3d.geometry.get_rotation_matrix_from_xyz([-direction[1], direction[0], 0])
    vertices_rotated = np.dot(vertices - nose_tip, rotation_matrix.T) + nose_tip
    return vertices_rotated

def crop_sphere(vertices, center, radius=80):
    dists = np.linalg.norm(vertices - center, axis=1)
    mask = dists <= radius
    return vertices[mask], mask

def unproject_2d_to_3d(landmarks_2d, depth_map, intrinsic, extrinsic):
    points_3d = []
    for (x, y) in landmarks_2d:
        depth = depth_map[y, x]  # Get depth from depth map
        uv_homogeneous = np.array([x, y, 1.0])
        cam_space = np.linalg.inv(intrinsic) @ (uv_homogeneous * depth)
        world_space = np.linalg.inv(extrinsic) @ np.append(cam_space, 1)
        points_3d.append(world_space[:3])
    return points_3d

def snap_to_mesh_surface(unprojected_points, mesh, intrinsic, extrinsic):
    mesh_ray_tracer = o3d.t.geometry.RaycastingScene()
    mesh_ray_tracer.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))

    snapped_points = []
    
    camera_origin = np.linalg.inv(extrinsic)[:3, 3]
    origins = []
    directions = []
    
    for point in unprojected_points:
        direction = (point - camera_origin)
        direction /= np.linalg.norm(direction)
        origins.append(camera_origin)
        directions.append(direction)

    origins_np = np.array(origins, dtype=np.float32)
    directions_np = np.array(directions, dtype=np.float32)

    rays_np = np.hstack([origins_np, directions_np])  # [N, 6]
    rays_tensor = o3d.core.Tensor(rays_np, dtype=o3d.core.Dtype.Float32)

    result = mesh_ray_tracer.cast_rays(rays_tensor)

    for i in range(len(unprojected_points)):
        if result["t_hit"].isfinite()[i]:
            t_hit = result["t_hit"][i].item()
            intersection_point = origins_np[i] + t_hit * directions_np[i]
            snapped_points.append(intersection_point)
        else:
            snapped_points.append(unprojected_points[i])

    return np.array(snapped_points)

def rotate_180_x(ptc):
    R_x = np.array([[1, 0,  0],
                    [0, -1, 0],
                    [0, 0, -1]])
    return np.dot(ptc, R_x.T)
