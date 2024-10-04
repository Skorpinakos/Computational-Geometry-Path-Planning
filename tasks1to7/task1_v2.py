import open3d as o3d
import numpy as np
from sklearn.decomposition import PCA
from scipy.spatial import Delaunay
from sklearn.cluster import KMeans

# Function to find boundary edges
def find_boundary_edges(mesh):
    triangles = np.asarray(mesh.triangles)
    edges = []
    
    for triangle in triangles:
        edges.append(tuple(sorted([triangle[0], triangle[1]])))
        edges.append(tuple(sorted([triangle[1], triangle[2]])))
        edges.append(tuple(sorted([triangle[2], triangle[0]])))
    
    edges = np.array(edges)
    unique, counts = np.unique(edges, axis=0, return_counts=True)
    boundary_edges = unique[counts == 1]
    
    return boundary_edges

# Function to find holes in the mesh
def find_holes(mesh):
    boundary_edges = find_boundary_edges(mesh)
    
    edge_dict = {}
    for edge in boundary_edges:
        for v in edge:
            if v not in edge_dict:
                edge_dict[v] = set()
            edge_dict[v].update(edge)
            edge_dict[v].remove(v)
    
    visited = set()
    holes = []
    
    def dfs(v, current_hole):
        visited.add(v)
        current_hole.append(v)
        for neighbor in edge_dict.get(v, []):
            if neighbor not in visited:
                dfs(neighbor, current_hole)
    
    for vertex in edge_dict:
        if vertex not in visited:
            current_hole = []
            dfs(vertex, current_hole)
            holes.append(current_hole)
    
    return holes

# Function to close a hole by using Delaunay triangulation
def close_holes(mesh, holes):
    vertices = np.asarray(mesh.vertices)
    
    new_triangles = []

    for hole in holes:
        hole_vertices = vertices[hole]

        if len(hole_vertices) < 3:
            print(f"Skipping hole with less than 3 vertices: {hole}")
            continue

        # Project the 3D vertices onto 2D plane for triangulation
        hole_2d = hole_vertices[:, :2]  # Using XY plane

        try:
            # Perform Delaunay triangulation on the 2D boundary vertices
            tri = Delaunay(hole_2d)
            for simplex in tri.simplices:
                new_triangles.append([hole[simplex[0]], hole[simplex[1]], hole[simplex[2]]])
        except Exception as e:
            print(f"Failed to triangulate hole: {str(e)}")
    
    if len(new_triangles) > 0:
        # Add the new triangles to the mesh
        new_triangles = np.asarray(new_triangles)
        mesh_triangles = np.vstack((np.asarray(mesh.triangles), new_triangles))
        mesh.triangles = o3d.utility.Vector3iVector(mesh_triangles)
    
    return mesh

def rotate_pcd(pcd, rotation_degrees):
    """
    Rotate the point cloud by given degrees on x, y, and z axes.

    :param pcd: open3d.geometry.PointCloud object
    :param rotation_degrees: A tuple (rx, ry, rz) indicating degrees of rotation for x, y, z axes
    :return: rotated point cloud
    """
    rotation_radians = np.radians(rotation_degrees)
    R_x = np.array([[1, 0, 0],
                    [0, np.cos(rotation_radians[0]), -np.sin(rotation_radians[0])],
                    [0, np.sin(rotation_radians[0]), np.cos(rotation_radians[0])]])

    R_y = np.array([[np.cos(rotation_radians[1]), 0, np.sin(rotation_radians[1])],
                    [0, 1, 0],
                    [-np.sin(rotation_radians[1]), 0, np.cos(rotation_radians[1])]])

    R_z = np.array([[np.cos(rotation_radians[2]), -np.sin(rotation_radians[2]), 0],
                    [np.sin(rotation_radians[2]), np.cos(rotation_radians[2]), 0],
                    [0, 0, 1]])

    R = R_z @ R_y @ R_x
    pcd.rotate(R, center=(0, 0, 0))
    
    return pcd

def ransac_plane_detection(pcd, distance_threshold=0.15):
    """
    Detect the dominant plane using RANSAC and return the plane model and inliers.
    """
    plane_model, inliers = pcd.segment_plane(distance_threshold=distance_threshold,
                                             ransac_n=3,
                                             num_iterations=1000)
    return plane_model, inliers

def calculate_ceiling_angle(plane_model):
    """
    Calculate the angle between the ceiling's normal vector and the z-axis.
    """
    normal_vector = np.array(plane_model[:3])  # Extract normal vector (a, b, c)
    z_axis = np.array([0, 0, 1])  # Normal of the z-axis

    cos_theta = np.dot(normal_vector, z_axis) / (np.linalg.norm(normal_vector) * np.linalg.norm(z_axis))
    angle_radians = np.arccos(np.clip(cos_theta, -1.0, 1.0))  # Clip for numerical stability
    angle_degrees = np.degrees(angle_radians)
    
    return angle_degrees, normal_vector

def align_ceiling_to_flat(pcd, plane_model):
    """
    Rotate the point cloud such that the ceiling plane becomes flat (aligned with the xy-plane).
    """
    _, normal_vector = calculate_ceiling_angle(plane_model)
    
    z_axis = np.array([0, 0, 1])
    rotation_axis = np.cross(normal_vector, z_axis)
    rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)  # Normalize

    angle_radians = np.arccos(np.dot(normal_vector, z_axis) / np.linalg.norm(normal_vector))
    R = o3d.geometry.get_rotation_matrix_from_axis_angle(rotation_axis * angle_radians)

    pcd.rotate(R, center=(0, 0, 0))
    
    return pcd

def create_axis_cylinders(origin=np.array([0, 0, 0]), length=9.0, radius=0.2):
    """
    Create three cylinders representing the x, y, and z axes.

    :param origin: The starting point of the cylinders.
    :param length: Length of the cylinders (default 9.0).
    :param radius: Radius of the cylinders (default 0.2).
    :return: A list of open3d.geometry.TriangleMesh objects representing the x, y, and z axes.
    """
    # Create cylinders for the axes
    x_cylinder = o3d.geometry.TriangleMesh.create_cylinder(radius=radius, height=length)
    x_cylinder.paint_uniform_color([1, 0, 0])  # Red for x-axis
    x_cylinder.compute_vertex_normals()

    y_cylinder = o3d.geometry.TriangleMesh.create_cylinder(radius=radius, height=length)
    y_cylinder.paint_uniform_color([0, 1, 0])  # Green for y-axis
    y_cylinder.compute_vertex_normals()

    z_cylinder = o3d.geometry.TriangleMesh.create_cylinder(radius=radius, height=length)
    z_cylinder.paint_uniform_color([0, 0, 1])  # Blue for z-axis
    z_cylinder.compute_vertex_normals()

    # Rotate the cylinders to align with the respective axes
    # Z-axis (no rotation needed, it's already aligned)
    z_cylinder.translate(origin)

    # X-axis: rotate 90 degrees around the y-axis to align it with the x-axis
    R_x = o3d.geometry.get_rotation_matrix_from_xyz([0, np.pi / 2, 0])
    x_cylinder.rotate(R_x, center=(0, 0, 0))
    x_cylinder.translate(origin)

    # Y-axis: rotate -90 degrees around the x-axis to align it with the y-axis
    R_y = o3d.geometry.get_rotation_matrix_from_xyz([-np.pi / 2, 0, 0])
    y_cylinder.rotate(R_y, center=(0, 0, 0))
    y_cylinder.translate(origin)

    return [x_cylinder, y_cylinder, z_cylinder]

def detect_ceiling_pca_bounding_box(ceiling_pcd):
    """
    Detect the oriented bounding box using PCA for better alignment with the ceiling's shape.
    """
    # Extract the points as a NumPy array
    ceiling_points = np.asarray(ceiling_pcd.points)

    # Perform PCA to find the main axes of the ceiling points
    pca = PCA(n_components=3)
    pca.fit(ceiling_points)
    principal_axes = pca.components_

    # Calculate the centroid of the ceiling points
    centroid = np.mean(ceiling_points, axis=0)

    # Apply PCA rotation to align the ceiling with the main axes
    ceiling_pcd.rotate(principal_axes.T, center=centroid)

    # Get the oriented bounding box from the rotated point cloud
    obb = ceiling_pcd.get_oriented_bounding_box()
    obb.color = (0, 0, 1)  # Color the OBB blue for visibility

    return obb, principal_axes

def rotate_pcd_to_align_with_pca(pcd, principal_axes):
    """
    Rotate the point cloud based on the PCA-derived principal axes to align the walls.
    """
    # Rotate the point cloud using the inverse of the PCA rotation matrix
    R_inv = np.linalg.inv(principal_axes.T)
    pcd.rotate(R_inv, center=(0, 0, 0))

    return pcd

def rotate_pcd_and_mesh(pcd, mesh, rotation_degrees):
    """
    Rotate the point cloud and mesh by given degrees on x, y, and z axes.
    """
    rotation_radians = np.radians(rotation_degrees)
    R_x = np.array([[1, 0, 0],
                    [0, np.cos(rotation_radians[0]), -np.sin(rotation_radians[0])],
                    [0, np.sin(rotation_radians[0]), np.cos(rotation_radians[0])]])

    R_y = np.array([[np.cos(rotation_radians[1]), 0, np.sin(rotation_radians[1])],
                    [0, 1, 0],
                    [-np.sin(rotation_radians[1]), 0, np.cos(rotation_radians[1])]])

    R_z = np.array([[np.cos(rotation_radians[2]), -np.sin(rotation_radians[2]), 0],
                    [np.sin(rotation_radians[2]), np.cos(rotation_radians[2]), 0],
                    [0, 0, 1]])

    R = R_z @ R_y @ R_x
    pcd.rotate(R, center=(0, 0, 0))
    mesh.rotate(R, center=(0, 0, 0))  # Rotate the mesh with the same matrix
    
    return pcd, mesh

def align_ceiling_to_flat_and_mesh(pcd, mesh, plane_model):
    """
    Rotate the point cloud and mesh such that the ceiling plane becomes flat (aligned with the xy-plane).
    """
    _, normal_vector = calculate_ceiling_angle(plane_model)
    
    z_axis = np.array([0, 0, 1])
    rotation_axis = np.cross(normal_vector, z_axis)
    rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)  # Normalize

    angle_radians = np.arccos(np.dot(normal_vector, z_axis) / np.linalg.norm(normal_vector))
    R = o3d.geometry.get_rotation_matrix_from_axis_angle(rotation_axis * angle_radians)

    pcd.rotate(R, center=(0, 0, 0))
    mesh.rotate(R, center=(0, 0, 0))  # Rotate the mesh with the same matrix
    
    return pcd, mesh

def rotate_pcd_and_mesh_to_align_with_pca(pcd, mesh, principal_axes):
    """
    Rotate the point cloud and mesh based on the PCA-derived principal axes to align the walls.
    """
    R_inv = np.linalg.inv(principal_axes.T)
    pcd.rotate(R_inv, center=(0, 0, 0))
    mesh.rotate(R_inv, center=(0, 0, 0))  # Rotate the mesh with the same matrix

    return pcd, mesh

def detect_vertical_planes(pcd, distance_threshold=0.15,verticality=0.5, ransac_n=6, num_iterations=2000):
    vertical_planes = []
    remaining_pcd = pcd

    while True:
        plane_model, inliers = remaining_pcd.segment_plane(distance_threshold=distance_threshold, ransac_n=ransac_n, num_iterations=num_iterations)
        inlier_cloud = remaining_pcd.select_by_index(inliers)

        # Check if the normal is close to vertical
        a, b, c, _ = plane_model
        normal_vector = np.array([a, b, c])
        normal_vector /= np.linalg.norm(normal_vector)
        if abs(normal_vector[2]) < verticality:  # Adjust this threshold to be more or less stringent
            inlier_cloud.paint_uniform_color(np.random.uniform(0, 1, 3))  # Color the inliers for better visualization
            vertical_planes.append(inlier_cloud)
        
        remaining_pcd = remaining_pcd.select_by_index(inliers, invert=True)
        
        if len(inliers) < 100:  # Stop if the set of inliers is too small
            break

    return vertical_planes

def detect_vertical_planes_v2(pcd, distance_threshold=0.3, ransac_n=3, num_iterations=1000, seed=1066584):
    vertical_planes = []
    remaining_pcd = pcd

    np.random.seed(seed)  # Set the seed for reproducibility

    while True:
        plane_model, inliers = remaining_pcd.segment_plane(distance_threshold=distance_threshold, ransac_n=ransac_n, num_iterations=num_iterations)
        if len(inliers) < 1000:  # Stop if the set of inliers is too small
            break
        
        a, b, c, _ = plane_model
        normal_vector = np.array([a, b, c])
        normal_vector /= np.linalg.norm(normal_vector)
        
        # Temporarily store inliers from potential vertical planes
        inlier_cloud = remaining_pcd.select_by_index(inliers)
        
        if abs(normal_vector[2]) < 0.3:  # Check if plane is vertical
            inlier_cloud.paint_uniform_color(np.random.uniform(0, 1, 3))
            vertical_planes.append(inlier_cloud)
            # Remove all inliers since plane is vertical
            remaining_pcd = remaining_pcd.select_by_index(inliers, invert=True)
        else:
            # Remove only part of the inliers randomly if the plane is not vertical
            num_to_remove = len(inliers) // 4
            inliers_to_remove = np.random.choice(inliers, num_to_remove, replace=False)
            remaining_pcd = remaining_pcd.select_by_index(inliers_to_remove, invert=True)

    return vertical_planes

def get_plane_normal_from_pcd(pcd):
    points = np.asarray(pcd.points)
    # Compute the covariance matrix of the point cloud
    mean_centered = points - np.mean(points, axis=0)
    cov_matrix = np.cov(mean_centered, rowvar=False)
    # Compute the eigenvalues and eigenvectors of the covariance matrix
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    # The eigenvector corresponding to the smallest eigenvalue
    normal_vector = eigenvectors[:, np.argmin(eigenvalues)]
    return normal_vector


def angle_with_x_axis(normal):
    x_axis = np.array([1, 0, 0])
    normal = normal / np.linalg.norm(normal)  # Normalize the vector
    cos_theta = np.dot(normal, x_axis)
    angle = np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0)))  # Clip for numerical stability
    return angle


def weighted_k_means_1d(data, weights, k, iterations=100):
    # Validate input lengths
    if len(data) != len(weights):
        raise ValueError("Data and weights must be of the same length")

    # Prepare data: repeat data points according to their integer weights
    # Note: This approach works when weights are integers or can be approximated by integers
    expanded_data = np.repeat(np.array(data), np.array(weights).astype(int))

    # Create a KMeans instance with desired number of clusters and max iterations
    kmeans = KMeans(n_clusters=k, max_iter=iterations, random_state=0)

    # Reshape expanded_data for KMeans (it expects a 2D array)
    expanded_data = expanded_data.reshape(-1, 1)

    # Compute KMeans clustering
    kmeans.fit(expanded_data)

    # Extract the cluster centers and return them
    cluster_centers = kmeans.cluster_centers_.flatten()

    return cluster_centers


# Function to create a rotation matrix for rotating in the xy-plane (about the z-axis)
def rotation_matrix_z(angle):
    cos_angle = np.cos(angle)
    sin_angle = np.sin(angle)
    return np.array([
        [cos_angle, -sin_angle, 0],
        [sin_angle, cos_angle, 0],
        [0, 0, 1]
    ])

# Function to rotate both a point cloud and a mesh in the xy-plane
def rotate_xy_plane(pcd, mesh, rotation_angle):
    """
    Rotates the given point cloud and mesh in the xy-plane by the specified angle.

    :param pcd: Open3D PointCloud object
    :param mesh: Open3D TriangleMesh object
    :param rotation_angle: Rotation angle in radians (positive values rotate counterclockwise, negative values rotate clockwise)
    """
    # Get the rotation matrix for the given rotation_angle
    R = rotation_matrix_z(rotation_angle)

    # Apply rotation to the point cloud and mesh
    pcd.rotate(R, center=(0, 0, 0))
    mesh.rotate(R, center=(0, 0, 0))

    return pcd, mesh

# Load point cloud and mesh
filename="25k_doors/amp.obj"
mesh = o3d.io.read_triangle_mesh(filename)  # Load mesh
from task3 import generate_point_cloud_from_mesh
pcd_filename="results/pcd_from_task1.ply"
generate_point_cloud_from_mesh(filename,pcd_filename,150)
pcd = o3d.io.read_point_cloud(pcd_filename)


# Rotate point cloud and mesh initially by (x, y, z) degrees
rotation_degrees = (8, 30, 60)  # Example degrees of rotation
pcd, mesh = rotate_pcd_and_mesh(pcd, mesh, rotation_degrees)

# Apply RANSAC to detect the dominant plane (the ceiling)
plane_model, inliers = ransac_plane_detection(pcd, distance_threshold=0.15)

# Extract ceiling points (before alignment) and shift them in the z-axis for visualization
ceiling_pcd_before_alignment = pcd.select_by_index(inliers)
ceiling_pcd_before_alignment.paint_uniform_color([0, 1, 0])  # Paint the ceiling points green

ceiling_pcd_before_alignment.translate((0,0,-0.5))

# Visualize the ceiling points (highlighted in blue) before alignment
cylinder_aligned = create_axis_cylinders(origin=np.array([0, 0, 0]), length=20.0, radius=0.2)
o3d.visualization.draw_geometries([pcd, ceiling_pcd_before_alignment, mesh] + cylinder_aligned, 
                                  window_name="Original Point Cloud with Ceiling and Mesh")

# Rotate the entire point cloud and mesh to align the ceiling with the xy-plane
pcd_aligned, mesh_aligned = align_ceiling_to_flat_and_mesh(pcd, mesh, plane_model)




#vertical_planes = detect_vertical_planes(pcd_aligned,0.1,0.5)

vertical_planes = detect_vertical_planes_v2(pcd_aligned)
print(len(vertical_planes))

angles=[]
weights=[]
for i in vertical_planes:
    normal=get_plane_normal_from_pcd(i)
    angle=angle_with_x_axis(normal)
    weight=len(i.points)
    angles.append(angle)
    weights.append(weight)

centroids=weighted_k_means_1d(angles,weights,2)

print(centroids)
o3d.visualization.draw_geometries( vertical_planes)
rot=min(centroids)






# Extract the ceiling points after alignment
#ceiling_pcd_after_alignment = pcd_aligned.select_by_index(inliers)

# Detect the PCA-based oriented bounding box
#ceiling_obb, pca_axes = detect_ceiling_pca_bounding_box(ceiling_pcd_after_alignment)

# Rotate the point cloud and mesh to align the walls with the x and y axes using PCA
#pcd_aligned_to_walls, mesh_aligned_to_walls = rotate_pcd_and_mesh_to_align_with_pca(pcd_aligned, mesh_aligned, pca_axes)
pcd_aligned_to_walls, mesh_aligned_to_walls = rotate_xy_plane(pcd_aligned, mesh_aligned,np.radians(-rot))
# Visualize the aligned point cloud, mesh, PCA-derived OBB, and axis cylinders
cylinder_aligned = create_axis_cylinders(origin=np.array([0, 0, 0]), length=20.0, radius=0.2)
o3d.visualization.draw_geometries([pcd_aligned_to_walls, mesh_aligned_to_walls] + cylinder_aligned,window_name="Final Aligned Point Cloud and Mesh with PCA OBB and Axes")

# Save the final aligned point cloud and mesh
aligned_mesh_filename="results/mesh_aligned.obj"
o3d.io.write_point_cloud("results/pcd_aligned.ply", pcd_aligned_to_walls)
o3d.io.write_triangle_mesh(aligned_mesh_filename, mesh_aligned_to_walls)




#now to create the non hole versions



# Load the mesh
mesh = o3d.io.read_triangle_mesh(aligned_mesh_filename)  # Replace with your mesh file


# Ensure the mesh is cleaned up before processing
mesh.remove_duplicated_vertices()
mesh.remove_duplicated_triangles()
mesh.remove_non_manifold_edges()
mesh.remove_degenerate_triangles()

# Find holes in the mesh
holes = find_holes(mesh)
print(f"Found {len(holes)} holes in the mesh.")

# Close all holes using Delaunay triangulation
mesh_with_closed_holes = close_holes(mesh, holes)

# Visualize the mesh after closing holes
o3d.visualization.draw_geometries([mesh_with_closed_holes])

# Save the mesh with closed holes
o3d.io.write_triangle_mesh("results/closed_holes_mesh.obj", mesh_with_closed_holes)
generate_point_cloud_from_mesh("results/closed_holes_mesh.obj","results/closed_holes_pcd.ply",150)

import os
# Delete the .mtl file
if os.path.exists("results/mesh_aligned.mtl"):
    os.remove("results/mesh_aligned.mtl")