import numpy as np
from vvrpywork.shapes import (
    Point3D, Line3D, Arrow3D, Sphere3D, Cuboid3D, Cuboid3DGeneralized,
    PointSet3D, LineSet3D, Mesh3D
)
import open3d as o3d

def merge_close_vertices(mesh: Mesh3D, tolerance=1e-5):
    # Get vertices and triangles
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)

    # Use a KDTree to find close vertices
    kdtree = o3d.geometry.KDTreeFlann(mesh._shape)
    merged_vertices = []
    vertex_map = {}

    for i, vertex in enumerate(vertices):
        if i in vertex_map:
            continue

        # Search for points within the tolerance distance
        [k, idx, _] = kdtree.search_radius_vector_3d(vertex, tolerance)

        if k > 1:
            # If there are multiple points within the radius, merge them
            grouped_vertices = vertices[idx, :]
            new_vertex = np.mean(grouped_vertices, axis=0)
            merged_vertices.append(new_vertex)
            new_index = len(merged_vertices) - 1
            for j in idx:
                vertex_map[j] = new_index
        else:
            # If no close points, keep the vertex as is
            merged_vertices.append(vertex)
            vertex_map[i] = len(merged_vertices) - 1

    # Reconstruct the triangles using the new vertex indices
    new_triangles = []
    for tri in triangles:
        new_tri = [vertex_map[vi] for vi in tri]
        new_triangles.append(new_tri)

    # Create a new Mesh3D object with the merged vertices and triangles
    new_mesh = Mesh3D(color=mesh.color)  # Use the same color
    new_mesh.vertices = merged_vertices
    new_mesh.triangles = new_triangles

    # Recompute normals if needed
    if mesh._shape.has_vertex_normals():
        new_mesh._shape.compute_vertex_normals()  # Compute the normals
        new_mesh.vertex_normals = np.asarray(new_mesh._shape.vertex_normals)  # Assign the computed normals

    return new_mesh

def find_boundary_edges(mesh3d):
    # Using the triangles property method to access triangle data
    triangles = mesh3d.triangles
    edges = []

    for triangle in triangles:
        edges.append(tuple(sorted([triangle[0], triangle[1]])))
        edges.append(tuple(sorted([triangle[1], triangle[2]])))
        edges.append(tuple(sorted([triangle[2], triangle[0]])))

    edges = np.array(edges)
    unique, counts = np.unique(edges, axis=0, return_counts=True)
    boundary_edges = unique[counts == 1]
    
    return boundary_edges

def find_holes(mesh3d):
    boundary_edges = find_boundary_edges(mesh3d)
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

def find_largest_hole(holes,index=0):
    if holes:
        sorted_holes = sorted(holes, key=len, reverse=True)
        largest_hole = sorted_holes[index]
        return largest_hole
    return []

import open3d as o3d
import numpy as np

def visualize_hole(mesh3d, hole_indices):
    print("Largest hole vertices indices:", hole_indices)

    # Extract the vertices corresponding to the largest hole
    hole_vertices = np.asarray(mesh3d.vertices)[hole_indices]

    # Create a point cloud for visualization of the hole vertices
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(hole_vertices)
    pcd.paint_uniform_color([1, 0, 0])  # Red color for the hole vertices

    # Convert Mesh3D to Open3D TriangleMesh for visualization
    triangle_mesh = o3d.geometry.TriangleMesh()
    triangle_mesh.vertices = o3d.utility.Vector3dVector(np.asarray(mesh3d.vertices))
    triangle_mesh.triangles = o3d.utility.Vector3iVector(np.asarray(mesh3d.triangles))
    triangle_mesh.paint_uniform_color([0.5, 0.5, 0.5])  # Grey color for the base mesh

    # Visualize the mesh and the point cloud highlighting the hole
    o3d.visualization.draw_geometries([triangle_mesh, pcd])


if __name__ == "__main__":
    #mesh3d = Mesh3D("25k_door/amp.obj")
    mesh3d = Mesh3D("results/mesh_aligned.obj")
    mesh3d=merge_close_vertices(mesh3d,0.001)

    # Prepare the mesh (ensure no duplicated vertices, etc.)
    mesh3d.remove_duplicated_vertices()
    mesh3d.remove_unreferenced_vertices()

    # Find holes and find the largest one
    holes = find_holes(mesh3d)
    print(len(holes))
    largest_hole = find_largest_hole(holes,0)
    
    # Visualize the largest hole
    if largest_hole:
        visualize_hole(mesh3d, largest_hole)
    else:
        print("No holes found.")