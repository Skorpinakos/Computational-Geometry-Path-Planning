import open3d as o3d
import numpy as np

# Function to find boundary edges
def find_boundary_edges(mesh):
    # Get triangles and edges from the mesh
    triangles = np.asarray(mesh.triangles)
    edges = []
    
    # Add each triangle's edges
    for triangle in triangles:
        edges.append(tuple(sorted([triangle[0], triangle[1]])))
        edges.append(tuple(sorted([triangle[1], triangle[2]])))
        edges.append(tuple(sorted([triangle[2], triangle[0]])))

    # Find boundary edges (those that appear only once)
    edges = np.array(edges)
    unique, counts = np.unique(edges, axis=0, return_counts=True)
    boundary_edges = unique[counts == 1]
    
    return boundary_edges

# Function to find holes
def find_holes(mesh):
    boundary_edges = find_boundary_edges(mesh)
    
    # Create a dictionary of adjacency list
    edge_dict = {}
    for edge in boundary_edges:
        for v in edge:
            if v not in edge_dict:
                edge_dict[v] = set()
            edge_dict[v].update(edge)
            edge_dict[v].remove(v)
    
    # Find connected components (i.e., holes)
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

# Function to select and visualize hole
def visualise_largest_hole(mesh, holes):
    # Sort holes by their size and pick the second-largest hole
    sorted_holes = sorted(holes, key=len, reverse=True)
    print(len(sorted_holes)," holes found.")


    sphere_list = []
    for i in range(len(sorted_holes)):
        hole = sorted_holes[i]  # Pick hole based on size ranking
        
        if i == 0:
            # Create a point cloud for visualization of the hole vertices
            hole_vertices = np.asarray(mesh.vertices)[hole]
            large_hole_vertices=hole_vertices.copy()
            
            # Create red spheres at each hole vertex
            
            for vertex in hole_vertices:
                sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.1)
                sphere.paint_uniform_color([1, 0, 0])  # Red color
                sphere.translate(vertex)
                sphere_list.append(sphere)
        else:
            # Create a point cloud for visualization of the hole vertices
            hole_vertices = np.asarray(mesh.vertices)[hole]
            
            # Create red spheres at each hole vertex
            
            for vertex in hole_vertices:
                sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.02)
                sphere.paint_uniform_color([0, 1, 0])  # Red color
                sphere.translate(vertex)
                sphere_list.append(sphere)
        
        # Set the mesh to grey color
    mesh.paint_uniform_color([0.5, 0.5, 0.5])  # Grey color
        
        # Visualize the mesh and spheres
    o3d.visualization.draw_geometries([mesh] + sphere_list)

    return large_hole_vertices

# Main execution
if __name__ == "__main__":
    # Load the mesh
    mesh = o3d.io.read_triangle_mesh("results/mesh_aligned.obj") 
    #mesh = o3d.io.read_triangle_mesh("results/closed_holes_mesh.obj")   

    # Ensure the mesh is watertight or clean up before processing (if needed)
    mesh.remove_duplicated_vertices()
    mesh.remove_duplicated_triangles()  # Might raise warnings for UVs, can be ignored
    mesh.remove_non_manifold_edges()    # Might raise warnings for UVs, can be ignored

    # Find holes in the mesh
    holes = find_holes(mesh)
    
    # Visualize the second-largest hole with red spheres on vertices
    hole_vertices = visualise_largest_hole(mesh, holes)
