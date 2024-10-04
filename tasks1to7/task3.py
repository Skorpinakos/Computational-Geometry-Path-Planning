import open3d as o3d
import numpy as np

def triangle_area(vertices):
    """
    Calculate the area of a triangle given its vertices using Heron's formula.
    
    Args:
        vertices (list of np.array): A list containing three vertices of the triangle.
        
    Returns:
        float: The area of the triangle.
    """
    v1, v2, v3 = vertices
    a = np.linalg.norm(v2 - v1)
    b = np.linalg.norm(v3 - v2)
    c = np.linalg.norm(v1 - v3)
    s = (a + b + c) / 2
    area = np.sqrt(s * (s - a) * (s - b) * (s - c))
    
    return area

def generate_point_cloud_from_mesh(input_filename, output_filename, point_density=150):
    """
    Generates a point cloud from a mesh file, with points sampled based on triangle area and saves the result.
    
    Args:
        input_filename (str): Path to the input mesh file.
        output_filename (str): Path to save the generated point cloud file.
        point_density (int): Density of points per unit area of the triangles.
    """
    # Load the mesh from the specified file path
    mesh = o3d.io.read_triangle_mesh(input_filename)

    # Ensure the mesh is valid
    if not mesh.has_triangle_normals():
        mesh.compute_triangle_normals()

    # Access vertices and triangles
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)

    # List to store points
    points = []

    # Iterate through each triangle
    for i, triangle in enumerate(triangles):
        # Get the vertices of the triangle
        v0, v1, v2 = vertices[triangle[0]], vertices[triangle[1]], vertices[triangle[2]]
        area = triangle_area([v0, v1, v2])
        points_count = int(point_density * area)
    
        vector1 = v1 - v0
        vector2 = v2 - v0
    
        for _ in range(points_count):
            r1, r2 = np.random.rand(2)
            is_ok = (r1 + r2) <= 1
            r1 = is_ok * r1 + (not is_ok) * (1 - r1)
            r2 = is_ok * r2 + (not is_ok) * (1 - r2)

            point = v0 + r1 * vector1 + r2 * vector2
            points.append(point)

    points = np.array(points)  # Convert the list of points to a numpy array

    # Create a PointCloud object
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)

    # Set the color of the point cloud to blue
    blue_color = [0, 0, 1]  # RGB for blue
    point_cloud.colors = o3d.utility.Vector3dVector(np.tile(blue_color, (len(points), 1)))

    # Visualization
    o3d.visualization.draw_geometries([point_cloud], window_name="Blue Point Cloud")

    # Save the point cloud to a file
    o3d.io.write_point_cloud(output_filename, point_cloud)

if __name__ == "__main__": 
    generate_point_cloud_from_mesh("25k_doors/amp.obj", "results/pcd_doors_task3.ply", point_density=150)
