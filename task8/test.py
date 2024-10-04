import open3d as o3d

# Load the mesh from file
mesh = o3d.io.read_triangle_mesh("results/mesh_aligned.obj")
#mesh = o3d.io.read_triangle_mesh("25k_doors/amp.obj")
# Create a coordinate frame for reference axes
axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])

# Visualize the mesh along with the reference axes
o3d.visualization.draw_geometries([mesh, axes], 
                                  window_name="Mesh with Reference Axes",
                                  width=800, height=600,
                                  mesh_show_back_face=True)
