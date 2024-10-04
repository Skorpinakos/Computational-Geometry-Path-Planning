import open3d as o3d

# Load the point cloud
pcd = o3d.io.read_point_cloud("results/pcd_door.ply")

# Downsample the point cloud for faster processing
voxel_down_pcd = pcd.voxel_down_sample(voxel_size=0.05)

# Initialize the list to store detected planes
detected_planes = []
colors = [[1, 0, 0], [0, 1, 0], [1, 0, 1], [1, 1, 0], [0, 1, 1]]  # Colors for visualization

# Set parameters for plane detection
distance_threshold = 0.15  # Adjust based on precision needs
num_iterations = 2000      # Increase for more accurate plane fitting

# Loop to extract multiple planes
max_planes = 2  # Set the max number of planes to detect
current_pcd = voxel_down_pcd
for i in range(max_planes):
    # Segment the largest plane from the remaining points
    plane_model, inliers = current_pcd.segment_plane(distance_threshold=distance_threshold,
                                                     ransac_n=3,
                                                     num_iterations=num_iterations)
    
    if len(inliers) < 50:
        print(f"Plane {i+1} has too few inliers, stopping plane extraction.")
        break  # Stop if the plane has too few inliers

    # Extract the inliers (plane points)
    inlier_cloud = current_pcd.select_by_index(inliers)
    inlier_cloud.paint_uniform_color(colors[i % len(colors)])  # Assign a unique color
    
    # Save the detected plane
    detected_planes.append(inlier_cloud)
    
    # Remove the inliers from the point cloud to detect the next plane
    current_pcd = current_pcd.select_by_index(inliers, invert=True)

# Visualize the detected planes
o3d.visualization.draw_geometries(detected_planes + [current_pcd])
