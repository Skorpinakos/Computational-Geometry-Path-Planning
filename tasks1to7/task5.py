import open3d as o3d
import numpy as np

def visualize_growth_process(pcd):
    o3d.visualization.draw_geometries([pcd], window_name="Multiple Regions Visualization")

def region_growing(pcd, threshold_distance=0.5, max_points=5000):
    points = np.asarray(pcd.points)
    num_points = len(points)
    processed = np.zeros(num_points, dtype=bool)
    regions = []

    while not all(processed):
        # Find a new seed point which has not been processed
        seed_indices = np.where(~processed)[0]
        if not len(seed_indices):
            break
        seed_index = seed_indices[0]

        # Initialize the region growing for this seed
        points_to_check = [seed_index]
        current_region = []

        while points_to_check and len(current_region) < max_points:
            current_point_index = points_to_check.pop(0)
            if not processed[current_point_index]:
                processed[current_point_index] = True
                current_region.append(current_point_index)
                current_point = points[current_point_index]
                distances = np.linalg.norm(points - current_point, axis=1)
                close_points = np.where((distances < threshold_distance) & (~processed))[0]
                points_to_check.extend(close_points)

        if current_region:
            region_pcd = o3d.geometry.PointCloud()
            region_pcd.points = o3d.utility.Vector3dVector(points[current_region])
            region_pcd.paint_uniform_color(np.random.rand(3).tolist())  # Random color for each region
            regions.append(region_pcd)

    # Combine all regions into one point cloud for visualization
    combined_pcd = o3d.geometry.PointCloud()
    for region in regions:
        combined_pcd += region

    return combined_pcd


# Load your point cloud
pcd = o3d.io.read_point_cloud("results/filtered_156k.pcd")

# Process the point cloud to segment multiple regions
segmented_pcd = region_growing(pcd)

# Visualize the result
visualize_growth_process(segmented_pcd)