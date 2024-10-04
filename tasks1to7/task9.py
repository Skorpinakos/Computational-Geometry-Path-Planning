import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from sklearn.neighbors import KDTree
from scipy.spatial import ConvexHull


def detect_vertical_planes_v2(pcd, distance_threshold=0.4, ransac_n=3, num_iterations=1000, seed=1066584):
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
        
        inlier_cloud = remaining_pcd.select_by_index(inliers)
        
        if abs(normal_vector[2]) < 0.1:  # Check if plane is vertical
            inlier_cloud.paint_uniform_color(np.random.uniform(0, 1, 3))
            vertical_planes.append(inlier_cloud)
            remaining_pcd = remaining_pcd.select_by_index(inliers, invert=True)
        else:
            num_to_remove = len(inliers)
            inliers_to_remove = np.random.choice(inliers, num_to_remove, replace=False)
            remaining_pcd = remaining_pcd.select_by_index(inliers_to_remove, invert=True)

    return vertical_planes

def get_walls(pcd_filename="results/pcd_aligned.ply"):
    # Load the first point cloud
    pcd = o3d.io.read_point_cloud(pcd_filename)

    min_z = np.min(np.asarray(pcd.points)[:, 2])
    translation_matrix = np.eye(4)
    translation_matrix[2, 3] = -min_z

    scale_matrix = np.diag([1, 1, 3])
    scaling_transformation_matrix = np.eye(4)
    scaling_transformation_matrix[:3, :3] = scale_matrix

    combined_transformation_matrix = scaling_transformation_matrix @ translation_matrix
    pcd.transform(combined_transformation_matrix)

    min_bound = np.min(np.asarray(pcd.points), axis=0)
    max_bound = np.max(np.asarray(pcd.points), axis=0)
    center = (min_bound + max_bound) / 2

    filtered_pcd = pcd.select_by_index(np.where(np.asarray(pcd.points)[:, 2] <= center[2])[0])
    vertical_walls = detect_vertical_planes_v2(filtered_pcd)
    
    return vertical_walls

def region_growing_2d(points_2d, threshold_distance=0.2, max_points=1500000, min_points=10000):
    num_points = len(points_2d)
    processed = np.zeros(num_points, dtype=bool)
    region_arrays = []

    # Using a KDTree for efficient nearest neighbor search in 2D
    tree = KDTree(points_2d)

    while not all(processed):
        seed_indices = np.where(~processed)[0]
        if not len(seed_indices):
            break
        seed_index = seed_indices[0]

        points_to_check = [seed_index]
        current_region = []

        while points_to_check and len(current_region) < max_points:
            current_point_index = points_to_check.pop(0)
            if not processed[current_point_index]:
                processed[current_point_index] = True
                current_region.append(current_point_index)
                current_point = points_2d[current_point_index]

                # Query nearby points using the KDTree in 2D
                idx = tree.query_radius([current_point], r=threshold_distance)[0]
                idx = [i for i in idx if not processed[i]]

                points_to_check.extend(idx)

        if current_region and len(current_region) >= min_points:
            region_points = points_2d[current_region]
            region_arrays.append(region_points)

    return region_arrays
def flatten_walls_to_2d(walls):
    all_points_2d = []
    for wall in walls:
        points = np.asarray(wall.points)
        points_2d = points[:, :2]  # Ignore the Z-coordinate
        all_points_2d.append(points_2d)
    
    return np.vstack(all_points_2d)
def plot_region_growing_results_2d(region_arrays):
    pass


import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
def plot_region_growing_with_contours(region_arrays):
    contours = []
    plt.figure(figsize=(10, 10))

    # Loop over each region and plot the points and convex hulls
    for i, region in enumerate(region_arrays):

        # Convert region to NumPy array for easier indexing
        region = np.array(region)
        
        # Check if the region contains valid points
        if len(region) == 0:
            continue  # Skip empty regions

        # Plot the points of the region
        plt.plot(region[:, 0], region[:, 1], 'o', markersize=5, label=f'Region {i+1} Points')

        # Compute the convex hull for the region if there are enough points
        if len(region) > 2:  # ConvexHull needs at least 3 points
            hull = ConvexHull(region)

            # Compute the centroid of the region
            centroid = np.mean(region, axis=0)

            # Get the convex hull vertices
            hull_vertices = region[hull.vertices]

            # Compute the distance of each vertex from the centroid
            distances = np.linalg.norm(hull_vertices - centroid, axis=1)

            # Normalize distances to [0, 1] range for scaling factor computation
            min_distance = np.min(distances)
            max_distance = np.max(distances)
            normalized_distances = (distances - min_distance) / (max_distance - min_distance)

            # Define scaling factors: closer points get a larger scale, farther points get a smaller scale
            # For example, you can scale the closest point by 1.4 and the farthest by 1.1
            scaling_factors = 1.6 - 0.4 * normalized_distances  # Closer points scale more

            # Enlarge the convex hull by applying variable scaling
            enlarged_hull_points = centroid + scaling_factors[:, np.newaxis] * (hull_vertices - centroid)

            # Compute the convex hull again for the enlarged points
            enlarged_hull = ConvexHull(enlarged_hull_points)
            contours.append(enlarged_hull)  # Return ConvexHull object for enlarged points

            # Plot the enlarged convex hull by connecting its vertices in order
            plt.plot(np.append(enlarged_hull_points[:, 0], enlarged_hull_points[0, 0]),
                     np.append(enlarged_hull_points[:, 1], enlarged_hull_points[0, 1]), 
                     'r-', lw=2, label=f'Region {i+1} Enlarged Convex Hull')

    plt.title('Region Growing with Scaled Enlarged Contours')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    #plt.legend()
    plt.show()

    return contours


def get_wall_convex_hulls():
    walls=get_walls()

    # Flatten the walls to 2D
    points_2d = flatten_walls_to_2d(walls)

    # Apply region growing in 2D
    regions = region_growing_2d(points_2d, threshold_distance=0.2, max_points=800, min_points=100)
    print(len(regions))

    # Plot the region growing results in 2D
    plot_region_growing_results_2d(regions)
    return plot_region_growing_with_contours(regions)
