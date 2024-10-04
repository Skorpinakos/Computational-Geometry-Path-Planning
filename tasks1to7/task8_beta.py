import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import time

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pathfinding.core.diagonal_movement import DiagonalMovement
from pathfinding.core.grid import Grid
from pathfinding.finder.a_star import AStarFinder

def load_image_and_find_path(image_path):
    # Load image and convert it into numpy array
    img = Image.open(image_path)
    data = np.array(img)

    # Identify special pixels
    red = np.where((data[:, :, 0] == 255) & (data[:, :, 1] == 0) & (data[:, :, 2] == 0))
    green = np.where((data[:, :, 0] == 0) & (data[:, :, 1] == 255) & (data[:, :, 2] == 0))
    start = (red[0][0], red[1][0])
    goal = (green[0][0], green[1][0])

    # Create a graph
    rows, cols, _ = data.shape
    G = nx.grid_graph(dim=[rows, cols])

    # Remove nodes that are barriers (black pixels)
    black = np.where((data[:, :, 0] == 0) & (data[:, :, 1] == 0) & (data[:, :, 2] == 0))
    for i in range(len(black[0])):
        if (black[0][i], black[1][i]) in G:
            G.remove_node((black[0][i], black[1][i]))

    # Find path using A*
    path = nx.astar_path(G, start, goal, heuristic=lambda a, b: np.linalg.norm(np.array(a) - np.array(b)))

    # Modify original image to show the path in blue
    for pixel in path:
        if data.shape[2] == 4:  # Check if the image has an alpha channel
            data[pixel[0], pixel[1]] = [0, 0, 255, 255]  # Set path as blue with full opacity
        else:
            data[pixel[0], pixel[1]] = [0, 0, 255]  # Set path as blue

    # Display the result
    plt.imshow(data)
    plt.show()

    return path

def pixel_to_world(px, py, min_x, max_x, min_y, max_y, image_width, image_height):
    """
    Convert pixel coordinates to world coordinates.
    
    Parameters:
    - px, py: Pixel coordinates.
    - min_x, max_x, min_y, max_y: Real-world coordinate boundaries of the PCD.
    - image_width, image_height: Dimensions of the image in pixels.
    
    Returns:
    - (x, y): A tuple of real-world coordinates.
    """
    # Calculate scale factors
    x_scale = (max_x - min_x) / image_width
    y_scale = (max_y - min_y) / image_height
    
    # Convert pixel to world coordinates
    x = min_x + px * x_scale/0.7
    y = max_y - py * y_scale/0.7  # This ensures the y-axis is inverted correctly for typical image coordinates
    
    return (x, y)

def world_to_pixel(x, y, min_x, max_x, min_y, max_y, image_width, image_height):
    """
    Convert world coordinates to pixel coordinates.
    
    Parameters:
    - x, y: World coordinates.
    - min_x, max_x, min_y, max_y: Real-world coordinate boundaries of the PCD.
    - image_width, image_height: Dimensions of the image in pixels.
    
    Returns:
    - (px, py): A tuple of pixel coordinates.
    """
    # Calculate inverse scale factors
    x_inv_scale = image_width / (max_x - min_x)
    y_inv_scale = image_height / (max_y - min_y)
    
    # Convert world to pixel coordinates
    px = (x - min_x) * x_inv_scale*0.7
    py = (max_y - y) * y_inv_scale *0.7 # This correctly maps the world y-coordinate into the pixel y-coordinate system
    
    return (int(px), int(py))


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
def grid_2d():
    # Load the first point cloud
    pcd_filename = "results/pcd_aligned.ply"
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

    # Load and transform the second point cloud
    second_pcd_filename = "results/filtered_156k.pcd"  
    second_pcd = o3d.io.read_point_cloud(second_pcd_filename)

    # Adjust second point cloud's minimum Z to 0
    second_min_z = np.min(np.asarray(second_pcd.points)[:, 2])
    second_translation_matrix = np.eye(4)
    second_translation_matrix[2, 3] = -second_min_z
    second_pcd.transform(second_translation_matrix)

    # Filter out points above the median Z value
    second_z_values = np.asarray(second_pcd.points)[:, 2]
    median_z = np.median(second_z_values)
    filtered_second_pcd = second_pcd.select_by_index(np.where(second_z_values <= median_z*1.75)[0]) #remove ceiling lights basically

    # DPI and figure dimensions calculation
    dpi = 100
    height_px = 200
    aspect_ratio = (max_bound[0] - min_bound[0]) / (max_bound[1] - min_bound[1])
    fig_width = height_px * aspect_ratio / dpi
    fig_height = height_px / dpi

    image_width = int(fig_width * dpi)  # convert width from inches to pixels
    image_height = int(fig_height * dpi)  # convert height from inches to pixels

    # Create figure with the calculated dimensions and DPI
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=dpi)
    colors = plt.cm.jet(np.linspace(0, 1, len(vertical_walls)))
    point_size=height_px//40

    for i, wall in enumerate(vertical_walls):
        points = np.asarray(wall.points)
        points_2d = points[:, :2]
        ax.scatter(points_2d[:, 0], points_2d[:, 1], color='black', label=f"Wall {i+1}", s=point_size)

    second_points_2d = np.asarray(filtered_second_pcd.points)[:, :2]
    ax.scatter(second_points_2d[:, 0], second_points_2d[:, 1], color='black', label='Filtered Second PCD', s=point_size)

    ax.set_xlim([min_bound[0], max_bound[0]])
    ax.set_ylim([min_bound[1], max_bound[1]])

    # Hide the axes
    ax.axis('off')

    # Save the figure
    img_path='results/layout.png'
    plt.savefig(img_path, bbox_inches='tight', pad_inches=0, transparent=True)
    plt.close(fig)  



    # Example usage of the pixel_to_world and world_to_pixel functions

    #

    # Example pixel coordinates (e.g., center of the image)
    #px, py = image_width // 2, image_height // 2

    # Convert pixel coordinates to world coordinates
    #world_x, world_y = pixel_to_world(px, py, min_bound[0], max_bound[0], min_bound[1], max_bound[1], image_width, image_height)
    #print(f"Pixel coordinates ({px}, {py}) correspond to world coordinates ({world_x}, {world_y})")


    # Convert world coordinates back to pixel coordinates
    #px, py = world_to_pixel(world_x, world_y, min_bound[0], max_bound[0], min_bound[1], max_bound[1], image_width, image_height)
    #print(f"World coordinates ({world_x}, {world_y}) correspond to pixel coordinates ({px}, {py})")
    return img_path,min_bound[0], max_bound[0], min_bound[1], max_bound[1], image_width, image_height

def paint_points_on_image(image_path, point1, point2, bounds, image_dimensions):
    # Load the image
    img = Image.open(image_path)
    pixels = np.array(img)

    # Convert any transparent pixels to white
    if pixels.shape[2] == 4:  # Check if the image has an alpha channel
        alpha_channel = pixels[:, :, 3]
        transparent_mask = alpha_channel == 0
        pixels[transparent_mask] = [255, 255, 255, 255]  # Set transparent pixels to white with full opacity

    min_x, max_x, min_y, max_y = bounds
    image_width, image_height = image_dimensions

    # Convert world coordinates to pixel coordinates
    px1, py1 = world_to_pixel(point1[0], point1[1], min_x, max_x, min_y, max_y, image_width, image_height)
    px2, py2 = world_to_pixel(point2[0], point2[1], min_x, max_x, min_y, max_y, image_width, image_height)

    # Ensure the coordinates are within image boundaries
    px1, py1 = min(max(px1, 0), image_width - 1), min(max(py1, 0), image_height - 1)
    px2, py2 = min(max(px2, 0), image_width - 1), min(max(py2, 0), image_height - 1)

    # Set the pixels to red and green respectively
    pixels[py1, px1] = [255, 0, 0, 255]  # Red with full opacity
    pixels[py2, px2] = [0, 255, 0, 255]  # Green with full opacity

    # Create a new image from the modified array
    updated_image = Image.fromarray(pixels)
    updated_image_path = 'results/updated_layout.png'
    updated_image.save(updated_image_path)

    return updated_image_path



def main(point1,point2):
    t1=time.time()
    image_path, min_x, max_x, min_y, max_y, image_width, image_height = grid_2d()

    bounds = (min_x, max_x, min_y, max_y)
    print(bounds)
    image_dimensions = (image_width, image_height)
    updated_image_path = paint_points_on_image(image_path, point1, point2, bounds, image_dimensions)
    path = load_image_and_find_path(updated_image_path)
    #print(time.time()-t1)
    #print(path)
    rw_path=[]
    for i in path:
        rw_path.append(pixel_to_world(i[1], i[0], min_x, max_x, min_y, max_y, image_width, image_height))
    return rw_path


if __name__ == "__main__":
    point1 = (1.0, 2.0)  # Example real-world coordinates for point 1
    point2 = (3.0, 4.0)  # Example real-world coordinates for point 2

    print(main(point1,point2))