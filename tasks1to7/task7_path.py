import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull, Delaunay
import heapq

def a_star_pathfinding(grid, start, end):
    """
    Finds a path from start to end on a grid using the A* search algorithm.
    
    Parameters:
    - grid: A 2D numpy array where 0 represents free space and 1 represents obstacles.
    - start: The starting coordinate tuple (x, y).
    - end: The ending coordinate tuple (x, y).
    
    Returns:
    - A list of tuples representing the path from start to end if found, otherwise an empty list.
    """
    # Define movement directions (up, down, left, right, and diagonals)
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]

    # Helper function to calculate the heuristic (Manhattan distance)
    def heuristic(a, b):
        return (abs(a[0] - b[0])**2 + abs(a[1] - b[1])**2)**0.5

    # Priority queue for the open set
    open_set = []
    heapq.heappush(open_set, (0 + heuristic(start, end), 0, start))
    came_from = {}
    cost_so_far = {}
    came_from[start] = None
    cost_so_far[start] = 0

    while open_set:
        _, current_cost, current = heapq.heappop(open_set)

        if current == end:
            # Reconstruct path
            path = []
            while current != start:
                path.append(current)
                current = came_from[current]
            path.append(start)
            path.reverse()
            return path

        # Explore neighbors
        for dx, dy in directions:
            neighbor = (current[0] + dx, current[1] + dy)
            if (0 <= neighbor[0] < grid.shape[1] and 0 <= neighbor[1] < grid.shape[0]):
                if grid[neighbor[1], neighbor[0]] == 1:
                    continue  # This grid cell is blocked
                new_cost = current_cost + 1
                if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                    cost_so_far[neighbor] = new_cost
                    priority = new_cost + heuristic(neighbor, end)
                    heapq.heappush(open_set, (priority, new_cost, neighbor))
                    came_from[neighbor] = current

    return []  # Return an empty path if no path is found

import os
def create_grid_with_start_end(detail_level, start, end, convex_hulls, cache_filename="results/cached_grid.npz"):
    regenerate = False  # Flag to determine if we need to regenerate the grid

    # Check if the grid is already cached
    if os.path.exists(cache_filename):
        # Load the cached grid and parameters
        data = np.load(cache_filename, allow_pickle=True)
        cached_detail_level = data['detail_level']
        cached_min_x = data['min_x']
        cached_max_x = data['max_x']
        cached_min_y = data['min_y']
        cached_max_y = data['max_y']

        # Compute the current bounding box for all convex hulls, start, and end
        all_x_coords = np.concatenate([hull.points[:, 0] for hull in convex_hulls] + [start[:, 0], end[:, 0]])
        all_y_coords = np.concatenate([hull.points[:, 1] for hull in convex_hulls] + [start[:, 1], end[:, 1]])
        min_x, max_x = np.min(all_x_coords), np.max(all_x_coords)
        min_y, max_y = np.min(all_y_coords), np.max(all_y_coords)
        min_x -= 1 / detail_level
        max_x += 1 / detail_level
        min_y -= 1 / detail_level
        max_y += 1 / detail_level

        # Check if cached parameters match the current bounding box and detail level
        if (cached_detail_level == detail_level and
            np.isclose(int(cached_min_x), int(min_x)) and
            np.isclose(int(cached_max_x), int(max_x)) and
            np.isclose(int(cached_min_y),int( min_y)) and
            np.isclose(int(cached_max_y), int(max_y))):
            # Load the cached grid
            grid = data['grid']
            print("Loaded cached grid with matching detail level and bounding box")
        else:
            print("Cache found but detail level or bounding box does not match, regenerating grid")
            regenerate = True
    else:
        regenerate = True

    if regenerate:
        # Compute the bounding box for all convex hulls, start, and end
        all_x_coords = np.concatenate([hull.points[:, 0] for hull in convex_hulls] + [start[:, 0], end[:, 0]])
        all_y_coords = np.concatenate([hull.points[:, 1] for hull in convex_hulls] + [start[:, 1], end[:, 1]])
        min_x, max_x = np.min(all_x_coords), np.max(all_x_coords)
        min_y, max_y = np.min(all_y_coords), np.max(all_y_coords)
        min_x -= 1 / detail_level
        max_x += 1 / detail_level
        min_y -= 1 / detail_level
        max_y += 1 / detail_level

        # Create the grid
        grid_width = int((max_x - min_x) * detail_level)
        grid_height = int((max_y - min_y) * detail_level)
        grid = np.zeros((grid_height, grid_width), dtype=int)

        # Compute scaling factors
        scale_factor_x = grid_width / (max_x - min_x)
        scale_factor_y = grid_height / (max_y - min_y)

        # Pre-compute the Delaunay triangulation for each convex hull
        triangulations = [Delaunay(hull.points) for hull in convex_hulls]

        # Check each grid cell's vertices against the convex hulls
        for i in range(grid_height):
            for j in range(grid_width):
                # Calculate the exact coordinates of the cell's corners
                corners = [
                    [min_x + j / scale_factor_x, min_y + i / scale_factor_y],
                    [min_x + (j + 1) / scale_factor_x, min_y + i / scale_factor_y],
                    [min_x + (j + 1) / scale_factor_x, min_y + (i + 1) / scale_factor_y],
                    [min_x + j / scale_factor_x, min_y + (i + 1) / scale_factor_y]
                ]
                # Check if any corner is inside any convex hull
                if any(tri.find_simplex(corner) >= 0 for tri in triangulations for corner in corners):
                    grid[i, j] = 1

        # Save the grid, bounding box values, and detail level to a file
        np.savez_compressed(cache_filename, 
                            grid=grid, 
                            min_x=min_x, 
                            max_x=max_x, 
                            min_y=min_y, 
                            max_y=max_y,
                            detail_level=detail_level)
        print("Cached grid saved")

    # Compute scaling factors again for start and end points
    grid_width = grid.shape[1]
    grid_height = grid.shape[0]
    scale_factor_x = grid_width / (max_x - min_x)
    scale_factor_y = grid_height / (max_y - min_y)

    # Scale start and end points
    scaled_start = np.round((start - [min_x, min_y]) * [scale_factor_x, scale_factor_y]).astype(int)
    scaled_end = np.round((end - [min_x, min_y]) * [scale_factor_x, scale_factor_y]).astype(int)

    # Mark start and end points on the grid
    grid[scaled_start[0][1], scaled_start[0][0]] = 2  # Use a different value to mark start
    grid[scaled_end[0][1], scaled_end[0][0]] = 3      # Use a different value to mark end

    return grid, (min_x, max_x, min_y, max_y), scaled_start, scaled_end


def plot_grid_with_path(grid, min_x, max_x, min_y, max_y, scaled_start, scaled_end, path):
    """
    Plots the grid with the start and end points highlighted and a path by coloring tiles.
    
    Parameters:
    - grid: The grid with convex hulls (0: free, 1: obstacle).
    - min_x, max_x, min_y, max_y: The original coordinate ranges to preserve axes.
    - scaled_start: The scaled start point within the grid coordinates.
    - scaled_end: The scaled end point within the grid coordinates.
    - path: A list of tuples (x, y) representing the path from start to end.
    """
    # Create a copy of the grid to modify for path visualization
    display_grid = np.copy(grid)

    # Mark path in the display grid
    for x, y in path:
        display_grid[y, x] = 2  # Assign a unique value for the path

    # Create a color map: 0 = free (white), 1 = obstacle (black), 2 = path (blue)
    from matplotlib.colors import ListedColormap
    cmap = ListedColormap(['white', 'black', 'blue'])

    # Grid dimensions are derived from the shape of the grid array
    grid_height, grid_width = grid.shape

    # Plot the grid with customized colors
    plt.imshow(display_grid, cmap=cmap, origin='lower', extent=(min_x, max_x, min_y, max_y))

    # Highlight the scaled start and end points
    descaled_start_x = min_x + (scaled_start[0][0] * (max_x - min_x) / grid_width)
    descaled_start_y = min_y + (scaled_start[0][1] * (max_y - min_y) / grid_height)
    descaled_end_x = min_x + (scaled_end[0][0] * (max_x - min_x) / grid_width)
    descaled_end_y = min_y + (scaled_end[0][1] * (max_y - min_y) / grid_height)

    plt.scatter([descaled_start_x], [descaled_start_y], color='red', s=100, label='Start', zorder=5)
    plt.scatter([descaled_end_x], [descaled_end_y], color='green', s=100, label='End', zorder=5)

    plt.gca().set_aspect('equal', adjustable='box')
    plt.title("Grid with Start (Red), End (Green), and Path (Blue Tiles)")
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.legend()
    plt.show()

def descale_points(grid, scaled_points, min_x, max_x, min_y, max_y):
    """
    Converts a list of points from scaled grid coordinates back to their original scale, using grid dimensions.

    Parameters:
    - grid: The grid (2D array or list of lists) to which the points were scaled.
    - scaled_points: List of tuples representing the scaled points (x, y).
    - min_x, max_x, min_y, max_y: The minimum and maximum values of the original coordinate system.

    Returns:
    - List of tuples representing the descaled points in the original coordinate system.
    """
    # Extract the dimensions of the grid
    grid_height, grid_width = grid.shape

    descaled_points = []
    for x, y in scaled_points:
        # Calculate the original x and y based on scaling formulas
        original_x = min_x + (x * (max_x - min_x) / grid_width)
        original_y = min_y + (y * (max_y - min_y) / grid_height)
        descaled_points.append((original_x, original_y))
    
    return descaled_points

def plot_grid_with_start_end(grid, min_x, max_x, min_y, max_y, scaled_start, scaled_end):
    """
    Plots the grid, with the start and end points highlighted after descaling them.

    Parameters:
    - grid: The grid with convex hulls.
    - min_x, max_x, min_y, max_y: The original coordinate ranges to preserve axes.
    - scaled_start: The scaled start point within the grid coordinates.
    - scaled_end: The scaled end point within the grid coordinates.
    """
    # Grid dimensions
    grid_height, grid_width = grid.shape
    
    # Descale the points
    descaled_start_x = min_x + (scaled_start[0][0] * (max_x - min_x) / grid_width)
    descaled_start_y = min_y + (scaled_start[0][1] * (max_y - min_y) / grid_height)
    descaled_end_x = min_x + (scaled_end[0][0] * (max_x - min_x) / grid_width)
    descaled_end_y = min_y + (scaled_end[0][1] * (max_y - min_y) / grid_height)

    plt.imshow(grid, cmap='gray', origin='lower', extent=(min_x, max_x, min_y, max_y))

    # Highlight the descaled start and end points
    plt.scatter([descaled_start_x], [descaled_start_y], color='red', label='Start', zorder=5)
    plt.scatter([descaled_end_x], [descaled_end_y], color='green', label='End', zorder=5)

    plt.gca().set_aspect('equal', adjustable='box')
    plt.title("Grid with Start (Red) and End (Green)")
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.legend()
    plt.show()



if __name__ == "__main__":
    # Example usage:
    start = np.array([[20.3, 30.7]])  # Start point
    end = np.array([[80.5, 20.9]])    # End point
    convex_hulls = [
        ConvexHull(np.array([[10.1, 10.4], [10.5, 20.7], [20.9, 15.2]])),
        ConvexHull(np.array([[50.2, 50.3], [55.6, 55.8], [60.7, 50.1], [55.9, 45.5]])),
    ]

    # Create the grid with a specified detail level
    detail_level = 6  # Adjust this to change the resolution of the grid
    grid, (min_x, max_x, min_y, max_y), scaled_start, scaled_end = create_grid_with_start_end(detail_level, start, end, convex_hulls)

    # Plot the grid with start and end points highlighted
    plot_grid_with_start_end(grid, min_x, max_x, min_y, max_y, scaled_start, scaled_end)

    path = a_star_pathfinding(grid, tuple(scaled_start[0]), tuple(scaled_end[0]))
    #print(descale_points(grid, path, min_x, max_x, min_y, max_y))

    plot_grid_with_path(grid, min_x, max_x, min_y, max_y, scaled_start, scaled_end,path)