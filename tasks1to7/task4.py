import open3d as o3d
import numpy as np
import time
np.set_printoptions(suppress=True)
import bisect

from multiprocessing import Pool

class TreeNode:
    def __init__(self, val, index):
        self.val = val
        self.index = index
        self.left = None
        self.right = None

class SortedArrayBST:
    def __init__(self, sorted_array_input,axis_type,og_array):
        self.og_array=og_array
        self.og_correspondance=sorted_array_input[:,-1]
        #print(self.og_correspondance)
        self.sorted_array_input=sorted_array_input
        self.sorted_array = sorted_array_input[:,axis_type].copy()
        #print(self.sorted_array.shape)
        self.root = self.build_bst(0, len(self.sorted_array) - 1)
        
    def build_bst(self, start, end):
        if start > end:
            return None
        mid = (start + end) // 2
        node = TreeNode(self.sorted_array[mid], mid)
        node.left = self.build_bst(start, mid - 1)
        node.right = self.build_bst(mid + 1, end)
        return node

    def find_range_indices(self, range_start, range_end):
        idx_left = bisect.bisect_left(self.sorted_array, range_start)
        idx_right = bisect.bisect_right(self.sorted_array, range_end) - 1
        # Ensure the indices are within the array bounds and correct
        if idx_left < len(self.sorted_array) and self.sorted_array[idx_left] < range_start:
            idx_left += 1
        if idx_right >= 0 and self.sorted_array[idx_right] > range_end:
            idx_right -= 1
        return (idx_left, idx_right)
    
    def get_range_array(self,range_start,range_end):
        range=self.find_range_indices(range_start,range_end)
        og_array_correspondance=self.og_correspondance[range[0]:range[1]].astype(int)

        return self.og_array[og_array_correspondance],og_array_correspondance

    




def sort_centers_by_coordinates(triangle_centers):
    # Calculate indices for sorting by x, y, and z dimensions
    indices_x_sorted = np.argsort(triangle_centers[:, 0])
    indices_y_sorted = np.argsort(triangle_centers[:, 1])
    indices_z_sorted = np.argsort(triangle_centers[:, 2])

    # Create sorted arrays including original indices
    sorted_by_x = np.hstack((triangle_centers[indices_x_sorted], indices_x_sorted[:, np.newaxis]))
    sorted_by_y = np.hstack((triangle_centers[indices_y_sorted], indices_y_sorted[:, np.newaxis]))
    sorted_by_z = np.hstack((triangle_centers[indices_z_sorted], indices_z_sorted[:, np.newaxis]))

    return sorted_by_x, sorted_by_y, sorted_by_z


def switch_case(x):
    return {
        0: (1, 2),
        1: (0, 2),
        2: (0, 1)
    }.get(x, "Invalid input")

def check_for_wall(in_question_triangle_center,slice_triangles,axis,threshold):
    column_box_size=0.15

    axes=switch_case(axis)
    

    for ax in axes:
        
        while True:
            try:

                
                orthogonal_to_column_threshold_axis=3-ax-axis #this gives the remaining axis

                center_of_column=in_question_triangle_center[orthogonal_to_column_threshold_axis]

                triangles_of_column=slice_triangles[(slice_triangles[:, orthogonal_to_column_threshold_axis] >= center_of_column-column_box_size) & (slice_triangles[:, orthogonal_to_column_threshold_axis] <= center_of_column+column_box_size)]

                min_index = np.argmin(triangles_of_column[:, ax])
                min_point = triangles_of_column[min_index]
                min_value=min_point[ax]

                max_index = np.argmax(triangles_of_column[:, ax])
                max_point = triangles_of_column[max_index]
                max_value=max_point[ax]

                range=max(max_value-min_value,0.000001)


                triangle_value=in_question_triangle_center[ax]

                if (abs(triangle_value-min_value)/range)<threshold[ax][0] or (abs(triangle_value-max_value)/range)<threshold[ax][1]:
                    return True
                break
            except:
                #print("oof")
                column_box_size=column_box_size*2
                if column_box_size>=36:
                    break
        
    return False



def remove_non_black_triangles_and_set_grey(file_path, wall_classify_input):
    # Load the mesh from the given file path
    mesh = o3d.io.read_triangle_mesh(file_path)
    
    # Ensure the mesh has triangles
    if len(mesh.triangles) == 0:
        print("No triangles found in the mesh.")
        return

    wall_classify = np.array(wall_classify_input[0:len(mesh.triangles), :])
    
    # Check if the classification array matches the number of triangles
    if len(mesh.triangles) != len(wall_classify):
        print("The classification array does not match the number of triangles in the mesh.")
        return
    
    # Identify only black triangles (where the color is [0, 0, 0])
    black_color = np.array([0, 0, 0])
    black_triangles_mask = np.all(wall_classify == black_color, axis=1)
    
    # Keep only black triangles
    black_triangles = np.asarray(mesh.triangles)[black_triangles_mask]
    
    # Update the mesh with only the black triangles
    mesh.triangles = o3d.utility.Vector3iVector(black_triangles)
    
    # Assign all vertex colors to grey [0.5, 0.5, 0.5]
    grey_color = np.array([0.2, 0.8, 0.1])
    vertex_colors = np.tile(grey_color, (len(mesh.vertices), 1))
    
    # Assign grey color to all vertices
    mesh.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)
    
    # Compute normals for better visualization
    if not mesh.has_vertex_normals():
        mesh.compute_vertex_normals()
    
    # Display the mesh with grey-colored triangles
    o3d.visualization.draw_geometries([mesh], mesh_show_back_face=True)
    return mesh

def rotate_point_cloud(point_cloud_np, rotation_angles):
    """
    Rotates a point cloud by given angles around each of the X, Y, and Z axes.

    Args:
    point_cloud_np (numpy.ndarray): The input point cloud as a N x 3 NumPy array.
    rotation_angles (tuple or list): The rotation angles in degrees for each axis (x, y, z).

    Returns:
    numpy.ndarray: The rotated point cloud as a N x 3 NumPy array.
    """
    # Convert the rotation angles from degrees to radians
    rotation_radians = np.radians(rotation_angles)

    # Create rotation matrices for each axis
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(rotation_radians[0]), -np.sin(rotation_radians[0])],
                   [0, np.sin(rotation_radians[0]), np.cos(rotation_radians[0])]])

    Ry = np.array([[np.cos(rotation_radians[1]), 0, np.sin(rotation_radians[1])],
                   [0, 1, 0],
                   [-np.sin(rotation_radians[1]), 0, np.cos(rotation_radians[1])]])

    Rz = np.array([[np.cos(rotation_radians[2]), -np.sin(rotation_radians[2]), 0],
                   [np.sin(rotation_radians[2]), np.cos(rotation_radians[2]), 0],
                   [0, 0, 1]])

    # Combined rotation matrix
    R = Rx @ Ry @ Rz

    # Apply the rotation to the point cloud
    rotated_point_cloud = point_cloud_np @ R.T

    return rotated_point_cloud

def color_point_cloud(points, classification):
    print(len(classification))
    print(len(points))
    # Convert numpy array to Open3D point cloud object
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)

    # Check if the classification array matches the number of points
    if len(points) != len(classification):
        print("The classification array does not match the number of points in the point cloud.")
        return

    # Initialize an array for colors, default all to red
    colors = np.full((len(points), 3), [1, 0, 0])  # Start with all points as red

    # Update colors where classification is 1 to green
    colors[classification == 1] = [0, 1, 0]  # Set points classified as '1' to green

    # Assign colors to the point cloud
    point_cloud.colors = o3d.utility.Vector3dVector(colors)

    # Create axes as cylinders for better visibility
    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.5, origin=[0, 0, 0])

    # Display the point cloud with cylinder axes
    o3d.visualization.draw_geometries([point_cloud, mesh_frame], "Point Cloud Visualization with Cylinder Axes")

    return point_cloud

def save_and_draw_filtered_points_to_pcd(points, classification, filename):
    """
    Filters points based on classification, saves to a PCD file, and displays the filtered point cloud using Open3D.

    Parameters:
        points (np.ndarray): The Nx3 array containing point cloud data.
        classification (np.ndarray): The N-length array where points corresponding
                                     to a classification of 0 will be removed.
        filename (str): The filename where the PCD file will be saved.
    """
    # Filter points where the classification is not 0
    filtered_points = points[classification != 0]

    # Create an Open3D point cloud object
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(filtered_points)

    # Save the point cloud to a PCD file
    o3d.io.write_point_cloud(filename, pcd, write_ascii=True)

    # Draw the point cloud
    o3d.visualization.draw_geometries([pcd], window_name="Filtered Point Cloud", width=800, height=600)



def worker(data_slice, box_size, threshold,points):
    triangle_centers = points

    sorted_by_x, sorted_by_y, sorted_by_z = sort_centers_by_coordinates(triangle_centers)
    x_tree = SortedArrayBST(sorted_by_x, axis_type=0, og_array=triangle_centers)
    y_tree = SortedArrayBST(sorted_by_y, axis_type=1, og_array=triangle_centers)
    z_tree = SortedArrayBST(sorted_by_z, axis_type=2, og_array=triangle_centers)
   
    trees = [x_tree, y_tree, z_tree]
    
    result = np.zeros((len(data_slice), 3))
    for i, triangle in enumerate(data_slice):
        for d, dimension in enumerate(triangle):
            triangles, indices = trees[d].get_range_array(dimension - box_size, dimension + box_size)
            if check_for_wall(triangle, triangles, d, threshold):
                result[i][d] = 1
    return result

def divide_chunks(data, num_chunks):
    """Yield successive n-sized chunks from data."""
    chunk_size = len(data) // num_chunks
    remainder = len(data) % num_chunks
    start = 0
    for i in range(num_chunks):
        end = start + chunk_size + (1 if i < remainder else 0)
        yield data[start:end]
        start = end

def main():
    #file_path = 'results/pcd_156k.ply'
    #file_path = 'results/pcd_aligned.ply'
    file_path = 'results/closed_holes_pcd.ply'
    t1 = time.time()
    pcd = o3d.io.read_point_cloud(file_path)
    points = np.asarray(pcd.points)
    #points=rotate_point_cloud(points,(-15,9,6)) #ROTATION DEMO

    box_size = 0.15
    threshold = np.zeros((3, 2)) + 0.1 + [(0, 0), (0, 0), (-0.02, -0.01)]

    num_processes = 12
    pool = Pool(processes=num_processes)
    data_chunks = list(divide_chunks(points, num_processes))

    results = pool.starmap(worker, [(chunk, box_size, threshold,points) for chunk in data_chunks])
    wall_classify = np.vstack(results)

    t2 = time.time()
    print(f"Processing time: {t2 - t1}s")

    zero_rows = np.all(wall_classify == 0, axis=1)
    percentage_zeros = np.mean(zero_rows) * 100
    print(f"Percentage of points that are [0, 0, 0]: {percentage_zeros}%")
    save_and_draw_filtered_points_to_pcd(points,zero_rows,"results/filtered_156k.pcd")
    color_point_cloud(points, zero_rows)

if __name__ == '__main__':
    main()