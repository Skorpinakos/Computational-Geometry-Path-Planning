import open3d as o3d
import numpy as np
import time
np.set_printoptions(suppress=True)
import bisect
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.spatial.transform import Rotation as R

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

    


def load_mesh_and_calculate_centers(mesh):
    mesh.compute_vertex_normals()



    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)
    #print(triangles)

    triangle_centers = np.mean(vertices[triangles], axis=1)
    print("triangles count:",len(triangle_centers))

    return triangle_centers, triangles#np.vstack((triangle_centers,vertices)), triangles#[0:3500]

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

    axes=switch_case(axis)
    

    for ax in axes:
        column_box_size=0.8
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

def color_mesh_by_triangles(file_path, wall_classify_input):
    # Load the mesh from the given file path
    mesh = o3d.io.read_triangle_mesh(file_path)
    
    # Ensure the mesh has triangles
    if len(mesh.triangles) == 0:
        print("No triangles found in the mesh.")
        return
    wall_classify=wall_classify_input[0:len(mesh.triangles),:]
    # Check if the classification array matches the number of triangles
    if len(mesh.triangles) != len(wall_classify):
        print("The classification array does not match the number of triangles in the mesh.")
        return
    
    # Convert wall_classify to a numpy array and normalize the colors
    triangle_colors = np.array(wall_classify)
    if triangle_colors.max() > 1:
        triangle_colors = triangle_colors / 255.0
    
    # Prepare colors for each vertex by averaging triangle colors
    vertex_colors = np.zeros((len(mesh.vertices), 3))
    count_colors = np.zeros(len(mesh.vertices))
    for triangle, color in zip(mesh.triangles, triangle_colors):
        for vertex in triangle:
            vertex = int(vertex)  # Ensure vertex indices are integers
            vertex_colors[vertex] += color
            count_colors[vertex] += 1
    vertex_colors /= count_colors[:, np.newaxis]  # Normalize by count to get average color

    # Assign colors to the vertices
    mesh.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)

    # Compute normals for better visualization
    if not mesh.has_vertex_normals():
        mesh.compute_vertex_normals()

    # Prepare to visualize the edges of triangles
    lines = []
    for triangle in mesh.triangles:
        lines.append([int(triangle[0]), int(triangle[1])])
        lines.append([int(triangle[1]), int(triangle[2])])
        lines.append([int(triangle[2]), int(triangle[0])])

    # Create a line set for edges
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(np.asarray(mesh.vertices)),
        lines=o3d.utility.Vector2iVector(lines)
    )
    line_set.colors = o3d.utility.Vector3dVector(np.tile(np.array([0, 0, 0]), (len(lines), 1)))  # Color lines black
    
    # Add a coordinate frame for reference
    bounding_box = mesh.get_axis_aligned_bounding_box()
    scale = bounding_box.get_extent().max() * 0.1  # 10% of the largest dimension
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=scale, origin=bounding_box.get_center() - bounding_box.get_extent() * 0.5)

    # Display the mesh with edges and coordinate frame
    o3d.visualization.draw_geometries([mesh, line_set, coord_frame], mesh_show_back_face=True)

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

def plot_component_size_distribution(mesh):
    # Initialize Union-Find data structure
    class UnionFind:
        def __init__(self, size):
            self.parent = list(range(size))
            self.rank = [0] * size

        def find(self, p):
            if self.parent[p] != p:
                self.parent[p] = self.find(self.parent[p])  # Path compression
            return self.parent[p]

        def union(self, p, q):
            rootP = self.find(p)
            rootQ = self.find(q)
            if rootP != rootQ:
                # Union by rank
                if self.rank[rootP] > self.rank[rootQ]:
                    self.parent[rootQ] = rootP
                elif self.rank[rootP] < self.rank[rootQ]:
                    self.parent[rootP] = rootQ
                else:
                    self.parent[rootQ] = rootP
                    self.rank[rootP] += 1

    num_triangles = len(mesh.triangles)
    uf = UnionFind(num_triangles)

    # Create a map from edges to triangles for quick lookup
    edge_map = {}
    for i, triangle in enumerate(mesh.triangles):
        for j in range(3):
            edge = tuple(sorted((triangle[j], triangle[(j + 1) % 3])))
            if edge in edge_map:
                uf.union(i, edge_map[edge])
            else:
                edge_map[edge] = i

    # Determine component sizes
    component_size = {}
    for i in range(num_triangles):
        root = uf.find(i)
        if root in component_size:
            component_size[root] += 1
        else:
            component_size[root] = 1

    # Extract sizes
    sizes = list(component_size.values())

    # Plot the distribution of component sizes
    plt.figure(figsize=(10, 5))
    plt.hist(sizes, bins=max(sizes), color='blue', alpha=0.7)
    plt.title('Distribution of Component Sizes')
    plt.xlabel('Size of Components')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()


def remove_small_clumps(mesh,thres):
    
    if not mesh.has_triangle_normals():
        mesh.compute_triangle_normals()
    
    # Initialize Union-Find data structure
    class UnionFind:
        def __init__(self, size):
            self.parent = list(range(size))
            self.rank = [0] * size

        def find(self, p):
            if self.parent[p] != p:
                self.parent[p] = self.find(self.parent[p])  # Path compression
            return self.parent[p]

        def union(self, p, q):
            rootP = self.find(p)
            rootQ = self.find(q)
            if rootP != rootQ:
                # Union by rank
                if self.rank[rootP] > self.rank[rootQ]:
                    self.parent[rootQ] = rootP
                elif self.rank[rootP] < self.rank[rootQ]:
                    self.parent[rootP] = rootQ
                else:
                    self.parent[rootQ] = rootP
                    self.rank[rootP] += 1

    num_triangles = len(mesh.triangles)
    uf = UnionFind(num_triangles)

    # Create a map from vertices to triangles for quick lookup
    vertex_map = {}
    for i, triangle in enumerate(mesh.triangles):
        for vertex in triangle:
            if vertex in vertex_map:
                # Union current triangle with all triangles that share this vertex
                for other_triangle_index in vertex_map[vertex]:
                    uf.union(i, other_triangle_index)
                vertex_map[vertex].append(i)
            else:
                vertex_map[vertex] = [i]

    # Determine component sizes and collect triangles of large components
    component_size = {}
    large_component_triangles = []
    triangle_color_map = {}
    for i in range(num_triangles):
        root = uf.find(i)
        if root in component_size:
            component_size[root] += 1
        else:
            component_size[root] = 1
    for i in range(num_triangles):
        root = uf.find(i)
        color = [1, 0, 0] if component_size[root] > thres else [0, 1, 0]  # Red or Green
        triangle_color_map[i] = color
        if component_size[root] > thres:
            large_component_triangles.append(mesh.triangles[i])

    # Apply colors to vertices based on triangle colors
    vertex_colors = np.zeros((len(mesh.vertices), 3))
    for i, triangle in enumerate(mesh.triangles):
        color = triangle_color_map[i]
        for vertex in triangle:
            vertex_colors[vertex] = color

    # Update mesh colors
    mesh.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)

    # Visualize the mesh
    o3d.visualization.draw_geometries([mesh], mesh_show_back_face=True)

    # Rebuild mesh from large components
    if large_component_triangles:
        new_mesh = o3d.geometry.TriangleMesh()
        new_mesh.vertices = mesh.vertices
        new_mesh.triangles = o3d.utility.Vector3iVector(large_component_triangles)
        new_mesh.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)  # apply vertex colors
        new_mesh.compute_vertex_normals()
    else:
        new_mesh = None

    return new_mesh



#welding
def merge_close_vertices(mesh, tolerance=1e-5):

    
    # Get vertices and triangles
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)
    
    # Use a KDTree to find close vertices
    kdtree = o3d.geometry.KDTreeFlann(mesh)
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
    
    # Create a new mesh
    new_mesh = o3d.geometry.TriangleMesh()
    new_mesh.vertices = o3d.utility.Vector3dVector(merged_vertices)
    new_mesh.triangles = o3d.utility.Vector3iVector(new_triangles)
    
    # Recompute normals if needed
    if mesh.has_vertex_normals():
        new_mesh.compute_vertex_normals()
    
    return new_mesh


def plot_normals_histogram(mesh):
    # Calculate triangle normals
    triangles = np.asarray(mesh.triangles)
    vertices = np.asarray(mesh.vertices)
    triangle_normals = np.cross(vertices[triangles[:, 1]] - vertices[triangles[:, 0]],
                                vertices[triangles[:, 2]] - vertices[triangles[:, 0]])
    norms = np.linalg.norm(triangle_normals, axis=1)
    
    # Filter out zero normals to avoid division by zero
    nonzero_normals = norms > 0
    triangle_normals = triangle_normals[nonzero_normals]
    norms = norms[nonzero_normals]
    
    # Normalize the normals
    triangle_normals = triangle_normals / norms[:, np.newaxis]

    # Convert normals to azimuth and elevation angles
    azimuth = np.arctan2(triangle_normals[:, 1], triangle_normals[:, 0])
    elevation = np.arcsin(triangle_normals[:, 2])

    # Plotting
    plt.figure(figsize=(10, 5))
    plt.hist2d(np.degrees(azimuth), np.degrees(elevation), bins=50, cmap='viridis')
    plt.colorbar(label='Frequency')
    plt.xlabel('Azimuth (Degrees)')
    plt.ylabel('Elevation (Degrees)')
    plt.title('Histogram of Normal Vectors in Azimuth-Elevation Space')
    plt.show()

def align_mesh_by_normals_clustering(mesh, n_clusters=6):
    # Extract triangle normals
    triangle_normals = np.asarray(mesh.triangle_normals)

    # Compute norms and create a mask for valid normals (non-zero length)
    norms = np.linalg.norm(triangle_normals, axis=1, keepdims=True)
    valid_normals_mask = norms > 0  # This creates a (n, 1) boolean array

    # Safely normalize triangle normals
    valid_normals = triangle_normals[valid_normals_mask[:, 0], :]  # Use the mask to filter rows
    valid_normals /= norms[valid_normals_mask[:, 0], :]  # Normalize the valid normals

    # Replace the original normals with the normalized ones
    triangle_normals[valid_normals_mask[:, 0], :] = valid_normals

    # Ensure no NaN values remain
    if np.isnan(triangle_normals).any():
        print("Warning: NaNs found in the normalized normals, likely due to zero-length normals.")
        return None

    # Proceed only if there are enough normals for clustering
    if len(triangle_normals) < n_clusters:
        print(f"Not enough valid normals for clustering. Required: {n_clusters}, Available: {len(triangle_normals)}")
        return None

    # Cluster normals using K-means
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(triangle_normals)
    labels = kmeans.labels_
    cluster_centers = kmeans.cluster_centers_

    # Aligning cluster centers to coordinate axes
    target_axes = np.eye(3)
    rotation_matrix = R.align_vectors(target_axes, cluster_centers[:3])[0].as_matrix()

    # Rotate vertices
    vertices = np.asarray(mesh.vertices)
    centroid = np.mean(vertices, axis=0)
    centered_vertices = vertices - centroid
    aligned_vertices = centered_vertices @ rotation_matrix

    # Update the mesh with aligned vertices
    aligned_mesh = o3d.geometry.TriangleMesh()
    aligned_mesh.vertices = o3d.utility.Vector3dVector(aligned_vertices + centroid)
    aligned_mesh.triangles = mesh.triangles
    aligned_mesh.vertex_colors = mesh.vertex_colors
    aligned_mesh.vertex_normals = mesh.vertex_normals

    return aligned_mesh

def check_and_load_mesh(filepath):
    print("Loading mesh from:", filepath)
    mesh = o3d.io.read_triangle_mesh(filepath)
    print("Mesh loaded.")

    # Check if the mesh has vertices and triangles
    if len(mesh.vertices) == 0 or len(mesh.triangles) == 0:
        print("Error: The mesh has no vertices or triangles.")
        return None

    # Check if the mesh has normals
    if len(mesh.triangle_normals) == 0:
        print("No triangle normals present. Computing normals.")
        mesh.compute_triangle_normals()
        if len(mesh.triangle_normals) == 0:
            print("Failed to compute triangle normals.")
            return None
        else:
            print("Triangle normals computed.")

    return mesh

def azimuth_angle(v):
    return np.arctan2(v[1], v[0])

def color_mesh_by_azimuth(mesh, n_clusters=4, radius_degrees=10):
    # Convert radius from degrees to radians
    radius_radians = np.radians(radius_degrees)

    # Extract triangle normals and convert to azimuth angles
    triangle_normals = np.asarray(mesh.triangle_normals)
    azimuths = np.array([azimuth_angle(normal) for normal in triangle_normals])

    # Handle circular clustering via cosine and sine transformation
    cos_azimuths = np.cos(azimuths)
    sin_azimuths = np.sin(azimuths)
    azimuth_vectors = np.stack((cos_azimuths, sin_azimuths), axis=-1)

    # Cluster the azimuth vectors
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(azimuth_vectors)
    cluster_centers = kmeans.cluster_centers_

    # Map clusters to colors
    colors = np.array([
        [1, 0, 0], [0, 1, 0], [0, 0, 1],
        [1, 1, 0], 
    ])

    # Assign colors based on closest cluster center within radius
    triangle_colors = np.zeros((len(triangle_normals), 3))
    for i, az_vector in enumerate(azimuth_vectors):
        distances = np.linalg.norm(cluster_centers - az_vector, axis=1)
        closest_cluster = np.argmin(distances)

        if distances[closest_cluster] < radius_radians:
            triangle_colors[i] = colors[closest_cluster]
        else:
            triangle_colors[i] = [0.5, 0.5, 0.5]  # Default gray for out-of-radius normals

    # Average the colors at each vertex
    vertex_colors = np.zeros((len(mesh.vertices), 3))
    triangle_to_vertices = np.asarray(mesh.triangles)
    for i, triangle in enumerate(triangle_to_vertices):
        for vertex in triangle:
            vertex_colors[vertex] += triangle_colors[i]
    vertex_colors /= np.bincount(triangle_to_vertices.flatten(), minlength=len(mesh.vertices))[:, np.newaxis]

    # Update mesh colors
    mesh.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)
    
    # Visualize the mesh
    o3d.visualization.draw_geometries([mesh])



#exploration

#file_path = '25k_random_rot/amp.obj' 
file_path = '25k_door/amp.obj' 
#file_path = 'results/closed_holes_mesh.obj' 
mesh = o3d.io.read_triangle_mesh(file_path)
#mesh = check_and_load_mesh(file_path)
#plot_normals_histogram(mesh)

#color_mesh_by_azimuth(mesh,radius_degrees=15)
#aligned_mesh=align_mesh_by_normals_clustering(mesh)
#plot_normals_histogram(aligned_mesh)

#start=time.time()
t1=time.time()
triangle_centers,triangle_array = load_mesh_and_calculate_centers(mesh)
sorted_by_x, sorted_by_y, sorted_by_z = sort_centers_by_coordinates(triangle_centers)

x_tree=SortedArrayBST(sorted_by_x,axis_type=0,og_array=triangle_centers)
y_tree=SortedArrayBST(sorted_by_y,axis_type=1,og_array=triangle_centers)
z_tree=SortedArrayBST(sorted_by_z,axis_type=2,og_array=triangle_centers)

trees=[x_tree,y_tree,z_tree]
wall_classify=np.zeros((len(triangle_centers),3))
box_size=0.8
threshold=np.zeros((3,2))+0.075+[(0,0),(0,0),(-0.05,0)] #the -0.05 is to tackle the issue of objects touching the ground loosing some triangles


for i,triangle in enumerate(triangle_centers):
    for d,dimension in enumerate(triangle):
        triangles,indices=trees[d].get_range_array(dimension-box_size,dimension+box_size)
        if check_for_wall(triangle,triangles,d,threshold):
            wall_classify[i][d]=1


print(len(wall_classify))
print(wall_classify)





t2=time.time()
print(t2-t1)


# Check if each row is all zeros
zero_rows = np.all(wall_classify == 0, axis=1)

# Calculate the percentage of zero-valued points
percentage_zeros = np.mean(zero_rows) * 100

print(f"Percentage of points that are [0, 0, 0]: {percentage_zeros}%")

#color_mesh_by_triangles(file_path,wall_classify)
color_mesh_by_triangles(file_path,wall_classify)
walls_removed_mesh=remove_non_black_triangles_and_set_grey(file_path,wall_classify)


#cleanup
plot_component_size_distribution(walls_removed_mesh)
walls_removed_mesh_welded=merge_close_vertices(walls_removed_mesh,0.1)
plot_component_size_distribution(walls_removed_mesh_welded)
inside_objects=remove_small_clumps(walls_removed_mesh_welded,10)