import numpy as np
import open3d as o3d

def region_growing(pcd, threshold_distance=0.5, max_points=5000, min_points=100):
    # Define UnionFind class within the function
    class UnionFind:
        def __init__(self, size):
            self.parent = list(range(size))
            self.rank = [1] * size

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

        def get_components(self):
            # Return all connected components
            components = {}
            for i in range(len(self.parent)):
                root = self.find(i)
                if root not in components:
                    components[root] = []
                components[root].append(i)
            return components.values()

    points = np.asarray(pcd.points)
    num_points = len(points)
    
    # Initialize Union-Find for all points
    uf = UnionFind(num_points)

    # Using a KDTree for efficient nearest neighbor search
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)

    # For each point, find its neighbors and unite them in the Union-Find structure
    for i in range(num_points):
        current_point = points[i]
        
        # Query nearby points using the KDTree
        k, idx, _ = pcd_tree.search_radius_vector_3d(current_point, threshold_distance)
        
        # Union the point with its neighbors
        for j in range(k):
            neighbor_index = idx[j]
            if neighbor_index != i:  # Avoid self-union
                uf.union(i, neighbor_index)
    
    # Extract connected components (regions)
    region_arrays = []
    components = uf.get_components()

    # Only keep regions that meet the minimum point threshold
    for component in components:
        if len(component) >= min_points:
            region_points = points[list(component)]
            region_arrays.append(region_points)

    return region_arrays
def visualize_regions(region_arrays):
    # Create an Open3D visualization window
    vis = o3d.visualization.Visualizer()
    vis.create_window()

    # Convert each region from numpy array to Open3D PointCloud and add to the visualizer
    for region in region_arrays:
        region_pcd = o3d.geometry.PointCloud()
        region_pcd.points = o3d.utility.Vector3dVector(region)
        region_pcd.paint_uniform_color(np.random.rand(3))  # Assign a random color to each region
        vis.add_geometry(region_pcd)

    # Run the visualizer
    vis.run()
    vis.destroy_window()


# Load the point cloud using Open3D
pcd = o3d.io.read_point_cloud("results/filtered_156k.pcd")

# Process the point cloud to extract regions
region_lists = region_growing(pcd, threshold_distance=0.5, max_points=5000, min_points=4)
for i in region_lists:
    print(len(i))

# Visualize the regions
visualize_regions(region_lists)
