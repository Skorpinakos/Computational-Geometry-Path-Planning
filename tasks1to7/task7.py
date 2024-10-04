from vvrpywork.constants import Key, Mouse, Color
from vvrpywork.scene import Scene3D, get_rotation_matrix, world_space
from vvrpywork.shapes import (
    Point3D, Line3D, Arrow3D, Sphere3D, Cuboid3D, Cuboid3DGeneralized,
    PointSet3D, LineSet3D, Mesh3D
)
import heapq
from task6 import find_holes,find_largest_hole,visualize_hole
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from matplotlib import colormaps as cm
import numpy as np
import open3d as o3d
from scipy import sparse
import time
from task9 import get_wall_convex_hulls
from task7_path import plot_grid_with_start_end,create_grid_with_start_end,a_star_pathfinding,plot_grid_with_path,descale_points
            
from vvrpywork.constants import Key, Mouse, Color
from vvrpywork.scene import Scene2D
from vvrpywork.shapes import (
    Point2D, Line2D, Triangle2D, Circle2D, Rectangle2D,
    PointSet2D, LineSet2D, Polygon2D
)

WIDTH = 1000
HEIGHT = 800



def find_closest_vertex(mesh: Mesh3D, query: tuple) -> int:

    dist_sq = np.sum((mesh.vertices - query) ** 2, -1)

    return np.argmin(dist_sq)

class Task7_scene(Scene3D):
    def __init__(self):
        super().__init__(WIDTH, HEIGHT,  "Task7_scene", output=True)
        self.reset_mesh()
        self.printHelp()
        self.start=None
        self.target=None

    def reset_mesh(self):

        # Choose mesh
        self.mesh_aligned_with_holes=Mesh3D("results/mesh_aligned.obj")
        self.mesh_aligned_with_holes.remove_duplicated_vertices()
        self.mesh_aligned_with_holes.remove_unreferenced_vertices()

        self.mesh = Mesh3D("results/mesh_aligned.obj")
        self.mesh.remove_duplicated_vertices()
        self.mesh.remove_unreferenced_vertices()
        #self.mesh = o3d.io.read_triangle_mesh("25k_door/amp.obj")
        self.mesh=remove_dominant_plane_from_mesh3d(self.mesh,distance_threshold=0.15)
        self.mesh=merge_close_vertices_in_mesh3d(self.mesh)
        self.mesh=remove_small_clumps_from_mesh3d(self.mesh,10)
        self.mesh.color=Color.GREY


        self.filtered_pcd=o3d.io.read_point_cloud("results/filtered_156k.pcd")
        self.room_objects_pcds=region_growing(self.filtered_pcd)
        self.objects_drawn=False
        


        self.removeShape("mesh")
        self.addShape(self.mesh, "mesh")

        self.wireframe = LineSet3D.create_from_mesh(self.mesh)
        self.removeShape("wireframe")
        self.addShape(self.wireframe, "wireframe")
        self.show_wireframe = True

        self.hole_index=-1



    def reset_sliders(self):
        self.set_slider_value(0, 0)
        self.set_slider_value(1, 0.5)

        
    @world_space
    def on_mouse_press(self, x, y, z, button, modifiers):
        if button == Mouse.MOUSELEFT and modifiers & Key.MOD_SHIFT:
            if np.isinf(z):
                return
            
            self.selected_vertex = find_closest_vertex(self.mesh, (x, y, z))
            print("Selected Vertex is at: "+str(self.mesh.vertices[self.selected_vertex]))

            vc = self.mesh.vertex_colors
            vc=vc*0
            vc=vc+0.7
            vc[self.selected_vertex] = (1, 0, 0)
            self.mesh.vertex_colors = vc
            self.updateShape("mesh", True)
            self.start=self.mesh.vertices[self.selected_vertex]

    def on_key_press(self, symbol, modifiers):

        if symbol == Key.R:
            self.reset_mesh()

        if symbol == Key.T:
            self.hole_index+=1

            
            
            holes=find_holes(self.mesh_aligned_with_holes)

            large_hole=find_largest_hole(holes,self.hole_index%len(holes))
            door=self.mesh_aligned_with_holes.vertices[large_hole]
            #visualize_hole(self.mesh_aligned_with_holes,large_hole)
            #print(door)

            door_frame=PointSet3D(door,color=Color.RED,size=2)
            self.target=np.mean(door*0.95, axis=0) #the 0.95 is to make the target inside the room
            targ_point=Point3D(self.target,color=Color.GREEN,size=8)

            try:
                self.removeShape("door_frame")
                self.removeShape("targ")
            except:
                pass
            self.addShape(door_frame,"door_frame")
            self.updateShape("door_frame")
            self.addShape(targ_point,"targ")
            self.updateShape("targ")         
            
            print("Selected door is at: "+str(self.target))

        if symbol == Key.C:
            

            if self.objects_drawn == False:
                for i, object in enumerate(self.room_objects_pcds):
                    try:
                        self.removeShape("obj_" + str(i))
                    except:
                        pass
                    
                    self.addShape(PointSet3D(object, color=int_to_distinct_color(i), size=3), "obj_" + str(i))
                    self.updateShape("obj_" + str(i))

                self.objects_drawn = True
            else:
                for i, object in enumerate(self.room_objects_pcds):
                    try:
                        self.removeShape("obj_" + str(i))
                    except:
                        pass
                self.objects_drawn=False
                #print(self.objects_drawn)

        if symbol == Key.P:
            t1=time.time()
            path_height=1
            if type(self.start)==type(None) or type(self.target)==type(None):
                print("Select start and target first please.")
                return
            start=tuple(self.start[0:2])
            end=tuple(self.target[0:2])
            ground_objects=self.room_objects_pcds#filter_roof_objects(self.room_objects_pcds,self.mesh_aligned_with_holes)
            #print(ground_objects)
            obstacles=project_to_2d(ground_objects)
            
            convex_hulls=[]

            # Create the plot
            fig, ax = plt.subplots()

            for obstacle in obstacles:
                points = zip(*obstacle)
                ax.scatter(*points, color='blue')  # Plot each set of obstacle points in blue

                # Calculate and plot the convex hull for each obstacle
                hull = ConvexHull(obstacle, qhull_options='QJ')

                convex_hulls.append(hull)
                for simplex in hull.simplices:
                    plt.plot([obstacle[i][0] for i in simplex], [obstacle[i][1] for i in simplex], 'k-')
            
            convex_hulls=convex_hulls+get_wall_convex_hulls()
            # Plot the start and end points
            ax.scatter(*start, color='red')   # Plot start point in red
            ax.scatter(*end, color='green')   # Plot end point in green

            # Set labels and titles
            ax.set_xlabel('X Coordinate')
            ax.set_ylabel('Y Coordinate')
            ax.set_title('Simple 2D Scene with Convex Hulls')
            plt.savefig("results/room2d.png")

            # Show the plot
            #plt.show()

           
            start = np.array(start).reshape(1, 2) 
            end = np.array(end).reshape(1, 2)  
            print("Finding path")
            
            grid, (min_x, max_x, min_y, max_y), scaled_start, scaled_end=create_grid_with_start_end(6,start,end,convex_hulls) #this is basically rasterisation so it is slow
            plot_grid_with_start_end(grid, min_x, max_x, min_y, max_y, scaled_start, scaled_end)
            path = a_star_pathfinding(grid, tuple(scaled_start[0]), tuple(scaled_end[0]))
            #plot_grid_with_path(grid, min_x, max_x, min_y, max_y, scaled_start, scaled_end,path)
            referenced_path_points=descale_points(grid, path, min_x, max_x, min_y, max_y)
            t2=time.time()
            print("Pathfinding took " +str(int(10*(t2-t1)/10)) + " seconds.")
            for i in range(0,1000):
                    try:
                        self.removeShape("path_node_" + str(i))
                    except:
                        pass
            for i,point in enumerate(referenced_path_points):

                    self.addShape(Point3D(list(point)+[path_height], color=(0,0,i/len(referenced_path_points)), size=8), "path_node_" + str(i))
                    self.updateShape("path_node_" + str(i))







        if symbol == Key.W:
            if self.show_wireframe:
                self.removeShape("wireframe")
                self.show_wireframe = False
            else:
                self.addShape(self.wireframe, "wireframe")
                self.show_wireframe = True
                




    
        if symbol == Key.SLASH:
            self.printHelp()



    def printHelp(self):
        self.print("\
        SHIFT+M1: Select start vertex\n\
        R: Reset mesh\n\
        W: Toggle wireframe\n\
        T: Find target hole (door), if clicked again changes to next biggest hole\n\
        C: Show/Hide Room Objects\n\
        P: Pathfind from currently selected Vertex\n\
        ?: Show this list\n\n")



################ PATHFINDER


##############
def project_to_2d(region_arrays):
    # Initialize a list to hold the 2D projections of each region
    regions_2d = []
    
    # Loop through each region in the input list
    for region in region_arrays:
        # Extract only the X and Y coordinates (discard the Z coordinate)
        region_2d = region[:, :2]  # Assuming the region data is in the format [x, y, z]
        # Append the 2D projection of the current region to the list
        regions_2d.append(region_2d)
    
    return regions_2d

def remove_dominant_plane_from_mesh3d(mesh3d, distance_threshold=0.1): #remove ceiling for better view
    # Ensure the input is an instance of Mesh3D
    if not isinstance(mesh3d, Mesh3D):
        raise TypeError("Input must be an instance of Mesh3D")
    
    # Get the point cloud from the mesh vertices
    pcd = o3d.geometry.PointCloud()
    pcd.points = mesh3d._shape.vertices

    # Use RANSAC to find the dominant plane and the inliers
    plane_model, inliers = pcd.segment_plane(distance_threshold=distance_threshold,
                                             ransac_n=3,
                                             num_iterations=1000)

    # Convert inliers to a set for quick lookup
    inliers_set = set(inliers)

    # Filter out triangles that have any vertex that is an inlier
    triangles_to_keep = [triangle for triangle in mesh3d.triangles if not (triangle[0] in inliers_set or triangle[1] in inliers_set or triangle[2] in inliers_set)]

    # Update the mesh with new triangles
    mesh3d.triangles = triangles_to_keep

    # Optional: clean up the mesh by removing unreferenced vertices and recalculating normals
    mesh3d.remove_unreferenced_vertices()
    mesh3d._shape.compute_vertex_normals()
    mesh3d._shape.compute_triangle_normals()

    return mesh3d

#from task2
def merge_close_vertices_in_mesh3d(mesh3d, tolerance=0.1):
    # Ensure the input is an instance of Mesh3D
    if not isinstance(mesh3d, Mesh3D):
        raise TypeError("Input must be an instance of Mesh3D")

    # Get vertices and triangles from the Mesh3D object
    vertices = np.asarray(mesh3d.vertices)
    triangles = np.asarray(mesh3d.triangles)
    
    # Use a KDTree to find close vertices
    kdtree = o3d.geometry.KDTreeFlann(mesh3d._shape)
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
    
    # Update the Mesh3D object with the new vertices and triangles
    mesh3d.vertices = np.array(merged_vertices)
    mesh3d.triangles = np.array(new_triangles)
    
    # Recompute normals if the original mesh had vertex normals
    if mesh3d._shape.has_vertex_normals():
        mesh3d._shape.compute_vertex_normals()

    return mesh3d

def remove_small_clumps_from_mesh3d(mesh3d, thres):
    if not isinstance(mesh3d, Mesh3D):
        raise TypeError("Input must be an instance of Mesh3D")
    
    # Ensure triangle normals are computed
    if not mesh3d._shape.has_triangle_normals():
        mesh3d._shape.compute_triangle_normals()
    
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

    num_triangles = len(mesh3d.triangles)
    uf = UnionFind(num_triangles)

    # Create a map from vertices to triangles for quick lookup
    vertex_map = {}
    for i, triangle in enumerate(mesh3d.triangles):
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
            large_component_triangles.append(mesh3d.triangles[i])

    # Apply colors to vertices based on triangle colors
    vertex_colors = np.zeros((len(mesh3d.vertices), 3))
    for i, triangle in enumerate(mesh3d.triangles):
        color = triangle_color_map[i]
        for vertex in triangle:
            vertex_colors[vertex] = color

    # Update vertex colors in Mesh3D
    #mesh3d.vertex_colors = vertex_colors

    # Rebuild mesh from large components, if any
    if large_component_triangles:
        new_mesh3d = Mesh3D()  # Create a new Mesh3D object
        new_mesh3d.vertices = mesh3d.vertices
        new_mesh3d.triangles = np.array(large_component_triangles)
        #new_mesh3d.vertex_colors = vertex_colors
        new_mesh3d.remove_unreferenced_vertices()
    else:
        new_mesh3d = None

    return new_mesh3d


def calculate_mesh_centroid_z(mesh):
    """
    Calculate the z-coordinate of the centroid of the mesh using Mesh3D object.
    """
    vertices = mesh.vertices  # Using the Mesh3D property to get vertices
    return np.mean(vertices[:, 2])

def filter_roof_objects(point_clouds, mesh):
    """
    Filter a list of 3D point clouds, removing those where the lowest point is above
    the height of the centroid of a given mesh defined by the Mesh3D object.

    Args:
    - point_clouds: List of numpy arrays, each representing a 3D point cloud.
    - mesh: An instance of the Mesh3D class.

    Returns:
    - A list of numpy arrays, each a filtered point cloud.
    """
    # Calculate the mesh centroid z-coordinate
    mesh_centroid_z = calculate_mesh_centroid_z(mesh)
    
    # Filter point clouds
    filtered_clouds = []
    for cloud in point_clouds:
        if np.min(cloud[:, 2]) <= mesh_centroid_z:  # Check the minimum z value of the cloud
            filtered_clouds.append(cloud)
    
    return filtered_clouds

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

def int_to_distinct_color(num):
    max_color_value = 256 ** 3  # Maximum value for a 24-bit color

    # Hash the integer and take the absolute value to ensure a positive result
    hashed_value = abs(hash(num)) % max_color_value
    
    # Convert the hash to an RGB tuple
    red = (hashed_value >> 16) & 0xFF
    green = (hashed_value >> 8) & 0xFF
    blue = hashed_value & 0xFF
    
    return (red, green, blue)

if __name__ == "__main__":
    app = Task7_scene()
    app.mainLoop()