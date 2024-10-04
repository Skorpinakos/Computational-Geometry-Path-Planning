from vvrpywork.constants import Key, Mouse, Color
from vvrpywork.scene import Scene3D, get_rotation_matrix, world_space
from vvrpywork.shapes import (
    Point3D, Line3D, Arrow3D, Sphere3D, Cuboid3D, Cuboid3DGeneralized,
    PointSet3D, LineSet3D, Mesh3D
)

from matplotlib import colormaps as cm
import numpy as np
import scipy
from scipy import sparse
import time


WIDTH = 1000
HEIGHT = 800

def find_adjacent_vertices(mesh: Mesh3D, selected_vertex: int) -> np.ndarray:
    """
    Find the vertices adjacent to a given vertex in a 3D mesh.

    Parameters:
    mesh (Mesh3D): The 3D mesh containing vertices and triangles.
    selected_vertex (int): The index of the vertex for which to find the adjacent vertices.

    Returns:
    list: A list of indices of adjacent vertices.
    """

    list_of_adjacent_vertices = []
    # degree = 0

    # Iterate through each triangle in the mesh
    for triangle in mesh.triangles:

        # Check if the the current triangle contains the selected vertex
        if selected_vertex in triangle: 

            # Examine the vertices opposite the selected vertex
            for vertex in triangle:

                if vertex != selected_vertex:

                    # Add vertex to the adjacency list if not already present
                    if vertex not in list_of_adjacent_vertices:

                        list_of_adjacent_vertices.append(vertex)
                        # degree += 1 

    return list_of_adjacent_vertices

def find_closest_vertex(mesh: Mesh3D, query: tuple) -> int:

    dist_sq = np.sum((mesh.vertices - query) ** 2, -1)

    return np.argmin(dist_sq)

class Task7_scene(Scene3D):
    def __init__(self):
        super().__init__(WIDTH, HEIGHT,  "Task7_scene", output=True, n_sliders=2)
        self.reset_mesh()
        self.reset_sliders()
        self.printHelp()

    def reset_mesh(self):

        # Choose mesh
        # self.mesh = Mesh3D.create_bunny(color=Color.GRAY)
        self.mesh = Mesh3D("testing/resources/unicorn_low.ply", color=Color.GRAY)
        # self.mesh = Mesh3D("resources/bun_zipper_res2.ply", color=Color.GRAY)
        # self.mesh = Mesh3D("resources/dragon_low_low.obj", color=Color.GRAY)

        self.mesh.remove_duplicated_vertices()
        self.mesh.remove_unreferenced_vertices()
        vertices = self.mesh.vertices
        vertices -= np.mean(vertices, axis=0)
        distanceSq = (vertices ** 2).sum(axis=-1)
        max_dist = np.sqrt(np.max(distanceSq))
        self.mesh.vertices = vertices / max_dist
        self.removeShape("mesh")
        self.addShape(self.mesh, "mesh")

        self.wireframe = LineSet3D.create_from_mesh(self.mesh)
        self.removeShape("wireframe")
        self.addShape(self.wireframe, "wireframe")
        self.show_wireframe = True

        self.eigenvectors = None
        self.eigenvector_idx = 0



    def reset_sliders(self):
        self.set_slider_value(0, 0)
        self.set_slider_value(1, 0.5)

        
    @world_space
    def on_mouse_press(self, x, y, z, button, modifiers):
        if button == Mouse.MOUSELEFT and modifiers & Key.MOD_SHIFT:
            if np.isinf(z):
                return
            
            self.selected_vertex = find_closest_vertex(self.mesh, (x, y, z))
            print(self.mesh.vertices[self.selected_vertex])

            vc = self.mesh.vertex_colors
            vc[self.selected_vertex] = (1, 0, 0)
            self.mesh.vertex_colors = vc
            self.updateShape("mesh", True)

    def on_key_press(self, symbol, modifiers):

        if symbol == Key.R:
            self.reset_mesh()

        if symbol == Key.W:
            if self.show_wireframe:
                self.removeShape("wireframe")
                self.show_wireframe = False
            else:
                self.addShape(self.wireframe, "wireframe")
                self.show_wireframe = True
                
        if symbol == Key.A and hasattr(self, "selected_vertex"):

            if hasattr(self, "vertex_adjacency_lists"):
                adj =  self.vertex_adjacency_lists[self.selected_vertex]
            else:
                adj = find_adjacent_vertices(self.mesh, self.selected_vertex)

            colors = self.mesh.vertex_colors
            for idx in adj:
                colors[idx] = (0, 0, 1)
            self.mesh.vertex_colors = colors
            self.updateShape("mesh")



    
        if symbol == Key.SLASH:
            self.printHelp()

    def on_slider_change(self, slider_id, value):
        # if slider_id == 0:
        #     self.eigenvector_idx = int(value * (len(self.mesh.vertices) - 1))
        #     if self.eigenvectors is not None:
        #         self.display_eigenvector(self.eigenvectors[:, self.eigenvector_idx])

        if slider_id == 1:
            self.percent = 0.2 * value

    def printHelp(self):
        self.print("\
        SHIFT+M1: Select vertex\n\
        R: Reset mesh\n\
        W: Toggle wireframe\n\
        A: Adjacent vertices\n\
        ?: Show this list\n\n")




if __name__ == "__main__":
    app = Task7_scene()
    app.mainLoop()