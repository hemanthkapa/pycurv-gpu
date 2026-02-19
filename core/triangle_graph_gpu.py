import torch
import vtk
import numpy as np 

class TriangleGraphGPU:
    def __init__(self, device = 'cuda'):
        ## Gemotric Tensors
        self.centers = None
        self.normals = None
        self.areas = None
        self.points = None
        ## Adjacency Tensors
        self.edge_src = None
        self.edge_dst = None
        self.edge_dist = None
        self.is_strong = None
        ## Distance Matrix
        self.dist_matrix = None
        ## Algorithm Outputs
        self.orientation_class = None
        self.n_v = None
        self.t_1 = None
        self.t_2 = None
        self.kappa_1 = None
        self.kappa_2 = None
        self.gauss_curvature = None
        self.mean_curvature = None
        self.curvedness = None
        self.shape_index = None
        ## Scalars
        self.num_vertices = None
        self.max_triangle_area = None
        self.device = device

    