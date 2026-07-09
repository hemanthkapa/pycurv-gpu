import torch


class TriangleGraphGPU:
    def __init__(self, device='cuda'):
        self.device = device

        # --- Triangle-level geometry ---
        self.centers = None       # [T, 3] triangle centroids
        self.normals = None       # [T, 3] triangle unit normals
        self.areas = None         # [T] triangle areas
        self.points = None        # [T, 3, 3] corner coordinates
        self.num_triangles = 0
        self.max_triangle_area = 0.0

        # --- Triangle adjacency (dual graph) ---
        self.edge_src = None      # [E_t] source triangle indices
        self.edge_dst = None      # [E_t] destination triangle indices
        self.edge_dist = None     # [E_t] centroid-to-centroid distances

        # --- Algorithm outputs (per triangle) ---
        self.n_v = None               # [T, 3] estimated normals (Pass 1)
        self.orientation_class = None  # [T] 1=surface, 2=crease, 3=noise
        self.t_1 = None               # [T, 3] principal direction 1
        self.t_2 = None               # [T, 3] principal direction 2
        self.kappa_1 = None           # [T] max principal curvature
        self.kappa_2 = None           # [T] min principal curvature
        self.gauss_curvature = None   # [T]
        self.mean_curvature = None    # [T]
        self.curvedness = None        # [T]
        self.shape_index = None       # [T]

        # --- Temporaries (numpy, discarded after build) ---
        self._face_point_ids = None   # [T, 3] global VTK point IDs
        self._all_vertices = None     # [N_vtk, 3] all VTK vertex positions
