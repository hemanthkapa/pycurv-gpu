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

        # --- Vertex-level geometry ---
        self.vertex_positions = None   # [P, 3] mesh point coordinates
        self.vertex_normals = None     # [P, 3] area-weighted averaged normals
        self.vertex_areas = None       # [P] 1/3 sum of incident triangle areas
        self.num_points = 0

        # --- Vertex adjacency (primal mesh graph) ---
        self.v_edge_src = None    # [E_v] source vertex indices
        self.v_edge_dst = None    # [E_v] destination vertex indices
        self.v_edge_dist = None   # [E_v] Euclidean edge lengths

        # --- Triangle <-> Vertex mappings ---
        self.face_vertex_ids = None    # [T, 3] local vertex IDs per triangle
        self.point_tri_offsets = None  # [P+1] CSR offsets: vertex -> triangles
        self.point_tri_indices = None  # [total] CSR values: triangle indices

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
