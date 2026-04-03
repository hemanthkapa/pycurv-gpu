import vtk
import numpy as np
import torch
from vtk.util import numpy_support


def load_vtp(filepath):
    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(filepath)
    reader.Update()
    return reader.GetOutput()


def build_from_vtp(filepath, tg):
    """
    Parse a .vtp mesh and fill tg with triangle-level geometry tensors.
    Stashes _face_point_ids and _all_vertices for later use by
    build_adjacency and build_vertex_graph.
    """
    surface = load_vtp(filepath)

    tri_filter = vtk.vtkTriangleFilter()
    tri_filter.SetInputData(surface)
    tri_filter.Update()
    mesh = tri_filter.GetOutput()

    vertices = numpy_support.vtk_to_numpy(mesh.GetPoints().GetData())
    polys_flat = numpy_support.vtk_to_numpy(mesh.GetPolys().GetData())
    faces = polys_flat.reshape(-1, 4)[:, 1:4]

    p0 = vertices[faces[:, 0]]
    p1 = vertices[faces[:, 1]]
    p2 = vertices[faces[:, 2]]

    centers = (p0 + p1 + p2) / 3.0
    v1 = p1 - p0
    v2 = p2 - p0
    cross = np.cross(v1, v2)
    cross_norm = np.linalg.norm(cross, axis=1)
    areas = cross_norm / 2.0
    normals = cross / (cross_norm[:, np.newaxis] + 1e-12)

    valid = areas > 0
    valid_faces = faces[valid]

    tg.centers = torch.tensor(centers[valid], dtype=torch.float32, device=tg.device)
    tg.normals = torch.tensor(normals[valid], dtype=torch.float32, device=tg.device)
    tg.areas = torch.tensor(areas[valid], dtype=torch.float32, device=tg.device)
    tg.points = torch.tensor(
        np.stack([p0[valid], p1[valid], p2[valid]], axis=1),
        dtype=torch.float32, device=tg.device)

    tg.num_triangles = int(tg.centers.shape[0])
    tg.max_triangle_area = tg.areas.max().item()

    tg._face_point_ids = valid_faces
    tg._all_vertices = vertices.copy()

    print(f"Loaded {tg.num_triangles} triangles from {filepath}")
    return tg


def build_adjacency(tg):
    """
    Build triangle-level adjacency (dual graph edges).

    Matches CPU pycurv: triangles are connected if they share ANY vertex
    (not just shared edges). Edges sharing 2 vertices are "strong" (manifold),
    edges sharing 1 vertex are "weak". Both participate in geodesics.
    """
    faces = tg._face_point_ids  # [T, 3] global point IDs
    T = faces.shape[0]

    # Build point -> triangle mapping
    # For each vertex, find all triangles that use it
    flat_pts = faces.ravel()       # [3T]
    flat_tris = np.repeat(np.arange(T), 3)  # [3T]

    # Sort by point ID for grouping
    sort_idx = np.argsort(flat_pts)
    sorted_pts = flat_pts[sort_idx]
    sorted_tris = flat_tris[sort_idx]

    # Find boundaries between different point IDs
    change = np.concatenate([[0], np.where(np.diff(sorted_pts) != 0)[0] + 1,
                             [len(sorted_pts)]])

    # For each point, all triangles sharing it are neighbors of each other
    src_list = []
    dst_list = []
    shared_count = {}  # (min_tri, max_tri) -> count of shared vertices

    for i in range(len(change) - 1):
        tris_at_point = sorted_tris[change[i]:change[i+1]]
        n = len(tris_at_point)
        if n < 2:
            continue
        # All pairs of triangles sharing this point
        for a in range(n):
            for b in range(a + 1, n):
                ta, tb = int(tris_at_point[a]), int(tris_at_point[b])
                key = (min(ta, tb), max(ta, tb))
                shared_count[key] = shared_count.get(key, 0) + 1

    # Create bidirectional edges
    edges_src = []
    edges_dst = []
    is_strong = []

    for (ta, tb), count in shared_count.items():
        edges_src.extend([ta, tb])
        edges_dst.extend([tb, ta])
        strong = 1 if count >= 2 else 0
        is_strong.extend([strong, strong])

    tg.edge_src = torch.tensor(edges_src, dtype=torch.long, device=tg.device)
    tg.edge_dst = torch.tensor(edges_dst, dtype=torch.long, device=tg.device)
    tg.is_strong = torch.tensor(is_strong, dtype=torch.long, device=tg.device)

    num_strong = sum(1 for s in is_strong if s == 1) // 2
    num_weak = sum(1 for s in is_strong if s == 0) // 2
    print(f"Built adjacency: {num_strong} strong + {num_weak} weak edges")
    return tg


def compute_edge_distances(tg):
    """Compute centroid-to-centroid distances for triangle adjacency edges."""
    src_centers = tg.centers[tg.edge_src]
    dst_centers = tg.centers[tg.edge_dst]
    tg.edge_dist = torch.linalg.norm(src_centers - dst_centers, dim=1)
    return tg


def build_vertex_graph(tg):
    """
    Build vertex-level graph from triangle data. Must run AFTER preprocessing
    (which may remove triangles). Remaps global VTK point IDs to contiguous
    local IDs [0..P-1].

    Builds: vertex_positions, vertex_normals, vertex_areas, vertex adjacency
    (v_edge_src/dst/dist), face_vertex_ids, CSR vertex->triangle mapping.
    """
    faces_global = tg._face_point_ids  # [T, 3] global VTK point IDs
    all_vtk_verts = tg._all_vertices   # [N_vtk, 3]

    # Remap global IDs -> contiguous local IDs
    unique_global, local_ids = np.unique(faces_global, return_inverse=True)
    local_faces = local_ids.reshape(-1, 3)  # [T, 3] local vertex IDs
    P = unique_global.shape[0]
    T = local_faces.shape[0]

    # Vertex positions
    positions = all_vtk_verts[unique_global]  # [P, 3]
    tg.vertex_positions = torch.tensor(
        positions, dtype=torch.float32, device=tg.device)
    tg.num_points = P
    tg.face_vertex_ids = torch.tensor(
        local_faces, dtype=torch.long, device=tg.device)

    # Build vertex adjacency from triangle edges (mesh edges, bidirectional)
    e0 = local_faces[:, [0, 1]]
    e1 = local_faces[:, [1, 2]]
    e2 = local_faces[:, [2, 0]]
    all_edges = np.vstack([e0, e1, e2])  # [3T, 2]

    # Deduplicate: sort each edge, then unique
    sorted_e = np.sort(all_edges, axis=1)
    unique_mesh_edges = np.unique(sorted_e, axis=0)  # [E_unique, 2]

    # Bidirectional
    src = np.concatenate([unique_mesh_edges[:, 0], unique_mesh_edges[:, 1]])
    dst = np.concatenate([unique_mesh_edges[:, 1], unique_mesh_edges[:, 0]])

    tg.v_edge_src = torch.tensor(src, dtype=torch.long, device=tg.device)
    tg.v_edge_dst = torch.tensor(dst, dtype=torch.long, device=tg.device)

    # Edge distances (Euclidean between mesh vertices)
    src_pos = tg.vertex_positions[tg.v_edge_src]
    dst_pos = tg.vertex_positions[tg.v_edge_dst]
    tg.v_edge_dist = torch.linalg.norm(src_pos - dst_pos, dim=1)

    # CSR mapping: vertex -> incident triangles
    # For each triangle t, its 3 vertices point to t
    tri_indices = np.repeat(np.arange(T), 3)  # [3T]
    vert_indices = local_faces.ravel()         # [3T]

    sort_order = np.argsort(vert_indices)
    sorted_verts = vert_indices[sort_order]
    sorted_tris = tri_indices[sort_order]

    # CSR offsets
    counts = np.bincount(sorted_verts, minlength=P)
    offsets = np.zeros(P + 1, dtype=np.int64)
    np.cumsum(counts, out=offsets[1:])

    tg.point_tri_offsets = torch.tensor(offsets, dtype=torch.long, device=tg.device)
    tg.point_tri_indices = torch.tensor(sorted_tris, dtype=torch.long, device=tg.device)

    # Per-vertex normals: area-weighted average of incident triangle normals
    normals_np = tg.normals.cpu().numpy()   # [T, 3]
    areas_np = tg.areas.cpu().numpy()       # [T]
    weighted_normals = normals_np * areas_np[:, np.newaxis]  # [T, 3]

    # Scatter-add weighted normals to vertices
    vertex_normal_sum = np.zeros((P, 3), dtype=np.float64)
    for c in range(3):  # for each corner of each triangle
        np.add.at(vertex_normal_sum, local_faces[:, c], weighted_normals)

    norms = np.linalg.norm(vertex_normal_sum, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    vertex_normals = vertex_normal_sum / norms

    tg.vertex_normals = torch.tensor(
        vertex_normals, dtype=torch.float32, device=tg.device)

    # Per-vertex areas: 1/3 of incident triangle areas
    vertex_area_sum = np.zeros(P, dtype=np.float64)
    for c in range(3):
        np.add.at(vertex_area_sum, local_faces[:, c], areas_np / 3.0)

    tg.vertex_areas = torch.tensor(
        vertex_area_sum, dtype=torch.float32, device=tg.device)

    # Clean up temporaries
    del tg._face_point_ids
    del tg._all_vertices
    tg._face_point_ids = None
    tg._all_vertices = None

    print(f"Built vertex graph: {P} vertices, "
          f"{unique_mesh_edges.shape[0]} edges")
    return tg
