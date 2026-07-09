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
    Stashes _face_point_ids and _all_vertices for later use by build_adjacency.
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
    flat_pts = faces.ravel()       # [3T]
    flat_tris = np.repeat(np.arange(T), 3)  # [3T]

    # Sort by point ID for grouping
    sort_idx = np.argsort(flat_pts)
    sorted_pts = flat_pts[sort_idx]
    sorted_tris = flat_tris[sort_idx]

    # Find group boundaries for each point
    change = np.concatenate([[0], np.where(np.diff(sorted_pts) != 0)[0] + 1,
                             [len(sorted_pts)]])
    group_sizes = np.diff(change)

    # Generate all triangle pairs sharing a vertex (fully vectorized)
    # For each entry in a group, pair it with every later entry in that group.
    # group_id[j] = which group sorted_tris[j] belongs to
    group_id = np.repeat(np.arange(len(group_sizes)), group_sizes)
    # local position within group
    local_pos = np.arange(len(sorted_tris)) - change[group_id]

    # For each element j, pair with elements j+1..end of group.
    # Repeat each element (group_size - 1 - local_pos) times.
    repeats = (group_sizes[group_id] - 1 - local_pos).astype(np.intp)
    repeats = np.maximum(repeats, 0)

    pair_a = np.repeat(sorted_tris, repeats)

    # For pair_b: for element j in group of size k, partners are
    # the elements at local positions (local_pos+1)..(k-1).
    # Build partner indices using cumulative offsets.
    total_pairs = repeats.sum()
    # For each repeated element j, the partners are consecutive entries after j
    partner_offsets = np.arange(len(sorted_tris)) + 1  # index of first partner
    partner_starts = np.repeat(partner_offsets, repeats)
    # Within each repeated block, add 0, 1, 2, ... to get successive partners
    block_lengths = repeats[repeats > 0]
    within_block = np.arange(total_pairs) - np.repeat(
        np.concatenate([[0], np.cumsum(block_lengths[:-1])]), block_lengths)
    pair_b = sorted_tris[partner_starts + within_block]

    # Canonicalize: (min, max) and count shared vertices per pair
    lo = np.minimum(pair_a, pair_b)
    hi = np.maximum(pair_a, pair_b)

    # Encode pairs as single int for fast grouping
    edge_keys = lo.astype(np.int64) * T + hi.astype(np.int64)
    unique_keys, inverse, counts = np.unique(edge_keys, return_inverse=True, return_counts=True)

    ta = unique_keys // T
    tb = unique_keys % T
    strong = (counts >= 2).astype(np.int64)

    # Bidirectional edges
    edges_src = np.concatenate([ta, tb])
    edges_dst = np.concatenate([tb, ta])
    is_strong = np.concatenate([strong, strong])

    tg.edge_src = torch.tensor(edges_src, dtype=torch.long, device=tg.device)
    tg.edge_dst = torch.tensor(edges_dst, dtype=torch.long, device=tg.device)
    tg.is_strong = torch.tensor(is_strong, dtype=torch.long, device=tg.device)

    num_strong = int(strong.sum())
    num_weak = len(strong) - num_strong
    print(f"Built adjacency: {num_strong} strong + {num_weak} weak edges")
    return tg


def compute_edge_distances(tg):
    """Compute centroid-to-centroid distances for triangle adjacency edges."""
    src_centers = tg.centers[tg.edge_src]
    dst_centers = tg.centers[tg.edge_dst]
    tg.edge_dist = torch.linalg.norm(src_centers - dst_centers, dim=1)
    return tg


def save_vtp(tg, filepath):
    """Write triangle mesh with curvature cell arrays to a VTP file."""
    points_np = tg.points.cpu().numpy()  # [T, 3, 3]
    T = points_np.shape[0]

    # Deduplicate vertices (shared points across triangles)
    flat_pts = points_np.reshape(-1, 3)  # [3T, 3]
    unique_pts, inverse = np.unique(flat_pts, axis=0, return_inverse=True)
    face_ids = inverse.reshape(T, 3)  # [T, 3] -> index into unique_pts

    # Build VTK points
    vtk_points = vtk.vtkPoints()
    vtk_points.SetData(numpy_support.numpy_to_vtk(unique_pts, deep=True))

    # Build VTK triangles
    cells = vtk.vtkCellArray()
    connectivity = np.column_stack([
        np.full(T, 3, dtype=np.int64), face_ids
    ]).ravel()
    cells.SetCells(T, numpy_support.numpy_to_vtkIdTypeArray(connectivity, deep=True))

    poly = vtk.vtkPolyData()
    poly.SetPoints(vtk_points)
    poly.SetPolys(cells)

    # Cell data arrays: (name, tensor, num_components)
    cell_arrays = [
        ('xyz', tg.centers, 3),
        ('area', tg.areas, 1),
        ('normal', tg.normals, 3),
    ]
    if tg.n_v is not None:
        cell_arrays.append(('n_v', tg.n_v, 3))
    if tg.orientation_class is not None:
        cell_arrays.append(('orientation_class', tg.orientation_class, 1))
    if tg.t_1 is not None:
        cell_arrays.append(('t_1', tg.t_1, 3))
    if tg.t_2 is not None:
        cell_arrays.append(('t_2', tg.t_2, 3))
    if tg.kappa_1 is not None:
        cell_arrays += [
            ('kappa_1', tg.kappa_1, 1),
            ('kappa_2', tg.kappa_2, 1),
            ('gauss_curvature_VV', tg.gauss_curvature, 1),
            ('mean_curvature_VV', tg.mean_curvature, 1),
            ('shape_index_VV', tg.shape_index, 1),
            ('curvedness_VV', tg.curvedness, 1),
        ]

    for name, tensor, ncomp in cell_arrays:
        arr = tensor.cpu().numpy().astype(np.float64)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        vtk_arr = numpy_support.numpy_to_vtk(arr, deep=True)
        vtk_arr.SetName(name)
        vtk_arr.SetNumberOfComponents(ncomp)
        poly.GetCellData().AddArray(vtk_arr)

    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(str(filepath))
    writer.SetInputData(poly)
    if writer.Write() != 1:
        raise RuntimeError(f"Failed to write VTP file: {filepath}")
    print(f"Wrote VTP with {T} triangles, {poly.GetCellData().GetNumberOfArrays()} arrays to {filepath}")


def save_gt(tg, filepath):
    """Write the triangle dual graph + curvature data as a graph-tool .gt file.

    Each graph vertex = one triangle; edges = shared-vertex adjacency. Curvature
    arrays are stored as vertex property maps under the same names as the VTP
    cell arrays, so pycurv's downstream readers pick them up. Requires graph-tool
    (conda install -c conda-forge graph-tool).
    """
    try:
        from graph_tool.all import Graph
    except ImportError as e:
        raise RuntimeError(
            "graph-tool is required for .gt output. Install with:\n"
            "  conda install -c conda-forge graph-tool"
        ) from e

    T = tg.num_triangles
    g = Graph(directed=False)
    g.add_vertex(T)

    # Vertex properties: (name, tensor, gt_type). Mirrors save_vtp cell arrays.
    vprops = [
        ('xyz', tg.centers, 'vector<float>'),
        ('area', tg.areas, 'float'),
        ('normal', tg.normals, 'vector<float>'),
    ]
    if tg.n_v is not None:
        vprops.append(('n_v', tg.n_v, 'vector<float>'))
    if tg.orientation_class is not None:
        vprops.append(('orientation_class', tg.orientation_class, 'int'))
    if tg.t_1 is not None:
        vprops.append(('t_1', tg.t_1, 'vector<float>'))
    if tg.t_2 is not None:
        vprops.append(('t_2', tg.t_2, 'vector<float>'))
    if tg.kappa_1 is not None:
        vprops += [
            ('kappa_1', tg.kappa_1, 'float'),
            ('kappa_2', tg.kappa_2, 'float'),
            ('gauss_curvature_VV', tg.gauss_curvature, 'float'),
            ('mean_curvature_VV', tg.mean_curvature, 'float'),
            ('shape_index_VV', tg.shape_index, 'float'),
            ('curvedness_VV', tg.curvedness, 'float'),
        ]

    for name, tensor, gt_type in vprops:
        arr = tensor.cpu().numpy()
        vp = g.new_vertex_property(gt_type)
        if gt_type == 'vector<float>':
            vp.set_2d_array(arr.astype(np.float64).T)  # [ncomp, T]
        elif gt_type == 'int':
            vp.a = arr.astype(np.int32)
        else:
            vp.a = arr.astype(np.float64)
        g.vertex_properties[name] = vp

    # Triangle corner coordinates. CPU pycurv stores these as python::object
    # (list of 3 xyz arrays per vertex). Downstream SM steps call
    # TriangleGraph.graph_to_triangle_poly(), which requires vp.points.
    if tg.points is not None:
        points_np = tg.points.cpu().numpy().astype(np.float64)  # [T, 3, 3]
        vp_points = g.new_vertex_property('object')
        for i in range(T):
            vp_points[i] = [points_np[i, 0], points_np[i, 1], points_np[i, 2]]
        g.vertex_properties['points'] = vp_points

    # Edges from the triangle dual graph (dedupe directed pairs to undirected).
    src = tg.edge_src.cpu().numpy()
    dst = tg.edge_dst.cpu().numpy()
    dist = tg.edge_dist.cpu().numpy().astype(np.float64)
    keep = src < dst
    edist = g.new_edge_property('float')
    g.add_edge_list(
        np.column_stack([src[keep], dst[keep], dist[keep]]),
        eprops=[edist],
    )
    g.edge_properties['distance'] = edist

    g.save(str(filepath))
    print(f"Wrote .gt with {T} triangles, {g.num_edges()} edges, "
          f"{len(g.vertex_properties)} vertex arrays to {filepath}")


