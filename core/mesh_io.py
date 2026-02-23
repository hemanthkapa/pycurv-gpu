import vtk
import numpy as np
import torch
from vtk.util import numpy_support


def load_vtp(filepath):
    """
    Load a VTK PolyData surface from a .vtp file.

    Args:
        filepath (str): path to the .vtp file
    Returns:
        vtk.vtkPolyData: the loaded surface
    """
    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(filepath)
    reader.Update()
    return reader.GetOutput()


def build_from_vtp(filepath, tg):
    """
    Parse a .vtp mesh file and fill a TriangleGraphGPU with geometry tensors.
    Uses fully vectorized numpy operations — no Python loop over triangles.
    Modifies tg in-place. Returns tg for convenience.

    Args:
        filepath (str): path to the .vtp file
        tg (TriangleGraphGPU): empty graph object to fill
    Returns:
        TriangleGraphGPU: the same tg object, now populated
    Notes:
        tg._face_point_ids (np.ndarray [N, 3]): global point IDs per triangle,
        stored as a temporary numpy array needed for adjacency building.
    """
    surface = load_vtp(filepath)

    # Ensure all cells are triangles 
    tri_filter = vtk.vtkTriangleFilter()
    tri_filter.SetInputData(surface)
    tri_filter.Update()
    mesh = tri_filter.GetOutput()

    # --- Extract raw arrays from VTK ---

    # All unique vertex coordinates: shape (N_points, 3)
    vertices = numpy_support.vtk_to_numpy(mesh.GetPoints().GetData())

    # Connectivity list: VTK stores [3, id0, id1, id2, 3, id3, id4, id5, ...]
    # Reshape to (N_faces, 4) then drop column 0 (the '3's) → (N_faces, 3)
    polys_flat = numpy_support.vtk_to_numpy(mesh.GetPolys().GetData())
    faces = polys_flat.reshape(-1, 4)[:, 1:4]  # global point IDs per triangle

    # --- Vectorized geometry: all N triangles at once ---

    # Corner coordinates: each shape (N_faces, 3)
    p0 = vertices[faces[:, 0]]
    p1 = vertices[faces[:, 1]]
    p2 = vertices[faces[:, 2]]

    # Centroids: average of 3 corners
    centers = (p0 + p1 + p2) / 3.0

    # Normals and areas via cross product
    # |v1 × v2| = 2 * area; normalize to get unit normal
    v1 = p1 - p0
    v2 = p2 - p0
    cross = np.cross(v1, v2)
    cross_norm = np.linalg.norm(cross, axis=1)           # (N,)
    areas = cross_norm / 2.0
    normals = cross / (cross_norm[:, np.newaxis] + 1e-12)  # avoid div-by-zero

    # --- Filter degenerate (zero-area) triangles ---
    valid_mask = areas > 0
    valid_faces = faces[valid_mask]  # needed later for adjacency building

    # --- Send geometry tensors to GPU ---
    tg.centers = torch.tensor(centers[valid_mask], dtype=torch.float32, device=tg.device)
    tg.normals = torch.tensor(normals[valid_mask], dtype=torch.float32, device=tg.device)
    tg.areas   = torch.tensor(areas[valid_mask],   dtype=torch.float32, device=tg.device)
    tg.points  = torch.tensor(
        np.stack([p0[valid_mask], p1[valid_mask], p2[valid_mask]], axis=1),
        dtype=torch.float32,
        device=tg.device
    )

    # Scalars
    tg.num_vertices = int(tg.centers.shape[0])
    tg.max_triangle_area = tg.areas.max().item()

    # Stash point IDs on tg temporarily — adjacency builder will use and delete this
    tg._face_point_ids = valid_faces  # np.ndarray [N, 3], not a tensor

    print(f"Loaded {tg.num_vertices} triangles from {filepath}")
    return tg

def build_adjacency(tg):
    """
    Build adjacency information for the triangle graph.
    """
    faces = tg._face_point_ids
    N = faces.shape[0]

    # Extract all edges and keep track of which triangle they belong to
    edge0 = faces[:, [0, 1]]
    edge1 = faces[:, [1, 2]]
    edge2 = faces[:, [2, 0]]
    
    all_edges = np.vstack([edge0, edge1, edge2])
    tri_ids = np.tile(np.arange(N), 3)

    # Sort the vertices in each edge so (A, B) matches (B, A)
    sorted_edges = np.sort(all_edges, axis=1)

    # Find unique edges and map them back
    unique_edges, inverse_indices, counts = np.unique(
        sorted_edges, axis=0, return_inverse=True, return_counts=True
    )

    # Filter for edges shared by exactly two triangles (manifold edges)
    valid_unique_mask = counts == 2
    
    # Create a mask for all original edges that are shared
    is_shared_edge = valid_unique_mask[inverse_indices]
    
    # Extract the triangle IDs and edge IDs only for shared edges
    shared_tri_ids = tri_ids[is_shared_edge]
    shared_edge_inverses = inverse_indices[is_shared_edge]

    # Group by the shared edge ID
    sort_order = np.argsort(shared_edge_inverses)
    sorted_shared_tris = shared_tri_ids[sort_order]

    # Split into source and destination (since they come in pairs of 2)
    src_tris = sorted_shared_tris[0::2]
    dst_tris = sorted_shared_tris[1::2]

    # Create bidirectional graph edges in PyTorch
    edges_src_concat = np.concatenate([src_tris, dst_tris])
    edges_dst_concat = np.concatenate([dst_tris, src_tris])

    tg.edge_src = torch.tensor(edges_src_concat, dtype=torch.long, device=tg.device)
    tg.edge_dst = torch.tensor(edges_dst_concat, dtype=torch.long, device=tg.device) 

    return tg

def compute_edge_distances(tg):
    """
    Compute edge distances for the triangle graph. 

    """
    src_centers = tg.centers[tg.edge_src]
    dst_centers = tg.centers[tg.edge_dst]

    tg.edge_dist = torch.linalg.norm(src_centers - dst_centers, dim=1)

    return tg

