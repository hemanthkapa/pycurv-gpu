import torch


def scale_surface(tg, pixel_size):
    """Scale all geometry by pixel_size (voxel -> physical units)."""
    if pixel_size == 1.0:
        return tg
    tg.centers *= pixel_size
    tg.points *= pixel_size
    tg.areas *= pixel_size ** 2
    tg.max_triangle_area = tg.areas.max().item()
    if tg.edge_dist is not None:
        tg.edge_dist *= pixel_size
    # _face_point_ids reference into _all_vertices which also needs scaling
    if tg._all_vertices is not None:
        tg._all_vertices *= pixel_size
    print(f"Scaled surface by pixel_size={pixel_size}")
    return tg


def find_border_triangles(tg):
    """
    Find border triangles: those with fewer than 3 STRONG neighbors.
    Matches CPU pycurv's find_graph_border (uses is_strong edges only).
    Returns bool tensor [T] where True = border triangle.
    """
    T = tg.num_triangles
    device = tg.device

    # Count only strong (shared-edge) neighbors
    if tg.is_strong is not None:
        strong_mask = tg.is_strong == 1
        strong_src = tg.edge_src[strong_mask]
    else:
        strong_src = tg.edge_src

    neighbor_count = torch.zeros(T, dtype=torch.long, device=device)
    neighbor_count.scatter_add_(0, strong_src, torch.ones_like(strong_src))
    return neighbor_count < 3


def remove_small_components(tg, min_component):
    """
    Remove connected components with fewer than min_component triangles.
    Uses GPU label propagation on the triangle adjacency graph.
    """
    T = tg.num_triangles
    device = tg.device

    # Label propagation: each triangle starts with its own label
    labels = torch.arange(T, device=device)

    for _ in range(T):
        # Propagate minimum label along edges
        new_labels = labels.clone()
        neighbor_labels = labels[tg.edge_dst]
        new_labels.scatter_reduce_(
            0, tg.edge_src, neighbor_labels,
            reduce='amin', include_self=True)

        if torch.equal(new_labels, labels):
            break
        labels = new_labels

    # Count component sizes
    unique_labels, inverse, counts = torch.unique(
        labels, return_inverse=True, return_counts=True)

    # Keep triangles in components >= min_component
    keep_mask = counts[inverse] >= min_component

    removed = T - keep_mask.sum().item()
    if removed > 0:
        _filter_triangles(tg, keep_mask)
        print(f"Removed {removed} triangles in small components "
              f"(threshold={min_component})")
    else:
        print("No small components to remove")
    return tg


def _filter_triangles(tg, keep_mask):
    """
    Remove triangles where keep_mask is False. Remaps adjacency edges.
    Updates _face_point_ids if present.
    """
    device = tg.device
    T_old = tg.num_triangles

    # Remap triangle indices: old -> new
    new_idx = torch.full((T_old,), -1, dtype=torch.long, device=device)
    kept_indices = keep_mask.nonzero(as_tuple=False).squeeze(1)
    new_idx[kept_indices] = torch.arange(kept_indices.shape[0], device=device)

    # Filter geometry tensors
    tg.centers = tg.centers[keep_mask]
    tg.normals = tg.normals[keep_mask]
    tg.areas = tg.areas[keep_mask]
    tg.points = tg.points[keep_mask]
    tg.num_triangles = int(tg.centers.shape[0])
    tg.max_triangle_area = tg.areas.max().item() if tg.num_triangles > 0 else 0.0

    # Remap edges: keep only edges between surviving triangles
    src_new = new_idx[tg.edge_src]
    dst_new = new_idx[tg.edge_dst]
    valid_edges = (src_new >= 0) & (dst_new >= 0)
    tg.edge_src = src_new[valid_edges]
    tg.edge_dst = dst_new[valid_edges]
    if tg.edge_dist is not None:
        tg.edge_dist = tg.edge_dist[valid_edges]
    if tg.is_strong is not None:
        tg.is_strong = tg.is_strong[valid_edges]

    # Filter _face_point_ids (numpy) if present
    if tg._face_point_ids is not None:
        keep_np = keep_mask.cpu().numpy()
        tg._face_point_ids = tg._face_point_ids[keep_np]

    return tg


def clean_mesh(tg, pixel_size=1.0, min_component=30):
    """Full preprocessing pipeline."""
    scale_surface(tg, pixel_size)
    if min_component > 0:
        remove_small_components(tg, min_component)
    return tg
