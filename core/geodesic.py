"""
GPU batched geodesic distance via frontier relaxation on triangle graph.

Matches CPU pycurv's TriangleGraph geodesic: Dijkstra on the dual graph where
each node is a triangle centroid, edges connect adjacent triangles, weighted
by centroid-to-centroid Euclidean distance.
"""

import torch
import math


def sssp_batch(edge_src, edge_dst, edge_dist, num_nodes,
               sources, g_max=None, max_iters=None):
    """
    Batched single-source shortest paths via frontier relaxation.

    Args:
        edge_src, edge_dst: [E] bidirectional graph edges
        edge_dist: [E] edge weights
        num_nodes: N (number of nodes)
        sources: [B] source node indices
        g_max: distance cutoff (None = unlimited)
        max_iters: max relaxation rounds

    Returns:
        dist: [B, N] shortest-path distances (inf where unreachable)
    """
    device = edge_src.device
    B = sources.shape[0]
    V = num_nodes

    if max_iters is None:
        max_iters = V

    dist = torch.full((B, V), float('inf'), dtype=torch.float32, device=device)
    dist[torch.arange(B, device=device), sources] = 0.0

    is_active = torch.zeros((B, V), dtype=torch.bool, device=device)
    is_active[torch.arange(B, device=device), sources] = True

    for _ in range(max_iters):
        any_active = is_active.any(dim=0)
        if not any_active.any():
            break

        frontier_mask = any_active[edge_src]
        if not frontier_mask.any():
            break

        a_src = edge_src[frontier_mask]
        a_dst = edge_dst[frontier_mask]
        a_w = edge_dist[frontier_mask]
        A = a_src.shape[0]

        row_active = is_active[:, a_src]
        proposal = dist[:, a_src] + a_w.unsqueeze(0)
        proposal[~row_active] = float('inf')

        if g_max is not None:
            proposal = proposal.clamp(max=g_max + 1e-6)

        # Scatter proposals into a small buffer for unique destinations only
        unique_dst, inv = a_dst.unique(return_inverse=True)
        D = unique_dst.shape[0]

        old_vals = dist[:, unique_dst].clone()  # [B, D]
        buf = old_vals.clone()

        inv_expanded = inv.unsqueeze(0).expand(B, A)
        buf.scatter_reduce_(1, inv_expanded, proposal,
                            reduce='amin', include_self=True)

        improved_at_dst = buf < old_vals - 1e-10
        if g_max is not None:
            improved_at_dst &= (buf <= g_max)

        dist[:, unique_dst] = buf
        is_active.zero_()
        is_active[:, unique_dst] = improved_at_dst

        if not is_active.any():
            break

    return dist


def sssp_triangle_batch(tg, sources, g_max):
    """
    Batched SSSP on the triangle adjacency graph.
    Returns [B, T] distance matrix between triangle centroids.
    """
    max_iters = _estimate_max_iters(tg.edge_dist, tg.num_triangles, g_max)
    return sssp_batch(
        tg.edge_src, tg.edge_dst, tg.edge_dist,
        tg.num_triangles, sources, g_max=g_max, max_iters=max_iters)


def _estimate_max_iters(edge_dist, num_nodes, g_max):
    if g_max is not None and edge_dist.numel() > 0:
        min_edge = edge_dist.min().item()
        if min_edge > 0:
            return min(int(math.ceil(g_max / min_edge)) + 2, num_nodes)
    return num_nodes


def get_free_memory(device):
    if 'cuda' in str(device) and torch.cuda.is_available():
        try:
            idx = int(str(device).split(':')[1]) if ':' in str(device) else 0
            free, _ = torch.cuda.mem_get_info(idx)
            return free
        except Exception:
            return 8e9
    elif str(device) == 'mps':
        return 8e9
    return 4e9


def auto_batch_size(num_nodes, requested, device):
    """Auto-tune batch size to fit GPU memory."""
    bytes_per_source = num_nodes * 9 + 8000
    free_mem = get_free_memory(device)
    max_bytes = free_mem * 0.35
    safe = max(1, int(max_bytes / bytes_per_source))
    chosen = min(requested, safe)
    if chosen < requested:
        print(f"Auto-reduced batch_size {requested} -> {chosen} "
              f"({free_mem / 1e9:.1f} GB free)")
    return chosen
