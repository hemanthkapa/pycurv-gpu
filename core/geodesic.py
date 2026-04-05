"""
GPU batched geodesic distance via frontier relaxation on triangle graph.

Matches CPU pycurv's TriangleGraph geodesic: Dijkstra on the dual graph where
each node is a triangle centroid, edges connect adjacent triangles, weighted
by centroid-to-centroid Euclidean distance.
"""

import torch
import math
import numpy as np


def build_csr(tg):
    """Precompute CSR adjacency for fast neighbor lookup. Call once after adjacency build."""
    T = tg.num_triangles
    device = tg.edge_src.device

    # Sort edges by source for CSR
    sort_idx = torch.argsort(tg.edge_src)
    sorted_src = tg.edge_src[sort_idx]
    tg._csr_dst = tg.edge_dst[sort_idx]
    tg._csr_dist = tg.edge_dist[sort_idx]

    # Compute offsets
    counts = torch.zeros(T, dtype=torch.long, device=device)
    counts.scatter_add_(0, sorted_src, torch.ones_like(sorted_src))
    offsets = torch.zeros(T + 1, dtype=torch.long, device=device)
    torch.cumsum(counts, dim=0, out=offsets[1:])
    tg._csr_offsets = offsets


def sssp_triangle_batch(tg, sources, g_max):
    """
    Batched SSSP on the triangle adjacency graph.

    Returns sparse neighbor results:
        src_local: [N] batch-local source index (0..B-1)
        nbr_idx:   [N] global triangle index of neighbor
        g_i:       [N] geodesic distance from source to neighbor
    """
    max_iters = _estimate_max_iters(tg.edge_dist, tg.num_triangles, g_max)

    # Extract local subgraph reachable from sources within g_max
    local_nodes, local_edge_src, local_edge_dst, local_edge_dist, local_sources = \
        _extract_subgraph(tg, sources, g_max, max_iters)

    L = local_nodes.shape[0]
    if L == 0:
        device = tg.edge_src.device
        empty = torch.empty(0, dtype=torch.long, device=device)
        return empty, empty, torch.empty(0, dtype=torch.float32, device=device)

    # Run dense SSSP on the compact subgraph [B, L]
    dist = _sssp_dense(
        local_edge_src, local_edge_dst, local_edge_dist,
        L, local_sources, g_max=g_max, max_iters=max_iters)

    # Extract sparse neighbors: dist > 0 and <= g_max
    is_nbr = (dist > 0) & (dist <= g_max)
    src_local, local_nbr = is_nbr.nonzero(as_tuple=True)

    if src_local.numel() == 0:
        device = tg.edge_src.device
        empty = torch.empty(0, dtype=torch.long, device=device)
        return empty, empty, torch.empty(0, dtype=torch.float32, device=device)

    g_i = dist[src_local, local_nbr]
    nbr_idx = local_nodes[local_nbr]

    return src_local, nbr_idx, g_i


def _extract_subgraph(tg, sources, g_max, max_iters):
    """
    BFS from sources using CSR adjacency to find reachable nodes within g_max.
    Returns remapped subgraph with local IDs.
    """
    device = tg.edge_src.device
    T = tg.num_triangles
    has_csr = hasattr(tg, '_csr_offsets') and tg._csr_offsets is not None

    reached = torch.zeros(T, dtype=torch.bool, device=device)
    reached[sources] = True
    min_dist = torch.full((T,), float('inf'), dtype=torch.float32, device=device)
    min_dist[sources] = 0.0
    frontier = sources.clone()

    for _ in range(max_iters):
        if frontier.numel() == 0:
            break

        if has_csr:
            # CSR lookup: gather only edges from frontier nodes
            starts = tg._csr_offsets[frontier]
            ends = tg._csr_offsets[frontier + 1]
            lengths = ends - starts
            total_edges = lengths.sum().item()

            if total_edges == 0:
                break

            # Build flat index into CSR arrays
            # For each frontier node, generate indices start, start+1, ..., end-1
            group_offsets = torch.zeros(lengths.shape[0] + 1, dtype=torch.long, device=device)
            torch.cumsum(lengths, dim=0, out=group_offsets[1:])
            within_group = torch.arange(total_edges, device=device) - \
                torch.repeat_interleave(group_offsets[:-1], lengths)
            flat_idx = torch.repeat_interleave(starts, lengths) + within_group

            f_src = torch.repeat_interleave(frontier, lengths)
            f_dst = tg._csr_dst[flat_idx]
            f_w = tg._csr_dist[flat_idx]
        else:
            # Fallback: scan all edges
            is_frontier = torch.zeros(T, dtype=torch.bool, device=device)
            is_frontier[frontier] = True
            mask = is_frontier[tg.edge_src]
            f_src = tg.edge_src[mask]
            f_dst = tg.edge_dst[mask]
            f_w = tg.edge_dist[mask]

        proposed = min_dist[f_src] + f_w
        within = proposed <= g_max + 1e-6
        f_dst = f_dst[within]
        proposed = proposed[within]
        if f_dst.numel() == 0:
            break

        old = min_dist[f_dst].clone()
        min_dist.scatter_reduce_(0, f_dst, proposed, reduce='amin', include_self=True)
        improved = min_dist[f_dst] < old

        new_reached = ~reached[f_dst]
        reached[f_dst[new_reached]] = True

        frontier_mask = new_reached | improved
        frontier = f_dst[frontier_mask].unique()

    # Build local ID mapping
    local_nodes = reached.nonzero(as_tuple=False).squeeze(1)
    L = local_nodes.shape[0]

    global_to_local = torch.full((T,), -1, dtype=torch.long, device=device)
    global_to_local[local_nodes] = torch.arange(L, device=device)

    # Filter edges to subgraph
    src_local = global_to_local[tg.edge_src]
    dst_local = global_to_local[tg.edge_dst]
    valid = (src_local >= 0) & (dst_local >= 0)

    local_edge_src = src_local[valid]
    local_edge_dst = dst_local[valid]
    local_edge_dist = tg.edge_dist[valid]

    local_sources = global_to_local[sources]

    return local_nodes, local_edge_src, local_edge_dst, local_edge_dist, local_sources


def _sssp_dense(edge_src, edge_dst, edge_dist, num_nodes,
                sources, g_max=None, max_iters=None):
    """
    Dense batched SSSP via frontier relaxation on a compact subgraph.
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

        unique_dst, inv = a_dst.unique(return_inverse=True)

        old_vals = dist[:, unique_dst].clone()
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
