"""
GPU batched geodesic distance via delta-stepping on triangle graph.

Delta-stepping: GPU-friendly Dijkstra variant that processes nodes in
distance-ordered buckets of width delta. Within each bucket, nodes are
relaxed in parallel. Settled nodes are never revisited.

Same interface as geodesic.py — drop-in replacement for comparison.
"""

import torch
import math


def build_csr(tg):
    """Precompute CSR adjacency for fast neighbor lookup. Call once after adjacency build."""
    T = tg.num_triangles
    device = tg.edge_src.device

    sort_idx = torch.argsort(tg.edge_src)
    sorted_src = tg.edge_src[sort_idx]
    tg._csr_dst = tg.edge_dst[sort_idx]
    tg._csr_dist = tg.edge_dist[sort_idx]

    counts = torch.zeros(T, dtype=torch.long, device=device)
    counts.scatter_add_(0, sorted_src, torch.ones_like(sorted_src))
    offsets = torch.zeros(T + 1, dtype=torch.long, device=device)
    torch.cumsum(counts, dim=0, out=offsets[1:])
    tg._csr_offsets = offsets


def sssp_triangle_batch(tg, sources, g_max):
    """
    Batched SSSP via delta-stepping on the triangle adjacency graph.

    Returns sparse neighbor results (same interface as geodesic.py):
        src_local: [N] batch-local source index (0..B-1)
        nbr_idx:   [N] global triangle index of neighbor
        g_i:       [N] geodesic distance from source to neighbor
    """
    max_iters = _estimate_max_iters(tg.edge_dist, tg.num_triangles, g_max)

    local_nodes, local_edge_src, local_edge_dst, local_edge_dist, local_sources = \
        _extract_subgraph(tg, sources, g_max, max_iters)

    L = local_nodes.shape[0]
    if L == 0:
        device = tg.edge_src.device
        empty = torch.empty(0, dtype=torch.long, device=device)
        return empty, empty, torch.empty(0, dtype=torch.float32, device=device)

    dist = _sssp_delta_stepping(
        local_edge_src, local_edge_dst, local_edge_dist,
        L, local_sources, g_max=g_max)

    is_nbr = (dist > 0) & (dist <= g_max)
    src_local, local_nbr = is_nbr.nonzero(as_tuple=True)

    if src_local.numel() == 0:
        device = tg.edge_src.device
        empty = torch.empty(0, dtype=torch.long, device=device)
        return empty, empty, torch.empty(0, dtype=torch.float32, device=device)

    g_i = dist[src_local, local_nbr]
    nbr_idx = local_nodes[local_nbr]

    return src_local, nbr_idx, g_i


def _sssp_delta_stepping(edge_src, edge_dst, edge_dist, num_nodes,
                         sources, g_max):
    """
    Batched delta-stepping SSSP on a compact subgraph.

    Instead of relaxing all active edges (Bellman-Ford), processes nodes
    in distance buckets of width delta. Within each bucket, only relax
    from nodes in that bucket. Settled nodes (in earlier buckets) are
    never revisited.

    Key differences from Bellman-Ford _sssp_dense:
    - No dist.clone() per iteration — settled nodes are frozen
    - Fewer total relaxations — each node processed in exactly one bucket
    - Naturally stops when bucket > g_max
    """
    device = edge_src.device
    B = sources.shape[0]
    V = num_nodes

    # Choose delta = median edge weight (good for mesh graphs with ~uniform weights)
    delta = edge_dist.median().item()
    if delta <= 0:
        delta = edge_dist[edge_dist > 0].min().item() if (edge_dist > 0).any() else 1.0

    dist = torch.full((B, V), float('inf'), dtype=torch.float32, device=device)
    dist[torch.arange(B, device=device), sources] = 0.0

    # Track which nodes have been permanently settled
    settled = torch.zeros((B, V), dtype=torch.bool, device=device)

    # Pre-classify edges: "light" (weight <= delta) vs "heavy"
    light_mask = edge_dist <= delta
    heavy_mask = ~light_mask

    light_src, light_dst, light_w = edge_src[light_mask], edge_dst[light_mask], edge_dist[light_mask]
    heavy_src, heavy_dst, heavy_w = edge_src[heavy_mask], edge_dst[heavy_mask], edge_dist[heavy_mask]

    bucket_lo = 0.0
    max_buckets = int(math.ceil((g_max + 1e-6) / delta)) + 1

    for _ in range(max_buckets):
        bucket_hi = bucket_lo + delta

        # Nodes in current bucket: distance in [bucket_lo, bucket_hi) and not settled
        in_bucket = (dist >= bucket_lo) & (dist < bucket_hi) & ~settled

        if not in_bucket.any():
            # No nodes in this bucket — check if any unsettled nodes remain within g_max
            unsettled_valid = ~settled & (dist <= g_max)
            if not unsettled_valid.any():
                break
            # Jump to next non-empty bucket
            masked_dist = dist.clone()
            masked_dist[settled | (dist > g_max)] = float('inf')
            min_remaining = masked_dist.min().item()
            if min_remaining == float('inf'):
                break
            bucket_lo = (min_remaining // delta) * delta
            continue

        # Phase 1: relax light edges within bucket (may need multiple passes)
        for _ in range(V):  # bounded but typically 1-3 passes
            # Find active nodes in bucket
            active = in_bucket  # [B, V]
            any_active_col = active.any(dim=0)  # [V]

            if not any_active_col.any():
                break

            # Filter to light edges from active source nodes
            src_active = any_active_col[light_src]
            if not src_active.any():
                break

            a_src = light_src[src_active]
            a_dst = light_dst[src_active]
            a_w = light_w[src_active]
            A = a_src.shape[0]

            # Compute proposals: dist[:, src] + weight, only for active (src, batch) pairs
            row_active = active[:, a_src]  # [B, A]
            proposal = dist[:, a_src] + a_w.unsqueeze(0)  # [B, A]
            proposal[~row_active] = float('inf')
            proposal = proposal.clamp(max=g_max + 1e-6)

            # Apply relaxation in-place via scatter
            dst_expanded = a_dst.unsqueeze(0).expand(B, A)
            old_at_dst = dist.gather(1, dst_expanded).clone()
            dist.scatter_reduce_(1, dst_expanded, proposal, reduce='amin', include_self=True)

            # Check what improved
            new_at_dst = dist.gather(1, dst_expanded)
            any_improved = (new_at_dst < old_at_dst - 1e-10).any()

            # Update in_bucket: newly reached nodes in this bucket range
            in_bucket = (dist >= bucket_lo) & (dist < bucket_hi) & ~settled

            if not any_improved:
                break

        # Settle all nodes in this bucket
        to_settle = (dist >= bucket_lo) & (dist < bucket_hi) & ~settled
        settled |= to_settle

        # Phase 2: relax heavy edges from newly settled nodes
        if heavy_src.numel() > 0:
            settled_col = to_settle.any(dim=0)
            src_settled = settled_col[heavy_src]

            if src_settled.any():
                a_src = heavy_src[src_settled]
                a_dst = heavy_dst[src_settled]
                a_w = heavy_w[src_settled]
                A = a_src.shape[0]

                row_settled = to_settle[:, a_src]
                proposal = dist[:, a_src] + a_w.unsqueeze(0)
                proposal[~row_settled] = float('inf')
                proposal = proposal.clamp(max=g_max + 1e-6)

                dst_expanded = a_dst.unsqueeze(0).expand(B, A)
                dist.scatter_reduce_(1, dst_expanded, proposal, reduce='amin', include_self=True)

        bucket_lo = bucket_hi

    return dist


# --- Subgraph extraction (identical to geodesic.py) ---

def _extract_subgraph(tg, sources, g_max, max_iters):
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
            starts = tg._csr_offsets[frontier]
            ends = tg._csr_offsets[frontier + 1]
            lengths = ends - starts
            total_edges = lengths.sum().item()

            if total_edges == 0:
                break

            group_offsets = torch.zeros(lengths.shape[0] + 1, dtype=torch.long, device=device)
            torch.cumsum(lengths, dim=0, out=group_offsets[1:])
            within_group = torch.arange(total_edges, device=device) - \
                torch.repeat_interleave(group_offsets[:-1], lengths)
            flat_idx = torch.repeat_interleave(starts, lengths) + within_group

            f_src = torch.repeat_interleave(frontier, lengths)
            f_dst = tg._csr_dst[flat_idx]
            f_w = tg._csr_dist[flat_idx]
        else:
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

    local_nodes = reached.nonzero(as_tuple=False).squeeze(1)
    L = local_nodes.shape[0]

    global_to_local = torch.full((T,), -1, dtype=torch.long, device=device)
    global_to_local[local_nodes] = torch.arange(L, device=device)

    src_local = global_to_local[tg.edge_src]
    dst_local = global_to_local[tg.edge_dst]
    valid = (src_local >= 0) & (dst_local >= 0)

    local_edge_src = src_local[valid]
    local_edge_dst = dst_local[valid]
    local_edge_dist = tg.edge_dist[valid]

    local_sources = global_to_local[sources]

    return local_nodes, local_edge_src, local_edge_dst, local_edge_dist, local_sources


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
    bytes_per_source = num_nodes * 9 + 8000
    free_mem = get_free_memory(device)
    max_bytes = free_mem * 0.35
    safe = max(1, int(max_bytes / bytes_per_source))
    chosen = min(requested, safe)
    if chosen < requested:
        print(f"Auto-reduced batch_size {requested} -> {chosen} "
              f"({free_mem / 1e9:.1f} GB free)")
    return chosen
