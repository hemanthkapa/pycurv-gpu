"""
GPU batched geodesic distance via frontier relaxation on triangle graph.

Matches CPU pycurv's TriangleGraph geodesic: Dijkstra on the dual graph where
each node is a triangle centroid, edges connect adjacent triangles, weighted
by centroid-to-centroid Euclidean distance.
"""

import torch
import math
import os
import time
import numpy as np


# --- Optional profiling (env-gated, zero overhead when off) -----------------
PROFILE = os.environ.get('PYCURV_PROFILE', '0') not in ('0', '', 'false')
_prof_times = {}
_prof_counts = {}
_prof_stats = {}   # non-time counters (e.g. subgraph size), stored as (sum, n)


def _prof_sync(device):
    """Force kernel completion so timers measure real GPU work, not dispatch."""
    d = str(device)
    if 'cuda' in d:
        torch.cuda.synchronize()
    elif d == 'mps':
        torch.mps.synchronize()


def prof_add(key, seconds, count=1):
    _prof_times[key] = _prof_times.get(key, 0.0) + seconds
    _prof_counts[key] = _prof_counts.get(key, 0) + count


def prof_stat(key, value):
    """Record a non-time diagnostic (averaged in the report)."""
    s, n = _prof_stats.get(key, (0.0, 0))
    _prof_stats[key] = (s + value, n + 1)


def prof_reset():
    _prof_times.clear()
    _prof_counts.clear()
    _prof_stats.clear()


def prof_report():
    """Return a formatted breakdown of accumulated profiling timers."""
    if not _prof_times and not _prof_stats:
        return "  (profiling disabled; set PYCURV_PROFILE=1)"
    lines = []
    if _prof_times:
        width = max(len(k) for k in _prof_times)
        for k in sorted(_prof_times, key=lambda x: -_prof_times[x]):
            t = _prof_times[k]
            c = _prof_counts[k]
            lines.append(f"    {k:<{width}}  {t:7.2f}s  ({c} calls, {t/max(c,1)*1e3:.1f} ms/call)")
    for k in sorted(_prof_stats):
        s, n = _prof_stats[k]
        lines.append(f"    {k:<16}  avg {s/max(n,1):.0f}  (n={n})")
    return "\n".join(lines)


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
    device = tg.edge_src.device

    # Extract local subgraph reachable from sources within g_max
    if PROFILE:
        _prof_sync(device); _t = time.perf_counter()
    local_nodes, local_edge_src, local_edge_dst, local_edge_dist, local_sources = \
        _extract_subgraph(tg, sources, g_max, max_iters)
    if PROFILE:
        _prof_sync(device); prof_add('extract_subgraph', time.perf_counter() - _t)

    L = local_nodes.shape[0]
    if L == 0:
        empty = torch.empty(0, dtype=torch.long, device=device)
        return empty, empty, torch.empty(0, dtype=torch.float32, device=device)

    if PROFILE:
        prof_stat('subgraph_L', float(L))
        prof_stat('subgraph_E', float(local_edge_src.numel()))

    # Run dense SSSP on the compact subgraph [B, L]
    if PROFILE:
        _prof_sync(device); _t = time.perf_counter()
    dist = _sssp_dense(
        local_edge_src, local_edge_dst, local_edge_dist,
        L, local_sources, g_max=g_max, max_iters=max_iters)
    if PROFILE:
        _prof_sync(device); prof_add('sssp_dense', time.perf_counter() - _t)

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

    if has_csr and L > 0:
        # Gather induced-subgraph edges via CSR: only edges leaving reached
        # nodes, then keep those whose destination is also reached. This is
        # O(edges incident to reached) instead of O(E) over the full graph.
        starts = tg._csr_offsets[local_nodes]
        lengths = tg._csr_offsets[local_nodes + 1] - starts
        total = int(lengths.sum())
        if total > 0:
            group_off = torch.zeros(L + 1, dtype=torch.long, device=device)
            torch.cumsum(lengths, dim=0, out=group_off[1:])
            within = torch.arange(total, device=device) - \
                torch.repeat_interleave(group_off[:-1], lengths)
            flat_idx = torch.repeat_interleave(starts, lengths) + within

            # local id of local_nodes[i] is i (local_nodes is sorted ascending)
            e_src_local = torch.repeat_interleave(
                torch.arange(L, device=device), lengths)
            e_dst_local = global_to_local[tg._csr_dst[flat_idx]]
            e_w = tg._csr_dist[flat_idx]

            valid = e_dst_local >= 0
            local_edge_src = e_src_local[valid]
            local_edge_dst = e_dst_local[valid]
            local_edge_dist = e_w[valid]
        else:
            empty = torch.empty(0, dtype=torch.long, device=device)
            local_edge_src = empty
            local_edge_dst = empty
            local_edge_dist = torch.empty(0, dtype=tg.edge_dist.dtype, device=device)
    else:
        # Fallback: full-edge scan and relabel
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
    Batched SSSP via SPFA-style frontier relaxation on a compact subgraph.

    Keeps a dense [B, V] distance matrix (V ~ few K after subgraph extraction).
    Each iteration gathers edges leaving only the currently active nodes via a
    source-CSR (built once), so late iterations — where the frontier is tiny —
    no longer rescan all E_local edges. The relaxation reduces onto the unique
    destination columns only, so the full [B, V] matrix is never cloned.
    """
    device = edge_src.device
    B = sources.shape[0]
    V = num_nodes

    if max_iters is None:
        max_iters = V

    INF = float('inf')
    b_arange = torch.arange(B, device=device)

    # Source-CSR over the local subgraph: offsets[n]..offsets[n+1] index the
    # edges leaving node n. Sort once (edges may arrive unsorted from the
    # fallback extraction path); the CSR then serves every iteration.
    order = torch.argsort(edge_src)
    csr_src = edge_src[order]
    csr_dst = edge_dst[order]
    csr_w = edge_dist[order]
    counts = torch.zeros(V, dtype=torch.long, device=device)
    counts.scatter_add_(0, csr_src, torch.ones_like(csr_src))
    offsets = torch.zeros(V + 1, dtype=torch.long, device=device)
    torch.cumsum(counts, dim=0, out=offsets[1:])

    dist = torch.full((B, V), INF, dtype=torch.float32, device=device)
    dist[b_arange, sources] = 0.0

    # Per-(row, node) activity plus the set of active node ids (frontier).
    is_active = torch.zeros((B, V), dtype=torch.bool, device=device)
    is_active[b_arange, sources] = True
    active_nodes = sources.unique()

    gmax_cut = None if g_max is None else g_max + 1e-6

    for _ in range(max_iters):
        if active_nodes.numel() == 0:
            break

        # Gather edges leaving active nodes only (CSR frontier expansion).
        starts = offsets[active_nodes]
        lengths = offsets[active_nodes + 1] - starts
        total = int(lengths.sum())
        if total == 0:
            break

        group_off = torch.zeros(lengths.shape[0] + 1, dtype=torch.long, device=device)
        torch.cumsum(lengths, dim=0, out=group_off[1:])
        within = torch.arange(total, device=device) - \
            torch.repeat_interleave(group_off[:-1], lengths)
        flat = torch.repeat_interleave(starts, lengths) + within

        a_src = torch.repeat_interleave(active_nodes, lengths)
        a_dst = csr_dst[flat]
        a_w = csr_w[flat]

        # Per-row proposals; mask rows where the source isn't active this round.
        proposal = dist[:, a_src] + a_w.unsqueeze(0)
        proposal[~is_active[:, a_src]] = INF
        if gmax_cut is not None:
            proposal = proposal.clamp(max=gmax_cut)

        # Reduce onto the unique destination columns only (sparse buffer).
        d_uniq, inv = torch.unique(a_dst, return_inverse=True)
        old = dist[:, d_uniq]
        buf = old.clone()
        buf.scatter_reduce_(1, inv.unsqueeze(0).expand(B, -1), proposal,
                            reduce='amin', include_self=True)

        improved = buf < old - 1e-10
        if g_max is not None:
            improved &= (buf <= g_max)

        # Commit distances and refresh frontier, touching changed columns only.
        dist[:, d_uniq] = buf
        is_active[:, active_nodes] = False
        is_active[:, d_uniq] = improved
        col_any = improved.any(dim=0)
        active_nodes = d_uniq[col_any]

        if active_nodes.numel() == 0:
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
