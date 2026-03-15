"""
GPU-accelerated geodesic distance computation using parallel Bellman-Ford.

Replaces graph_tool.topology.shortest_distance used in the original pycurv.
Geodesic distances on the triangle graph are approximated as shortest paths
along edges weighted by Euclidean distance between triangle centers.

All operations run on GPU via PyTorch tensor operations — no Python loops
over vertices or edges.
"""

import torch
import math


# ---------------------------------------------------------------------------
# Core: batched parallel Bellman-Ford
# ---------------------------------------------------------------------------

def _bellman_ford_batch(edge_src, edge_dst, edge_dist, num_vertices,
                        source_indices, g_max=None, max_iters=None):
    """
    Parallel Bellman-Ford shortest paths from a batch of source vertices.

    Every iteration relaxes ALL edges simultaneously on GPU.  A g_max cutoff
    clamps distances so that the wavefront stops spreading beyond the
    neighborhood of interest, which also speeds convergence.

    Args:
        edge_src (Tensor [E], long):   source vertex for each directed edge
        edge_dst (Tensor [E], long):   destination vertex for each directed edge
        edge_dist (Tensor [E], float): weight (distance) of each edge
        num_vertices (int):            total number of vertices in the graph
        source_indices (Tensor [B], long): batch of source vertex indices
        g_max (float or None):         if given, distances are clamped to this
                                       value (vertices beyond g_max are treated
                                       as unreachable)
        max_iters (int or None):       hard cap on relaxation iterations;
                                       defaults to num_vertices (guaranteed
                                       convergence for any graph)

    Returns:
        dist (Tensor [B, V], float32): shortest-path distance from each source
            to every vertex.  Unreachable vertices have value ``g_max`` (if
            given) or ``inf``.
    """
    device = edge_src.device
    B = source_indices.shape[0]
    E = edge_src.shape[0]

    if max_iters is None:
        max_iters = num_vertices

    # Initialise distances: 0 at sources, inf everywhere else
    dist = torch.full((B, num_vertices), float('inf'),
                      dtype=torch.float32, device=device)
    dist[torch.arange(B, device=device), source_indices] = 0.0

    # Pre-expand edge indices for the batch dimension → (B, E)
    edge_dst_exp = edge_dst.unsqueeze(0).expand(B, E)

    for _ in range(max_iters):
        # Proposed new distances through each edge: dist[b, src] + w
        proposal = dist[:, edge_src] + edge_dist.unsqueeze(0)   # (B, E)

        # Scatter-min into destinations
        updated = dist.clone()
        updated.scatter_reduce_(1, edge_dst_exp, proposal, reduce='amin',
                                include_self=True)

        # Clamp to g_max so the wavefront doesn't spread further
        if g_max is not None:
            updated.clamp_(max=g_max)

        # Convergence check — if nothing changed we're done
        if torch.equal(updated, dist):
            break
        dist = updated

    return dist


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_geodesic_distances(tg, g_max=None, batch_size=512):
    """
    Compute all-pairs geodesic (shortest-path) distances on the triangle
    graph, optionally clamped to *g_max*.

    Stores the result in ``tg.dist_matrix`` (shape ``[V, V]``).

    For large meshes the computation is split into batches of source vertices
    to control GPU memory.  Each batch processes ``batch_size`` source
    vertices in parallel.

    Args:
        tg (TriangleGraphGPU): triangle graph with adjacency and edge
            distances already computed (``edge_src``, ``edge_dst``,
            ``edge_dist`` must be populated).
        g_max (float or None): geodesic distance cutoff.  Distances larger
            than this are stored as *g_max* (effectively "unreachable").
            When ``None``, no cutoff is applied (full shortest paths).
        batch_size (int): number of source vertices per batch (default 512).
            Larger values use more GPU memory but run faster.  For a mesh
            with V vertices the peak memory is roughly
            ``batch_size * V * 4`` bytes.

    Returns:
        tg (TriangleGraphGPU): the same object, with ``tg.dist_matrix``
            filled in.

    Note:
        For large meshes (> ~30K vertices) the full V×V matrix may not fit
        in GPU memory.  In that case, use ``find_geodesic_neighbors`` with
        on-the-fly computation (``tg.dist_matrix = None``) or use
        ``compute_geodesic_neighbors_sparse`` which stores only the
        neighbors within g_max in a compact CSR format.
    """
    V = tg.num_vertices
    device = tg.device

    # Check if the full matrix will fit (rough estimate)
    matrix_bytes = V * V * 4
    if matrix_bytes > 4e9:  # > 4 GB
        print(f"Warning: full {V}x{V} distance matrix would use "
              f"{matrix_bytes / 1e9:.1f} GB. Consider using "
              f"compute_geodesic_neighbors_sparse() instead.")

    # Estimate max iterations from the cutoff and edge lengths
    max_iters = _estimate_max_iters(tg.edge_dist, V, g_max)

    # Auto-tune batch size based on available memory
    batch_size = _auto_batch_size(V, tg.edge_src.shape[0], batch_size, device)

    # Process in batches, storing results on CPU to save GPU memory
    rows = []
    for start in range(0, V, batch_size):
        end = min(start + batch_size, V)
        sources = torch.arange(start, end, device=device)
        batch_dist = _bellman_ford_batch(
            tg.edge_src, tg.edge_dst, tg.edge_dist,
            V, sources, g_max=g_max, max_iters=max_iters)
        rows.append(batch_dist)

    tg.dist_matrix = torch.cat(rows, dim=0)  # (V, V)

    print(f"Computed geodesic distance matrix [{V}, {V}]"
          f"{' with g_max=' + f'{g_max:.4f}' if g_max is not None else ''}")
    return tg


def compute_geodesic_neighbors_sparse(tg, g_max, batch_size=256):
    """
    Compute geodesic neighbors within *g_max* for ALL vertices, storing the
    result in a memory-efficient CSR (compressed sparse row) format.

    Unlike ``compute_geodesic_distances`` this does NOT build a full V×V
    matrix, making it suitable for large meshes (100K+ triangles).

    The results are stored on ``tg`` as:
        - ``tg.neighbor_offsets``  [V+1]  (long)
        - ``tg.neighbor_indices``  [total_pairs]  (long)
        - ``tg.neighbor_dists``    [total_pairs]  (float32)

    Vertex ``i`` has neighbors at positions
    ``tg.neighbor_indices[offsets[i]:offsets[i+1]]`` with distances
    ``tg.neighbor_dists[offsets[i]:offsets[i+1]]``.

    Args:
        tg (TriangleGraphGPU): triangle graph with adjacency built.
        g_max (float): maximum geodesic distance.
        batch_size (int): source vertices per batch (default 256).

    Returns:
        tg: the same object, with neighbor_offsets/indices/dists populated.
    """
    V = tg.num_vertices
    device = tg.device
    max_iters = _estimate_max_iters(tg.edge_dist, V, g_max)
    batch_size = _auto_batch_size(V, tg.edge_src.shape[0], batch_size, device)

    all_counts = []
    all_indices = []
    all_dists = []

    num_batches = (V + batch_size - 1) // batch_size
    for b_idx, start in enumerate(range(0, V, batch_size)):
        end = min(start + batch_size, V)
        sources = torch.arange(start, end, device=device)
        batch_dist = _bellman_ford_batch(
            tg.edge_src, tg.edge_dst, tg.edge_dist,
            V, sources, g_max=g_max, max_iters=max_iters)

        # For each source in this batch, extract neighbors
        mask = (batch_dist < g_max) & (batch_dist > 0)
        counts = mask.sum(dim=1)           # (batch,)
        rows, cols = mask.nonzero(as_tuple=True)
        dists_flat = batch_dist[rows, cols]

        # Move to CPU immediately to free GPU memory
        all_counts.append(counts.cpu())
        all_indices.append(cols.cpu())
        all_dists.append(dists_flat.cpu())

        # Aggressively free GPU memory
        del batch_dist, mask, counts, rows, cols, dists_flat, sources
        if device != 'cpu' and hasattr(torch, 'mps') and device == 'mps':
            torch.mps.empty_cache()
        elif device != 'cpu' and torch.cuda.is_available():
            torch.cuda.empty_cache()

        if (b_idx + 1) % 10 == 0 or (b_idx + 1) == num_batches:
            print(f"  batch {b_idx+1}/{num_batches} "
                  f"({end}/{V} vertices done)")

    # Concatenate and build offsets
    counts_cat = torch.cat(all_counts)
    offsets = torch.zeros(V + 1, dtype=torch.long)
    torch.cumsum(counts_cat, dim=0, out=offsets[1:])

    indices_cat = torch.cat(all_indices)
    dists_cat = torch.cat(all_dists)

    # Move to device
    tg.neighbor_offsets = offsets.to(device)
    tg.neighbor_indices = indices_cat.to(device)
    tg.neighbor_dists = dists_cat.to(device)

    total_pairs = indices_cat.shape[0]
    avg_neighbors = total_pairs / V if V > 0 else 0
    print(f"Computed sparse geodesic neighbors: {total_pairs} pairs, "
          f"avg {avg_neighbors:.1f} neighbors/vertex (g_max={g_max:.4f})")
    return tg


def find_geodesic_neighbors(tg, vertex_idx, g_max, dist_row=None):
    """
    Find all vertices within geodesic distance *g_max* of a given vertex.

    This mirrors the original pycurv
    ``SegmentationGraph.find_geodesic_neighbors`` but returns GPU tensors
    instead of a Python dict.

    Works with three backends (checked in order):
    1. Pre-supplied ``dist_row`` tensor
    2. Pre-computed ``tg.dist_matrix``
    3. Pre-computed sparse neighbors (``tg.neighbor_offsets``)
    4. On-the-fly single-source Bellman-Ford

    Args:
        tg (TriangleGraphGPU): populated triangle graph.
        vertex_idx (int): index of the query vertex.
        g_max (float): maximum geodesic distance.
        dist_row (Tensor [V] or None): pre-computed distance row for this
            vertex (e.g. from ``tg.dist_matrix[vertex_idx]``).  If ``None``
            and ``tg.dist_matrix`` exists it will be read from there;
            otherwise a single-source Bellman-Ford is run on the fly.

    Returns:
        neighbor_indices (Tensor [K], long): indices of neighbors within
            *g_max* (excluding the vertex itself).
        neighbor_dists (Tensor [K], float32): corresponding geodesic
            distances.
    """
    # 1. Explicit dist_row
    if dist_row is not None:
        mask = (dist_row < g_max) & (dist_row > 0)
        neighbor_indices = mask.nonzero(as_tuple=False).squeeze(1)
        return neighbor_indices, dist_row[neighbor_indices]

    # 2. Full distance matrix
    if tg.dist_matrix is not None:
        dists = tg.dist_matrix[vertex_idx]
        mask = (dists < g_max) & (dists > 0)
        neighbor_indices = mask.nonzero(as_tuple=False).squeeze(1)
        return neighbor_indices, dists[neighbor_indices]

    # 3. Sparse neighbor data
    if hasattr(tg, 'neighbor_offsets') and tg.neighbor_offsets is not None:
        start = tg.neighbor_offsets[vertex_idx].item()
        end = tg.neighbor_offsets[vertex_idx + 1].item()
        return (tg.neighbor_indices[start:end],
                tg.neighbor_dists[start:end])

    # 4. On-the-fly single-source Bellman-Ford
    src = torch.tensor([vertex_idx], device=tg.device)
    max_iters = _estimate_max_iters(tg.edge_dist, tg.num_vertices, g_max)
    dists = _bellman_ford_batch(
        tg.edge_src, tg.edge_dst, tg.edge_dist,
        tg.num_vertices, src, g_max=g_max,
        max_iters=max_iters).squeeze(0)

    mask = (dists < g_max) & (dists > 0)
    neighbor_indices = mask.nonzero(as_tuple=False).squeeze(1)
    return neighbor_indices, dists[neighbor_indices]


def find_geodesic_neighbors_batch(tg, g_max):
    """
    For **every** vertex in the graph, find all neighbors within *g_max*.

    Returns results in a CSR-like (compressed sparse row) format so that
    downstream GPU kernels can process all vertices in parallel without
    ragged Python lists.

    If sparse neighbors were pre-computed via
    ``compute_geodesic_neighbors_sparse``, returns those directly.
    Otherwise requires ``tg.dist_matrix`` to be populated.

    Args:
        tg (TriangleGraphGPU): triangle graph.
        g_max (float): maximum geodesic distance.

    Returns:
        offsets (Tensor [V+1], long):  ``offsets[i]`` is the start position
            in *indices* / *dists* for vertex ``i``.  Vertex ``i`` has
            ``offsets[i+1] - offsets[i]`` neighbors.
        indices (Tensor [total_neighbors], long): neighbor vertex indices,
            concatenated for all vertices.
        dists (Tensor [total_neighbors], float32): corresponding geodesic
            distances.
    """
    # Return pre-computed sparse data if available
    if hasattr(tg, 'neighbor_offsets') and tg.neighbor_offsets is not None:
        return tg.neighbor_offsets, tg.neighbor_indices, tg.neighbor_dists

    # Fall back to dense matrix
    D = tg.dist_matrix                      # (V, V)
    mask = (D < g_max) & (D > 0)            # exclude self and clamped vertices

    counts = mask.sum(dim=1)
    offsets = torch.zeros(tg.num_vertices + 1, dtype=torch.long,
                          device=tg.device)
    torch.cumsum(counts, dim=0, out=offsets[1:])

    rows, cols = mask.nonzero(as_tuple=True)
    dists_flat = D[rows, cols]

    return offsets, cols, dists_flat


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _estimate_max_iters(edge_dist, num_vertices, g_max):
    """
    Estimate the number of Bellman-Ford iterations needed.

    With a g_max cutoff the wavefront can travel at most
    ``g_max / min_edge_dist`` hops, which is typically much smaller than V.
    """
    if g_max is not None and edge_dist.numel() > 0:
        min_edge = edge_dist.min().item()
        if min_edge > 0:
            # Add a small margin for safety
            return min(int(math.ceil(g_max / min_edge)) + 2, num_vertices)
    return num_vertices


def _auto_batch_size(num_vertices, num_edges, requested_batch, device):
    """
    Reduce batch size if the requested size would use too much memory.

    Peak memory per batch ≈ batch * (V + E) * 4 bytes  (dist + proposal).
    """
    bytes_per_source = (num_vertices + num_edges) * 4
    # Target: use at most 2 GB per batch
    max_bytes = 2e9
    safe_batch = max(1, int(max_bytes / bytes_per_source))
    chosen = min(requested_batch, safe_batch)
    if chosen < requested_batch:
        print(f"Auto-reduced batch_size {requested_batch} → {chosen} "
              f"to fit memory")
    return chosen
