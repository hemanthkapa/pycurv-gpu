"""
Tensor Voting on the triangle graph, matching CPU pycurv's TriangleGraph.

Both passes operate on the triangle dual graph: each node is a triangle
centroid, edges connect adjacent triangles, geodesic distances are between
centroids.

Pass 1 (Normal estimation):
  Each triangle v collects normal votes from neighboring triangles c_i
  within g_max. Accumulates weighted covariance matrix V_v.

Pass 2 (Curvature estimation — AVV, area-weighted):
  Each surface triangle v collects curvature votes from neighboring
  surface triangles c_i within g_max. Tong-Tang normal curvature formula.

All outputs are per-triangle [T].

Reference: Page et al. 2002, Tong & Tang 2005, CPU pycurv surface_graphs.py
"""

import torch
import math
import time

from .geodesic import sssp_triangle_batch, get_free_memory


# ---------------------------------------------------------------------------
# Pass 1: Normal Vector Voting (per-triangle)
# ---------------------------------------------------------------------------

def normal_vector_voting(tg, g_max, batch_size=256):
    T = tg.num_triangles
    device = tg.device
    sigma = g_max / 3.0
    a_max = tg.max_triangle_area

    V_matrices = torch.zeros(T, 3, 3, dtype=torch.float32, device=device)

    batch_size = _auto_voting_batch(T, batch_size, device)

    t0 = time.time()
    num_batches = (T + batch_size - 1) // batch_size

    for b_idx, start in enumerate(range(0, T, batch_size)):
        end = min(start + batch_size, T)
        sources = torch.arange(start, end, device=device)

        # SSSP on triangle adjacency graph — returns sparse neighbors
        src_local, nbr_idx, g_i = sssp_triangle_batch(tg, sources, g_max)

        if src_local.numel() == 0:
            continue

        src_global = sources[src_local]

        # Normal vote: n_i = n_c + 2*cos(theta)*vc_hat  (Page et al. Eq. 6)
        v_pos = tg.centers[src_global]       # [N, 3] source centroid
        c_i = tg.centers[nbr_idx]            # [N, 3] neighbor centroid
        n_c = tg.normals[nbr_idx]            # [N, 3] neighbor triangle normal

        vc = c_i - v_pos
        vc_len = torch.linalg.norm(vc, dim=1, keepdim=True).clamp(min=1e-12)
        vc_hat = vc / vc_len
        cos_theta = -(n_c * vc).sum(dim=1, keepdim=True) / vc_len
        n_i = n_c + 2 * cos_theta * vc_hat

        # Weight: (area_i / max_area) * exp(-g_i / sigma)
        a_i = tg.areas[nbr_idx]
        w_i = (a_i / a_max) * torch.exp(-g_i / sigma)

        # Weighted outer product V_i = w_i * n_i ⊗ n_i
        wn = w_i.unsqueeze(1) * n_i
        V_i = wn.unsqueeze(2) * n_i.unsqueeze(1)  # [N, 3, 3]

        idx = src_global.view(-1, 1, 1).expand_as(V_i)
        V_matrices.scatter_add_(0, idx, V_i)

        del V_i, wn

        if (b_idx + 1) % max(1, num_batches // 5) == 0 or (b_idx + 1) == num_batches:
            elapsed = time.time() - t0
            print(f"  Pass 1: batch {b_idx+1}/{num_batches} [{elapsed:.1f}s]")

    # Clean NaN/Inf before eigendecomposition (degenerate triangles)
    bad = ~torch.isfinite(V_matrices).all(dim=2).all(dim=1)
    if bad.any():
        V_matrices[bad] = 0.0
        print(f"  Warning: zeroed {bad.sum().item()} degenerate V matrices")

    # Force symmetry (numerical safety)
    V_matrices = (V_matrices + V_matrices.transpose(1, 2)) / 2

    # Eigendecompose in chunks (cusolver can fail on very large batches)
    eigenvalues, eigenvectors = _batched_eigh(V_matrices, device)
    # eigh ascending: [:, 2] is largest eigenvalue

    # All surface (epsilon=0, eta=0 default)
    n_v = eigenvectors[:, :, 2].clone().to(torch.float32)
    tg.orientation_class = torch.ones(T, dtype=torch.long, device=device)

    # Fix orientation: align with original triangle normals
    cos_orig = (n_v * tg.normals).sum(dim=1)
    n_v[cos_orig < 0] *= -1

    # For degenerate triangles, fall back to original normal
    n_v[bad] = tg.normals[bad]

    # Propagate orientation consistency via triangle adjacency
    _fix_normal_orientation(tg, n_v)

    tg.n_v = n_v

    print(f"Pass 1 done: {T} triangles [{time.time()-t0:.1f}s]")
    return tg


# ---------------------------------------------------------------------------
# Pass 2: Curvature Voting (AVV — area-weighted, per-triangle)
# ---------------------------------------------------------------------------

def curvature_voting(tg, g_max, batch_size=256):
    T = tg.num_triangles
    device = tg.device
    sigma = g_max / 3.0
    a_max = tg.max_triangle_area
    pi = math.pi

    B_matrices = torch.zeros(T, 3, 3, dtype=torch.float64, device=device)
    weight_sums = torch.zeros(T, dtype=torch.float64, device=device)

    is_surface = tg.orientation_class == 1
    surface_ids = is_surface.nonzero(as_tuple=False).squeeze(1)
    num_surface = surface_ids.shape[0]

    batch_size = _auto_voting_batch(T, batch_size, device)

    t0 = time.time()
    num_batches = (num_surface + batch_size - 1) // batch_size

    for b_idx, start in enumerate(range(0, num_surface, batch_size)):
        end = min(start + batch_size, num_surface)
        sources = surface_ids[start:end]

        # SSSP on triangle adjacency graph — returns sparse neighbors
        src_local, nbr_idx, g_i = sssp_triangle_batch(tg, sources, g_max)

        # Filter to surface-only neighbors
        if src_local.numel() > 0:
            surface_mask = is_surface[nbr_idx]
            src_local = src_local[surface_mask]
            nbr_idx = nbr_idx[surface_mask]
            g_i = g_i[surface_mask]

        if src_local.numel() == 0:
            continue

        src_global = sources[src_local]

        # Weight: (area_i / max_area) * exp(-g_i / sigma)  [AVV]
        a_i = tg.areas[nbr_idx]
        w_i = (a_i / a_max) * torch.exp(-g_i / sigma)

        # Tangent direction: project v->v_i onto tangent plane of v
        v_pos = tg.centers[src_global]
        v_i_pos = tg.centers[nbr_idx]
        n_v = tg.n_v[src_global]

        vv_i = v_i_pos - v_pos
        t_i = vv_i - (n_v * vv_i).sum(dim=1, keepdim=True) * n_v
        t_i_len = torch.linalg.norm(t_i, dim=1, keepdim=True).clamp(min=1e-12)
        t_i = t_i / t_i_len

        # Normal curvature: Tong-Tang formula
        p_i = torch.cross(n_v, t_i, dim=1)
        n_v_i = tg.n_v[nbr_idx]
        n_v_i_p = n_v_i - (p_i * n_v_i).sum(dim=1, keepdim=True) * p_i
        n_v_i_p_len = torch.linalg.norm(n_v_i_p, dim=1).clamp(min=1e-12)

        cos_phi = ((n_v * n_v_i_p).sum(dim=1) / n_v_i_p_len).clamp(-1, 1)
        phi = torch.acos(cos_phi)

        vv_i_len = torch.linalg.norm(vv_i, dim=1).clamp(min=1e-12)
        kappa_i = (2 * torch.cos((pi - phi) / 2) / vv_i_len).abs()

        # Sign: opposite to dot(t_i, n_v_i_p)
        kappa_sign = -torch.sign((t_i * n_v_i_p).sum(dim=1))
        kappa_i = kappa_i * kappa_sign

        # Accumulate B_v = sum(w_i * kappa_i * outer(t_i, t_i))
        wk = (w_i * kappa_i).unsqueeze(1)
        B_i = (wk * t_i).unsqueeze(2) * t_i.unsqueeze(1)

        idx = src_global.view(-1, 1, 1).expand_as(B_i)
        B_matrices.scatter_add_(0, idx, B_i.to(torch.float64))
        weight_sums.scatter_add_(0, src_global, w_i.to(torch.float64))

        del B_i

        if (b_idx + 1) % max(1, num_batches // 5) == 0 or (b_idx + 1) == num_batches:
            elapsed = time.time() - t0
            print(f"  Pass 2: batch {b_idx+1}/{num_batches} [{elapsed:.1f}s]")

    # Normalize: B_v /= sum_w_i
    valid = weight_sums > 0
    B_matrices[valid] /= weight_sums[valid].unsqueeze(1).unsqueeze(2)

    # Clean NaN/Inf before eigendecomposition
    bad = ~torch.isfinite(B_matrices).all(dim=2).all(dim=1)
    if bad.any():
        B_matrices[bad] = 0.0
        print(f"  Warning: zeroed {bad.sum().item()} degenerate B matrices")

    B_matrices = (B_matrices + B_matrices.transpose(1, 2)) / 2

    # Eigendecompose — identify which eigenvector aligns with n_v
    eigenvalues, eigenvectors = _batched_eigh(B_matrices, device)

    n_v_all = tg.n_v
    dots = torch.zeros(T, 3, device=device)
    for k in range(3):
        dots[:, k] = (eigenvectors[:, :, k] * n_v_all).sum(dim=1).abs()

    normal_idx = dots.argmax(dim=1)

    all_idx = torch.arange(3, device=device).unsqueeze(0).expand(T, -1)
    mask = all_idx != normal_idx.unsqueeze(1)
    curv_idx = all_idx[mask].reshape(T, 2)

    arange_T = torch.arange(T, device=device)
    ev_0 = eigenvalues[arange_T, curv_idx[:, 0]]
    ev_1 = eigenvalues[arange_T, curv_idx[:, 1]]

    b_1 = torch.max(ev_0, ev_1)
    b_2 = torch.min(ev_0, ev_1)

    idx_b1 = torch.where(ev_0 >= ev_1, curv_idx[:, 0], curv_idx[:, 1])
    idx_b2 = torch.where(ev_0 >= ev_1, curv_idx[:, 1], curv_idx[:, 0])
    tg.t_1 = eigenvectors[arange_T, :, idx_b1]
    tg.t_2 = eigenvectors[arange_T, :, idx_b2]

    kappa_1 = 3 * b_1 - b_2
    kappa_2 = 3 * b_2 - b_1

    swap = kappa_1 < kappa_2
    if swap.any():
        k1_old = kappa_1.clone()
        kappa_1[swap] = kappa_2[swap]
        kappa_2[swap] = k1_old[swap]
        t1_old = tg.t_1.clone()
        tg.t_1[swap] = tg.t_2[swap]
        tg.t_2[swap] = t1_old[swap]

    tg.kappa_1 = kappa_1
    tg.kappa_2 = kappa_2
    tg.gauss_curvature = kappa_1 * kappa_2
    tg.mean_curvature = (kappa_1 + kappa_2) / 2
    tg.curvedness = torch.sqrt((kappa_1**2 + kappa_2**2) / 2)

    both_zero = (kappa_1 == 0) & (kappa_2 == 0)
    shape_index = (2 / pi) * torch.atan2(
        kappa_1 + kappa_2, kappa_1 - kappa_2)
    shape_index[both_zero] = 0
    tg.shape_index = shape_index

    print(f"Pass 2 done: curvature for {valid.sum().item()}/{T} triangles "
          f"[{time.time()-t0:.1f}s]")
    return tg


# ---------------------------------------------------------------------------
# Pipeline entry point
# ---------------------------------------------------------------------------

def run_voting(tg, radius_hit, batch_size=256):
    g_max = math.pi * radius_hit / 2.0
    print(f"\nVoting: radius_hit={radius_hit}, g_max={g_max:.4f}")
    print(f"  {tg.num_triangles} triangles")

    normal_vector_voting(tg, g_max, batch_size)
    curvature_voting(tg, g_max, batch_size)
    return tg


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fix_normal_orientation(tg, n_v):
    """Propagate normal orientation consistency via triangle adjacency."""
    T = tg.num_triangles
    device = tg.device

    nbr_sum = torch.zeros(T, 3, dtype=torch.float32, device=device)
    nbr_sum.scatter_add_(
        0, tg.edge_src.unsqueeze(1).expand(-1, 3), n_v[tg.edge_dst])

    norms = torch.linalg.norm(nbr_sum, dim=1, keepdim=True).clamp(min=1e-12)
    avg_n = nbr_sum / norms

    flip = (n_v * avg_n).sum(dim=1) < 0
    n_v[flip] *= -1


def _auto_voting_batch(num_triangles, requested, device):
    # With sparse SSSP, the subgraph is the UNION of reachable nodes from
    # all B sources. More sources = larger union. At B=256, subgraph ~2% of T.
    # Union grows sublinearly with B (sources overlap). Estimate:
    #   L ~ min(T, per_source_reach * B^0.6)
    # Conservative: use full T for memory estimate (safe upper bound).
    # The real gain from sparse SSSP is avoiding the clone, not the allocation.
    bytes_per_source = num_triangles * 9 + 4000
    free_mem = get_free_memory(device)
    max_bytes = free_mem * 0.40
    safe = max(1, int(max_bytes / bytes_per_source))
    chosen = min(requested, safe)
    if chosen < requested:
        print(f"Auto-reduced voting batch {requested} -> {chosen}")
    return chosen


def _batched_eigh(matrices, device, chunk_size=16384):
    """
    Chunked eigendecomposition to avoid cusolver crashes on large batches.
    """
    N = matrices.shape[0]
    if N <= chunk_size:
        return torch.linalg.eigh(matrices)

    eigenvalues = torch.empty(N, 3, dtype=matrices.dtype, device=device)
    eigenvectors = torch.empty(N, 3, 3, dtype=matrices.dtype, device=device)

    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        ev, evec = torch.linalg.eigh(matrices[start:end])
        eigenvalues[start:end] = ev
        eigenvectors[start:end] = evec

    return eigenvalues, eigenvectors
