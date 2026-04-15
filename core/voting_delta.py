"""
Voting using delta-stepping SSSP. Thin wrapper — all voting logic lives in voting.py,
this just swaps the geodesic backend.
"""

from .voting import (normal_vector_voting as _nvv, curvature_voting as _cv,
                     _spatial_sort, _auto_voting_batch, _fix_normal_orientation,
                     _batched_eigh)
from .geodesic_delta import sssp_triangle_batch, get_free_memory
import torch
import math
import time
import numpy as np


def normal_vector_voting(tg, g_max, batch_size=256, spatial_order=None):
    T = tg.num_triangles
    device = tg.device
    sigma = g_max / 3.0
    a_max = tg.max_triangle_area

    V_matrices = torch.zeros(T, 3, 3, dtype=torch.float32, device=device)
    batch_size = _auto_voting_batch(T, batch_size, device)
    all_sources = spatial_order if spatial_order is not None else torch.arange(T, device=device)

    t0 = time.time()
    num_batches = (T + batch_size - 1) // batch_size

    for b_idx, start in enumerate(range(0, T, batch_size)):
        end = min(start + batch_size, T)
        sources = all_sources[start:end]

        src_local, nbr_idx, g_i = sssp_triangle_batch(tg, sources, g_max)

        if src_local.numel() == 0:
            continue

        src_global = sources[src_local]

        v_pos = tg.centers[src_global]
        c_i = tg.centers[nbr_idx]
        n_c = tg.normals[nbr_idx]

        vc = c_i - v_pos
        vc_len = torch.linalg.norm(vc, dim=1, keepdim=True).clamp(min=1e-12)
        vc_hat = vc / vc_len
        cos_theta = -(n_c * vc).sum(dim=1, keepdim=True) / vc_len
        n_i = n_c + 2 * cos_theta * vc_hat

        a_i = tg.areas[nbr_idx]
        w_i = (a_i / a_max) * torch.exp(-g_i / sigma)

        wn = w_i.unsqueeze(1) * n_i
        V_i = wn.unsqueeze(2) * n_i.unsqueeze(1)

        idx = src_global.view(-1, 1, 1).expand_as(V_i)
        V_matrices.scatter_add_(0, idx, V_i)

        del V_i, wn

        if (b_idx + 1) % max(1, num_batches // 5) == 0 or (b_idx + 1) == num_batches:
            elapsed = time.time() - t0
            print(f"  Pass 1 [delta]: batch {b_idx+1}/{num_batches} [{elapsed:.1f}s]")

    bad = ~torch.isfinite(V_matrices).all(dim=2).all(dim=1)
    if bad.any():
        V_matrices[bad] = 0.0
        print(f"  Warning: zeroed {bad.sum().item()} degenerate V matrices")

    V_matrices = (V_matrices + V_matrices.transpose(1, 2)) / 2
    eigenvalues, eigenvectors = _batched_eigh(V_matrices, device)

    n_v = eigenvectors[:, :, 2].clone().to(torch.float32)
    tg.orientation_class = torch.ones(T, dtype=torch.long, device=device)

    cos_orig = (n_v * tg.normals).sum(dim=1)
    n_v[cos_orig < 0] *= -1

    n_v[bad] = tg.normals[bad]
    _fix_normal_orientation(tg, n_v)
    tg.n_v = n_v

    print(f"Pass 1 [delta] done: {T} triangles [{time.time()-t0:.1f}s]")
    return tg


def curvature_voting(tg, g_max, batch_size=256, spatial_order=None):
    T = tg.num_triangles
    device = tg.device
    sigma = g_max / 3.0
    a_max = tg.max_triangle_area
    pi = math.pi

    B_matrices = torch.zeros(T, 3, 3, dtype=torch.float64, device=device)
    weight_sums = torch.zeros(T, dtype=torch.float64, device=device)

    is_surface = tg.orientation_class == 1
    if spatial_order is not None:
        surface_ids = spatial_order[is_surface[spatial_order]]
    else:
        surface_ids = is_surface.nonzero(as_tuple=False).squeeze(1)
    num_surface = surface_ids.shape[0]

    batch_size = _auto_voting_batch(T, batch_size, device)

    t0 = time.time()
    num_batches = (num_surface + batch_size - 1) // batch_size

    for b_idx, start in enumerate(range(0, num_surface, batch_size)):
        end = min(start + batch_size, num_surface)
        sources = surface_ids[start:end]

        src_local, nbr_idx, g_i = sssp_triangle_batch(tg, sources, g_max)

        if src_local.numel() > 0:
            surface_mask = is_surface[nbr_idx]
            src_local = src_local[surface_mask]
            nbr_idx = nbr_idx[surface_mask]
            g_i = g_i[surface_mask]

        if src_local.numel() == 0:
            continue

        src_global = sources[src_local]

        a_i = tg.areas[nbr_idx]
        w_i = (a_i / a_max) * torch.exp(-g_i / sigma)

        v_pos = tg.centers[src_global]
        v_i_pos = tg.centers[nbr_idx]
        n_v = tg.n_v[src_global]

        vv_i = v_i_pos - v_pos
        t_i = vv_i - (n_v * vv_i).sum(dim=1, keepdim=True) * n_v
        t_i_len = torch.linalg.norm(t_i, dim=1, keepdim=True).clamp(min=1e-12)
        t_i = t_i / t_i_len

        p_i = torch.cross(n_v, t_i, dim=1)
        n_v_i = tg.n_v[nbr_idx]
        n_v_i_p = n_v_i - (p_i * n_v_i).sum(dim=1, keepdim=True) * p_i
        n_v_i_p_len = torch.linalg.norm(n_v_i_p, dim=1).clamp(min=1e-12)

        cos_phi = ((n_v * n_v_i_p).sum(dim=1) / n_v_i_p_len).clamp(-1, 1)
        phi = torch.acos(cos_phi)

        vv_i_len = torch.linalg.norm(vv_i, dim=1).clamp(min=1e-12)
        kappa_i = (2 * torch.cos((pi - phi) / 2) / vv_i_len).abs()

        kappa_sign = -torch.sign((t_i * n_v_i_p).sum(dim=1))
        kappa_i = kappa_i * kappa_sign

        wk = (w_i * kappa_i).unsqueeze(1)
        B_i = (wk * t_i).unsqueeze(2) * t_i.unsqueeze(1)

        idx = src_global.view(-1, 1, 1).expand_as(B_i)
        B_matrices.scatter_add_(0, idx, B_i.to(torch.float64))
        weight_sums.scatter_add_(0, src_global, w_i.to(torch.float64))

        del B_i

        if (b_idx + 1) % max(1, num_batches // 5) == 0 or (b_idx + 1) == num_batches:
            elapsed = time.time() - t0
            print(f"  Pass 2 [delta]: batch {b_idx+1}/{num_batches} [{elapsed:.1f}s]")

    valid = weight_sums > 0
    B_matrices[valid] /= weight_sums[valid].unsqueeze(1).unsqueeze(2)

    bad = ~torch.isfinite(B_matrices).all(dim=2).all(dim=1)
    if bad.any():
        B_matrices[bad] = 0.0
        print(f"  Warning: zeroed {bad.sum().item()} degenerate B matrices")

    B_matrices = (B_matrices + B_matrices.transpose(1, 2)) / 2
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

    print(f"Pass 2 [delta] done: curvature for {valid.sum().item()}/{T} triangles "
          f"[{time.time()-t0:.1f}s]")
    return tg


def run_voting(tg, radius_hit, batch_size=256):
    g_max = math.pi * radius_hit / 2.0
    print(f"\nVoting [delta-stepping]: radius_hit={radius_hit}, g_max={g_max:.4f}")
    print(f"  {tg.num_triangles} triangles")

    spatial_order = _spatial_sort(tg)
    normal_vector_voting(tg, g_max, batch_size, spatial_order)
    curvature_voting(tg, g_max, batch_size, spatial_order)
    return tg
