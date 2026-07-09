"""
High-level, importable pipeline API for pycurv-gpu.

`run_pipeline()` mirrors CPU pycurv's function signature for
`normals_directions_and_curvature_estimation` / surface_morphometrics'
`curvature.run_pycurv(filename, folder, ...)`, so callers can drive the full
GPU curvature pipeline from Python instead of only via the run_gpu.py CLI.
See run_gpu.py for the thin CLI wrapper around this module.
"""
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .triangle_graph_gpu import TriangleGraphGPU
from .mesh_io import build_from_vtp, build_adjacency, compute_edge_distances, save_vtp, save_gt
from .preprocessing import clean_mesh, find_border_triangles
from .geodesic import build_csr
from .voting import run_voting


def shape_index_classify(si):
    """Classify shape index into category label, matching CPU pycurv."""
    if si < -1:
        return None
    elif si < -7/8:
        return 'Spherical cup'
    elif si < -5/8:
        return 'Trough'
    elif si < -3/8:
        return 'Rut'
    elif si < -1/8:
        return 'Saddle rut'
    elif si < 1/8:
        return 'Saddle'
    elif si < 3/8:
        return 'Saddle ridge'
    elif si < 5/8:
        return 'Ridge'
    elif si < 7/8:
        return 'Dome'
    elif si <= 1:
        return 'Spherical cap'
    return None


def pick_device(requested=None):
    if requested:
        return requested
    if torch.cuda.is_available():
        return 'cuda'
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'


def format_radius_hit(rh):
    """Format radius_hit: use int if whole number (matches CPU pycurv filenames)."""
    return str(int(rh)) if rh == int(rh) else str(rh)


def find_triangles_near_border(tg, distance):
    """
    BFS from border triangles on the triangle adjacency graph.
    Returns bool mask [T] of triangles to KEEP (farther than `distance` from border).
    Used for the exclude_borders CSV variants (post-hoc, does not modify tg).
    """
    T = tg.num_triangles
    device = tg.device

    is_border = find_border_triangles(tg)
    border_ids = is_border.nonzero(as_tuple=False).squeeze(1)

    if border_ids.numel() == 0:
        return torch.ones(T, dtype=torch.bool, device=device)

    dist = torch.full((T,), float('inf'), dtype=torch.float32, device=device)
    dist[border_ids] = 0.0

    active = border_ids
    for _ in range(T):
        if active.numel() == 0:
            break
        is_active = torch.zeros(T, dtype=torch.bool, device=device)
        is_active[active] = True
        active_mask = is_active[tg.edge_src]

        src = tg.edge_src[active_mask]
        dst = tg.edge_dst[active_mask]
        w = tg.edge_dist[active_mask]

        proposal = dist[src] + w
        within = proposal <= distance
        if not within.any():
            break

        updated = dist.clone()
        updated.scatter_reduce_(0, dst[within], proposal[within],
                                reduce='amin', include_self=True)

        improved = updated < dist
        dist = updated
        active = improved.nonzero(as_tuple=False).squeeze(1)

    return dist > distance


def extract_curvatures(tg, output_path, mask=None):
    """Write per-triangle curvature CSV matching surface_morphometrics format."""
    kappa_1 = tg.kappa_1.cpu().numpy()
    kappa_2 = tg.kappa_2.cpu().numpy()
    gauss = tg.gauss_curvature.cpu().numpy()
    mean = tg.mean_curvature.cpu().numpy()
    si = tg.shape_index.cpu().numpy()
    curv = tg.curvedness.cpu().numpy()
    areas = tg.areas.cpu().numpy()

    if mask is not None:
        m = mask.cpu().numpy()
        kappa_1 = kappa_1[m]
        kappa_2 = kappa_2[m]
        gauss = gauss[m]
        mean = mean[m]
        si = si[m]
        curv = curv[m]
        areas = areas[m]

    si_class = [shape_index_classify(float(s)) for s in si]

    df = pd.DataFrame({
        'kappa1': kappa_1,
        'kappa2': kappa_2,
        'gauss_curvature': gauss,
        'mean_curvature': mean,
        'shape_index': si,
        'shape_index_class': si_class,
        'curvedness': curv,
        'triangleAreas': areas,
    })
    df.to_csv(output_path, sep=';')
    print(f"Wrote {len(df)} rows to {output_path}")


# ---------------------------------------------------------------------------
# NVV (Pass 1) caching -- mirrors CPU pycurv's `.NVV_rh{rh}.gt` skip-if-exists
# semantics, but uses a dependency-free numpy format so it works without
# graph-tool. refine_mesh.py-style callers that want a fresh run each time
# (e.g. after displacing vertices) should delete this file alongside the
# other stale caches -- see stale_exts in surface_morphometrics/refine_mesh.py.
# ---------------------------------------------------------------------------

def nvv_cache_path(output_dir, basename, rh_str):
    return Path(output_dir) / f"{basename}.NVV_rh{rh_str}.gpu_normals.npz"


def _centers_checksum(tg):
    """Cheap order-sensitive fingerprint to detect a changed/rebuilt mesh."""
    c = tg.centers.detach().cpu().numpy()
    return float(c.sum()) + float(c[0].sum()) + float(c[-1].sum())


def save_nvv_cache(tg, path):
    np.savez(
        path,
        n_v=tg.n_v.cpu().numpy(),
        orientation_class=tg.orientation_class.cpu().numpy(),
        num_triangles=np.array(tg.num_triangles),
        centers_checksum=np.array(_centers_checksum(tg)),
    )
    print(f"Cached Pass 1 normals -> {path}")


def load_nvv_cache(tg, path):
    """Try to load cached Pass 1 output into tg. Returns True on success."""
    try:
        data = np.load(path)
    except (OSError, ValueError):
        return False

    if int(data['num_triangles']) != tg.num_triangles:
        return False
    if abs(float(data['centers_checksum']) - _centers_checksum(tg)) > 1e-3:
        return False

    tg.n_v = torch.tensor(data['n_v'], dtype=torch.float32, device=tg.device)
    tg.orientation_class = torch.tensor(
        data['orientation_class'], dtype=torch.long, device=tg.device)
    return True


# ---------------------------------------------------------------------------
# Pipeline entry point
# ---------------------------------------------------------------------------

def run_pipeline(vtp_path, output_dir=None, radius_hit=10.0, pixel_size=1.0,
                 min_component=30, exclude_borders=0, remove_wrong_borders=False,
                 epsilon=0.0, eta=0.0, batch_size=1024, sssp='spfa', device=None,
                 no_clean=False, write_vtp=True, write_gt=False,
                 cache_normals=True, no_cache_sssp=False, cores=None):
    """
    Run the full GPU curvature pipeline on a single `.surface.vtp` mesh.

    Mirrors CPU pycurv's `normals_directions_and_curvature_estimation` /
    surface_morphometrics' `curvature.run_pycurv(filename, folder, ...)`
    parameters so it can be dispatched to programmatically. See
    surface_morphometrics/curvature.py:run_pycurv_gpu for the cross-repo
    (subprocess) integration, since the two pipelines normally run in
    separate conda environments (torch+GPU vs graph-tool+pycurv).

    `cores` is accepted for signature compatibility with CPU pycurv but is
    ignored -- GPU batch parallelism replaces CPU multiprocessing.

    Returns a dict of output paths (vtp/gt/csv/runtimes_csv) and timings.
    """
    device = pick_device(device)
    vtp_path = Path(vtp_path)
    output_dir = Path(output_dir) if output_dir else vtp_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    name = vtp_path.name
    basename = name[:-len('.surface.vtp')] if name.endswith('.surface.vtp') else vtp_path.stem
    rh_str = format_radius_hit(radius_hit)

    print(f"Device: {device}")
    t_total = time.time()

    tg = TriangleGraphGPU(device=device)
    build_from_vtp(str(vtp_path), tg)
    build_adjacency(tg)
    compute_edge_distances(tg)

    if not no_clean:
        clean_mesh(tg, pixel_size=pixel_size, min_component=min_component,
                   remove_wrong_borders=remove_wrong_borders)

    build_csr(tg)

    cache_path = nvv_cache_path(output_dir, basename, rh_str)
    used_cache = False
    if cache_normals and cache_path.exists():
        used_cache = load_nvv_cache(tg, cache_path)
        if used_cache:
            print(f"Loaded cached Pass 1 normals from {cache_path}")

    run_voting(tg, radius_hit, batch_size, cache_sssp=not no_cache_sssp, sssp=sssp,
              epsilon=epsilon, eta=eta, skip_normals=used_cache)

    if cache_normals and not used_cache:
        save_nvv_cache(tg, cache_path)

    outputs = {'device': device, 'num_triangles': tg.num_triangles}

    if write_vtp:
        vtp_out = output_dir / f"{basename}.AVV_rh{rh_str}.vtp"
        save_vtp(tg, str(vtp_out))
        outputs['vtp'] = str(vtp_out)

    if write_gt:
        gt_out = output_dir / f"{basename}.AVV_rh{rh_str}.gt"
        save_gt(tg, str(gt_out))
        outputs['gt'] = str(gt_out)

    csv_path = output_dir / f"{basename}.AVV_rh{rh_str}.csv"
    extract_curvatures(tg, str(csv_path))
    outputs['csv'] = str(csv_path)

    outputs['csv_excluding_borders'] = []
    for dist in range(1, exclude_borders + 1):
        keep_mask = find_triangles_near_border(tg, dist)
        csv_eb = output_dir / f"{basename}.AVV_rh{rh_str}_excluding{dist}borders.csv"
        extract_curvatures(tg, str(csv_eb), mask=keep_mask)
        outputs['csv_excluding_borders'].append(str(csv_eb))

    total_time = time.time() - t_total
    outputs['total_seconds'] = total_time
    minutes, seconds = divmod(total_time, 60)
    print(f"\nTotal time: {int(minutes)} min {seconds:.1f} s")

    rt_path = output_dir / f"{basename}_runtimes.csv"
    pd.DataFrame({
        'num_triangles': [tg.num_triangles],
        'radius_hit': [radius_hit],
        'total_seconds': [total_time],
    }).to_csv(str(rt_path), index=False)
    outputs['runtimes_csv'] = str(rt_path)
    print(f"Wrote runtimes to {rt_path}")

    return outputs
