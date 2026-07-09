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
# Pipeline entry point
# ---------------------------------------------------------------------------

def run_pipeline(vtp_path, output_dir=None, radius_hit=10.0, pixel_size=1.0,
                 min_component=30, exclude_borders=0, remove_wrong_borders=False,
                 epsilon=0.0, eta=0.0, batch_size=1024, sssp='spfa', device=None,
                 no_clean=False, write_vtp=True, write_gt=False,
                 no_cache_sssp=False, cores=None):
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

    run_voting(tg, radius_hit, batch_size, cache_sssp=not no_cache_sssp, sssp=sssp,
              epsilon=epsilon, eta=eta)

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
