#!/usr/bin/env python
"""
GPU curvature estimation pipeline (per-triangle, matching CPU pycurv TriangleGraph).

Usage:
    python run_gpu.py surface.vtp --radius-hit 10
    python run_gpu.py surface.vtp --config config.yml
"""

import argparse
import time
from pathlib import Path

import pandas as pd
import torch
import yaml

from core import (TriangleGraphGPU, build_from_vtp, build_adjacency,
                  compute_edge_distances, clean_mesh, find_border_triangles,
                  run_voting, save_vtp, build_csr)


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


def find_triangles_near_border(tg, distance):
    """
    BFS from border triangles on the triangle adjacency graph.
    Returns bool mask [T] of triangles to KEEP (farther than `distance` from border).
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


def _format_rh(rh):
    """Format radius_hit: use int if whole number."""
    return str(int(rh)) if rh == int(rh) else str(rh)


def main():
    parser = argparse.ArgumentParser(description='GPU curvature estimation')
    parser.add_argument('vtp_file', help='Input .vtp mesh file')
    parser.add_argument('--radius-hit', type=float, default=10.0)
    parser.add_argument('--pixel-size', type=float, default=1.0)
    parser.add_argument('--min-component', type=int, default=30)
    parser.add_argument('--exclude-borders', type=int, default=0)
    parser.add_argument('--batch-size', type=int, default=1024)
    parser.add_argument('--no-clean', action='store_true')
    parser.add_argument('--no-vtp', action='store_true', help='Skip VTP output')
    parser.add_argument('--no-cache-sssp', action='store_true',
                        help='Disable SSSP caching between passes (saves ~3.5GB RAM)')
    parser.add_argument('--config', type=str, default=None)
    parser.add_argument('--device', type=str, default=None)
    args = parser.parse_args()

    if args.config:
        with open(args.config) as f:
            cfg = yaml.safe_load(f)
        cm = cfg.get('curvature_measurements', {})
        args.radius_hit = cm.get('radius_hit', args.radius_hit)
        args.pixel_size = cm.get('pixel_size', args.pixel_size)
        args.min_component = cm.get('min_component', args.min_component)
        args.exclude_borders = cm.get('exclude_borders', args.exclude_borders)

    if args.device:
        device = args.device
    elif torch.cuda.is_available():
        device = 'cuda'
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'
    print(f"Device: {device}")

    vtp_path = Path(args.vtp_file)
    basename = vtp_path.stem
    output_dir = vtp_path.parent
    rh_str = _format_rh(args.radius_hit)

    t_total = time.time()

    # 1. Load mesh
    tg = TriangleGraphGPU(device=device)
    build_from_vtp(str(vtp_path), tg)
    build_adjacency(tg)
    compute_edge_distances(tg)
    build_csr(tg)

    # 2. Preprocess
    if not args.no_clean:
        clean_mesh(tg, pixel_size=args.pixel_size,
                   min_component=args.min_component)

    # 3. Run voting (per-triangle on triangle adjacency graph)
    run_voting(tg, args.radius_hit, args.batch_size,
               cache_sssp=not args.no_cache_sssp)

    # 4. Write VTP output
    if not args.no_vtp:
        vtp_out = output_dir / f"{basename}.AVV_rh{rh_str}.vtp"
        save_vtp(tg, str(vtp_out))

    # 5. Extract curvatures (CSV)
    csv_path = output_dir / f"{basename}.AVV_rh{rh_str}.csv"
    extract_curvatures(tg, str(csv_path))

    # Border-excluded variants
    for dist in range(1, args.exclude_borders + 1):
        keep_mask = find_triangles_near_border(tg, dist)
        csv_eb = output_dir / f"{basename}.AVV_rh{rh_str}_excluding{dist}borders.csv"
        extract_curvatures(tg, str(csv_eb), mask=keep_mask)

    total_time = time.time() - t_total
    minutes, seconds = divmod(total_time, 60)
    print(f"\nTotal time: {int(minutes)} min {seconds:.1f} s")

    rt_path = output_dir / f"{basename}_runtimes.csv"
    pd.DataFrame({
        'num_triangles': [tg.num_triangles],
        'radius_hit': [args.radius_hit],
        'total_seconds': [total_time],
    }).to_csv(str(rt_path), index=False)
    print(f"Wrote runtimes to {rt_path}")


if __name__ == '__main__':
    main()
