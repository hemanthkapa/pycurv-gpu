#!/usr/bin/env python
"""
GPU curvature estimation using delta-stepping SSSP.

Drop-in replacement for run_gpu.py — same args, same output format.
Compare output CSVs and runtimes against run_gpu.py to evaluate.

Usage:
    python run_gpu_delta.py surface.vtp --radius-hit 9
"""

import argparse
import time
from pathlib import Path

import pandas as pd
import torch
import yaml

from core import (TriangleGraphGPU, build_from_vtp, build_adjacency,
                  compute_edge_distances, clean_mesh, find_border_triangles,
                  save_vtp)
from core.geodesic_delta import build_csr
from core.voting_delta import run_voting
from run_gpu import extract_curvatures, find_triangles_near_border, shape_index_classify


def _format_rh(rh):
    return str(int(rh)) if rh == int(rh) else str(rh)


def main():
    parser = argparse.ArgumentParser(description='GPU curvature (delta-stepping)')
    parser.add_argument('vtp_file', help='Input .vtp mesh file')
    parser.add_argument('--radius-hit', type=float, default=10.0)
    parser.add_argument('--pixel-size', type=float, default=1.0)
    parser.add_argument('--min-component', type=int, default=30)
    parser.add_argument('--exclude-borders', type=int, default=0)
    parser.add_argument('--batch-size', type=int, default=1024)
    parser.add_argument('--no-clean', action='store_true')
    parser.add_argument('--no-vtp', action='store_true', help='Skip VTP output')
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
    print(f"SSSP backend: delta-stepping")

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

    # 3. Run voting with delta-stepping SSSP
    run_voting(tg, args.radius_hit, args.batch_size)

    # 4. Write VTP output
    if not args.no_vtp:
        vtp_out = output_dir / f"{basename}.AVV_rh{rh_str}_delta.vtp"
        save_vtp(tg, str(vtp_out))

    # 5. Extract curvatures (CSV)
    csv_path = output_dir / f"{basename}.AVV_rh{rh_str}_delta.csv"
    extract_curvatures(tg, str(csv_path))

    for dist in range(1, args.exclude_borders + 1):
        keep_mask = find_triangles_near_border(tg, dist)
        csv_eb = output_dir / f"{basename}.AVV_rh{rh_str}_delta_excluding{dist}borders.csv"
        extract_curvatures(tg, str(csv_eb), mask=keep_mask)

    total_time = time.time() - t_total
    minutes, seconds = divmod(total_time, 60)
    print(f"\nTotal time: {int(minutes)} min {seconds:.1f} s")

    rt_path = output_dir / f"{basename}_runtimes_delta.csv"
    pd.DataFrame({
        'num_triangles': [tg.num_triangles],
        'radius_hit': [args.radius_hit],
        'total_seconds': [total_time],
        'sssp_backend': ['delta-stepping'],
    }).to_csv(str(rt_path), index=False)
    print(f"Wrote runtimes to {rt_path}")


if __name__ == '__main__':
    main()
