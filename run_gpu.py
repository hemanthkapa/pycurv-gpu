#!/usr/bin/env python
"""
GPU curvature estimation pipeline (per-triangle, matching CPU pycurv TriangleGraph).

This is a thin CLI wrapper around core.api.run_pipeline(). Import
`core.api.run_pipeline` directly for programmatic / in-process use.

Usage:
    python run_gpu.py surface.vtp --radius-hit 10
    python run_gpu.py surface.vtp --config config.yml
"""

import argparse

import yaml

from core.api import run_pipeline


def main():
    parser = argparse.ArgumentParser(description='GPU curvature estimation')
    parser.add_argument('vtp_file', help='Input .vtp mesh file')
    parser.add_argument('--radius-hit', type=float, default=10.0)
    parser.add_argument('--pixel-size', type=float, default=1.0)
    parser.add_argument('--min-component', type=int, default=30)
    parser.add_argument('--exclude-borders', type=int, default=0)
    parser.add_argument('--remove-wrong-borders', action='store_true',
                        help='Eat back the surface before calculations '
                             '(for segmentation-derived meshes; leave off '
                             'for screened-Poisson meshes). Matches CPU '
                             'pycurv remove_wrong_borders=True.')
    parser.add_argument('--epsilon', type=float, default=0.0,
                        help='Orientation classification parameter (Page et '
                             'al. 2002). Default 0 classifies all triangles '
                             'as surface patches, matching surface_morphometrics.')
    parser.add_argument('--eta', type=float, default=0.0,
                        help='Orientation classification parameter (Page et al. 2002).')
    parser.add_argument('--batch-size', type=int, default=1024)
    parser.add_argument('--sssp', choices=['spfa', 'delta'], default='spfa',
                        help='SSSP solver: spfa (frontier) or delta (delta-stepping)')
    parser.add_argument('--no-clean', action='store_true')
    parser.add_argument('--no-vtp', action='store_true', help='Skip VTP output')
    parser.add_argument('--gt', action='store_true',
                        help='Also write graph-tool .gt output (needs graph-tool)')
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

    run_pipeline(
        args.vtp_file,
        radius_hit=args.radius_hit,
        pixel_size=args.pixel_size,
        min_component=args.min_component,
        exclude_borders=args.exclude_borders,
        remove_wrong_borders=args.remove_wrong_borders,
        epsilon=args.epsilon,
        eta=args.eta,
        batch_size=args.batch_size,
        sssp=args.sssp,
        device=args.device,
        no_clean=args.no_clean,
        write_vtp=not args.no_vtp,
        write_gt=args.gt,
        no_cache_sssp=args.no_cache_sssp,
    )


if __name__ == '__main__':
    main()
