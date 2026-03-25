#!/usr/bin/env python
"""Benchmark geodesic distance: CPU (pycurv/graph_tool) vs GPU (pycurv-gpu).

Isolates just the geodesic computation step for apples-to-apples comparison.
Same CSV format as surface_morphometrics/benchmark.py.

Usage:
    # Single file, both backends:
    python benchmark_geodesic.py surface.vtp

    # From config.yml:
    python benchmark_geodesic.py --config config.yml

    # GPU only (skip CPU):
    python benchmark_geodesic.py --skip-cpu surface.vtp

    # CPU only (skip GPU):
    python benchmark_geodesic.py --skip-gpu surface.vtp

    # Custom radius_hit (default: from config or 10):
    python benchmark_geodesic.py --radius-hit 8 surface.vtp
"""

import argparse
import csv
import math
import time
from pathlib import Path


# ---------------------------------------------------------------------------
# GPU backend
# ---------------------------------------------------------------------------

def _gpu_geodesic(vtp_path, radius_hit, g_max, sparse, batch_size=256):
    import torch
    from core.triangle_graph_gpu import TriangleGraphGPU
    from core.mesh_io import build_from_vtp, build_adjacency, compute_edge_distances
    from core.geodesic import compute_geodesic_distances, compute_geodesic_neighbors_sparse

    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'

    # Load + build graph
    tg = TriangleGraphGPU(device=device)
    t0 = time.perf_counter()
    build_from_vtp(str(vtp_path), tg)
    build_adjacency(tg)
    compute_edge_distances(tg)
    t_load = time.perf_counter() - t0

    num_v = tg.num_vertices
    num_e = tg.edge_src.shape[0]

    # Geodesic
    t0 = time.perf_counter()
    if sparse:
        compute_geodesic_neighbors_sparse(tg, g_max=g_max, batch_size=batch_size)
    else:
        compute_geodesic_distances(tg, g_max=g_max, batch_size=batch_size)
    t_geo = time.perf_counter() - t0

    return {
        "device": device,
        "num_v": num_v,
        "num_e": num_e,
        "t_load": t_load,
        "t_geodesic": t_geo,
    }


# ---------------------------------------------------------------------------
# CPU backend (original pycurv via graph_tool)
# ---------------------------------------------------------------------------

def _cpu_geodesic(vtp_path, radius_hit, g_max):
    from graph_tool.topology import shortest_distance
    from pycurv import TriangleGraph
    from pycurv import pycurv_io as io

    surface = io.load_poly(str(vtp_path))

    # Build triangle graph (same as new_workflow does)
    t0 = time.perf_counter()
    tg = TriangleGraph()
    tg.build_graph_from_vtk_surface(surface, scale=(1, 1, 1))
    t_load = time.perf_counter() - t0

    num_v = tg.graph.num_vertices()
    num_e = tg.graph.num_edges()

    # Full distance map (equivalent to what GPU dense does)
    t0 = time.perf_counter()
    shortest_distance(tg.graph, weights=tg.graph.ep.distance, max_dist=g_max)
    t_geo = time.perf_counter() - t0

    return {
        "device": "cpu(graph_tool)",
        "num_v": num_v,
        "num_e": num_e,
        "t_load": t_load,
        "t_geodesic": t_geo,
    }


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def benchmark_file(vtp_path, radius_hit, g_max, run_cpu, run_gpu, sparse,
                   batch_size=256):
    results = []
    name = Path(vtp_path).name

    if run_cpu:
        try:
            r = _cpu_geodesic(vtp_path, radius_hit, g_max)
            results.append({
                "label": f"cpu:{name}:load",
                "seconds": round(r["t_load"], 2),
                "status": "OK",
                "command": f"graph_tool [{r['num_v']} tri, {r['num_e']} edges]",
            })
            results.append({
                "label": f"cpu:{name}:geodesic",
                "seconds": round(r["t_geodesic"], 2),
                "status": "OK",
                "command": f"shortest_distance(max_dist={g_max:.4f})",
            })
            results.append({
                "label": f"cpu:{name}:total",
                "seconds": round(r["t_load"] + r["t_geodesic"], 2),
                "status": "OK",
                "command": r["device"],
            })
        except Exception as e:
            results.append({
                "label": f"cpu:{name}",
                "seconds": 0,
                "status": f"FAIL({e})",
                "command": "",
            })

    if run_gpu:
        try:
            r = _gpu_geodesic(vtp_path, radius_hit, g_max, sparse, batch_size)
            geo_type = "sparse" if sparse else "dense"
            results.append({
                "label": f"gpu:{name}:load",
                "seconds": round(r["t_load"], 2),
                "status": "OK",
                "command": f"pycurv-gpu [{r['num_v']} tri, {r['num_e']} edges]",
            })
            results.append({
                "label": f"gpu:{name}:geodesic({geo_type})",
                "seconds": round(r["t_geodesic"], 2),
                "status": "OK",
                "command": f"g_max={g_max:.4f} device={r['device']}",
            })
            results.append({
                "label": f"gpu:{name}:total",
                "seconds": round(r["t_load"] + r["t_geodesic"], 2),
                "status": "OK",
                "command": r["device"],
            })
        except Exception as e:
            results.append({
                "label": f"gpu:{name}",
                "seconds": 0,
                "status": f"FAIL({e})",
                "command": "",
            })

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark geodesic: CPU (graph_tool) vs GPU (pycurv-gpu)")
    parser.add_argument("files", nargs="*", help=".vtp files")
    parser.add_argument("--config", help="config.yml (reads work_dir + radius_hit)")
    parser.add_argument("--radius-hit", type=float, default=None)
    parser.add_argument("--sparse", action="store_true",
                        help="GPU: use sparse neighbor computation")
    parser.add_argument("--batch-size", type=int, default=256,
                        help="GPU batch size (default 256, try 2048+ for large VRAM)")
    parser.add_argument("--skip-cpu", action="store_true")
    parser.add_argument("--skip-gpu", action="store_true")
    parser.add_argument("-o", "--output", default="benchmark_results.csv")
    args = parser.parse_args()

    # Resolve files
    vtp_files = []
    radius_hit = args.radius_hit

    if args.config:
        import yaml
        with open(args.config) as f:
            config = yaml.safe_load(f)
        work_dir = Path(config["work_dir"])
        vtp_files = sorted(work_dir.glob("*.surface.vtp"))
        if radius_hit is None:
            radius_hit = config.get("curvature_measurements", {}).get(
                "radius_hit", 10)

    if args.files:
        vtp_files += [Path(f) for f in args.files]

    if not vtp_files:
        print("ERROR: No .vtp files found. Provide files or --config.")
        parser.error("No .vtp files. Provide files or --config.")

    if radius_hit is None:
        radius_hit = 10
    g_max = math.pi * radius_hit / 2

    run_cpu = not args.skip_cpu
    run_gpu = not args.skip_gpu

    print(f"radius_hit={radius_hit}  g_max={g_max:.4f}")
    print(f"Files: {len(vtp_files)}")
    print(f"Backends: {'CPU ' if run_cpu else ''}{'GPU' if run_gpu else ''}")

    all_results = []
    for vtp in vtp_files:
        print(f"\n{'='*60}")
        print(f"{vtp.name}")
        results = benchmark_file(vtp, radius_hit, g_max, run_cpu, run_gpu,
                                 args.sparse, args.batch_size)
        all_results.extend(results)

        # Print speedup if both ran
        cpu_geo = [r for r in results
                   if r["label"].startswith("cpu:") and ":geodesic" in r["label"]
                   and r["status"] == "OK"]
        gpu_geo = [r for r in results
                   if r["label"].startswith("gpu:") and ":geodesic" in r["label"]
                   and r["status"] == "OK"]
        if cpu_geo and gpu_geo:
            speedup = cpu_geo[0]["seconds"] / max(gpu_geo[0]["seconds"], 0.01)
            print(f"  -> geodesic speedup: {speedup:.1f}x")

    # Write CSV
    out_path = Path(args.output)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f,
                                fieldnames=["label", "seconds", "status", "command"])
        writer.writeheader()
        writer.writerows(all_results)

    # Summary table
    print(f"\n{'='*60}")
    print(f"Results -> {out_path}")
    print(f"\n{'label':<50} {'seconds':>8}  {'status'}")
    print("-" * 68)
    for r in all_results:
        print(f"{r['label']:<50} {r['seconds']:>8.2f}  {r['status']}")

    totals = [r for r in all_results if ":total" in r["label"]]
    for t in totals:
        print(f"\n  {t['label']}: {t['seconds']:.2f}s")


if __name__ == "__main__":
    main()
