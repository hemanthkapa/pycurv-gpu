#!/usr/bin/env python
"""Benchmark full GPU voting pipeline vs CPU reference CSVs.

Runs run_gpu.py-equivalent pipeline (load, clean, CSR, voting, CSV export),
records total time + PYCURV_PROFILE breakdown, and compares kappa1/kappa2
against CPU pycurv reference outputs.

Usage:
    PYCURV_PROFILE=1 python benchmark_voting.py
    PYCURV_PROFILE=1 python benchmark_voting.py --batch-size 256 1024
    PYCURV_PROFILE=1 python benchmark_voting.py --sssp delta
"""

import argparse
import csv
import io
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PYTHON = os.environ.get("PYCURV_PYTHON", sys.executable)

OMM_VTP = Path(
    "/home/kapa/surface_morphometrics/morphometrics/test1/results/"
    "YTC042_2_lam10_ts_004_labels_OMM.surface.vtp"
)
IMM_VTP = Path(
    "/home/kapa/surface_morphometrics/morphometrics/test1/results/"
    "YTC042_2_lam10_ts_004_labels_IMM.surface.vtp"
)
CPU_DIR = Path("/home/kapa/surface_morphometrics/morphometrics/test1/results")
OUT_DIR = Path("/home/kapa/pycurv-gpu/compare/test1")

CPU_TARGETS = {"OMM": 23.0, "IMM": 102.1}


def _load_cpu_kappa(cpu_csv):
    """Load kappa_1/kappa1 from CPU reference CSV."""
    df = pd.read_csv(cpu_csv)
    cols = {c.lower(): c for c in df.columns}
    k1 = df[cols.get("kappa_1", cols.get("kappa1"))].to_numpy(dtype=np.float64)
    k2 = df[cols.get("kappa_2", cols.get("kappa2"))].to_numpy(dtype=np.float64)
    return k1, k2


def _load_gpu_kappa(gpu_csv):
    """Load kappa1 from GPU output CSV (semicolon-separated)."""
    df = pd.read_csv(gpu_csv, sep=";")
    if df.columns[0] == "Unnamed: 0" or df.columns[0] == "":
        df = df.iloc[:, 1:]
    k1 = df["kappa1"].to_numpy(dtype=np.float64)
    k2 = df["kappa2"].to_numpy(dtype=np.float64)
    return k1, k2


def compare_curvature(gpu_csv, cpu_csv):
    """Return accuracy metrics vs CPU reference."""
    gpu_k1, gpu_k2 = _load_gpu_kappa(gpu_csv)
    cpu_k1, cpu_k2 = _load_cpu_kappa(cpu_csv)
    n = min(len(gpu_k1), len(cpu_k1))
    if n == 0:
        return {"status": "FAIL(empty)", "n": 0}

    g1, c1 = gpu_k1[:n], cpu_k1[:n]
    g2, c2 = gpu_k2[:n], cpu_k2[:n]

    def _metrics(g, c):
        diff = np.abs(g - c)
        valid = np.isfinite(g) & np.isfinite(c)
        if not valid.any():
            return {"max_abs": float("nan"), "corr": float("nan")}
        gv, cv = g[valid], c[valid]
        corr = np.corrcoef(gv, cv)[0, 1] if gv.size > 1 else 1.0
        return {"max_abs": float(diff[valid].max()), "corr": float(corr)}

    m1 = _metrics(g1, c1)
    m2 = _metrics(g2, c2)
    ok = m1["max_abs"] < 1e-3 and m2["max_abs"] < 1e-3
    ok = ok and m1["corr"] > 0.9999 and m2["corr"] > 0.9999
    return {
        "status": "OK" if ok else "FAIL(accuracy)",
        "n": n,
        "kappa1_max_abs": m1["max_abs"],
        "kappa1_corr": m1["corr"],
        "kappa2_max_abs": m2["max_abs"],
        "kappa2_corr": m2["corr"],
    }


def run_voting_pipeline(vtp_path, out_dir, radius_hit, batch_size, sssp="spfa",
                        no_clean=False):
    """Run full voting pipeline in-process; return timing + profile text."""
    os.environ["PYCURV_PROFILE"] = "1"

    import torch
    from core import (TriangleGraphGPU, build_from_vtp, build_adjacency,
                      compute_edge_distances, clean_mesh, build_csr, run_voting)
    from run_gpu import extract_curvatures, _format_rh

    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    vtp_path = Path(vtp_path)
    basename = vtp_path.stem
    rh_str = _format_rh(radius_hit)
    csv_path = out_dir / f"{basename}.AVV_rh{rh_str}.csv"

    t0 = time.perf_counter()
    tg = TriangleGraphGPU(device=device)
    build_from_vtp(str(vtp_path), tg)
    build_adjacency(tg)
    compute_edge_distances(tg)
    if not no_clean:
        clean_mesh(tg, pixel_size=1.0, min_component=30)
    build_csr(tg)

    profile_text = ""
    buf = io.StringIO()
    old_stdout = sys.stdout

    class _Tee:
        def write(self, s):
            old_stdout.write(s)
            buf.write(s)
        def flush(self):
            old_stdout.flush()

    sys.stdout = _Tee()
    try:
        run_voting(tg, radius_hit, batch_size, cache_sssp=True, sssp=sssp)
    finally:
        sys.stdout = old_stdout

    output = buf.getvalue()
    if "Voting profile breakdown:" in output:
        profile_text = output.split("Voting profile breakdown:")[-1].strip()

    extract_curvatures(tg, str(csv_path))
    total = time.perf_counter() - t0

    return {
        "total_seconds": total,
        "num_triangles": tg.num_triangles,
        "csv_path": str(csv_path),
        "profile": profile_text,
        "device": device,
        "sssp": sssp,
    }


def benchmark_mesh(mesh_key, vtp_path, batch_sizes, out_dir, radius_hit=9,
                   sssp="spfa"):
    """Benchmark one mesh at multiple batch sizes."""
    cpu_csv = CPU_DIR / f"{vtp_path.stem.replace('.surface', '')}.AVV_rh9.csv"
    if not cpu_csv.exists():
        # OMM/IMM naming: strip .surface from stem
        stem = vtp_path.stem  # e.g. ..._OMM.surface
        cpu_csv = CPU_DIR / f"{stem}.AVV_rh9.csv"

    results = []
    for bs in batch_sizes:
        print(f"\n{'='*60}")
        print(f"{mesh_key}  batch_size={bs}  sssp={sssp}")
        r = run_voting_pipeline(vtp_path, out_dir, radius_hit, bs, sssp=sssp)
        acc = compare_curvature(r["csv_path"], cpu_csv)
        cpu_target = CPU_TARGETS[mesh_key]
        speed_ok = r["total_seconds"] < cpu_target
        print(f"  Total: {r['total_seconds']:.1f}s  (CPU target {cpu_target}s) "
              f"{'PASS' if speed_ok else 'FAIL'}")
        print(f"  Accuracy: {acc['status']}  "
              f"k1 max={acc.get('kappa1_max_abs', '?'):.2e}  "
              f"corr={acc.get('kappa1_corr', '?'):.6f}")
        if r["profile"]:
            print(f"  Profile:\n{r['profile']}")

        results.append({
            "mesh": mesh_key,
            "batch_size": bs,
            "sssp": sssp,
            "total_seconds": round(r["total_seconds"], 2),
            "num_triangles": r["num_triangles"],
            "cpu_target": cpu_target,
            "speed_ok": speed_ok,
            "accuracy_status": acc["status"],
            "kappa1_max_abs": acc.get("kappa1_max_abs"),
            "kappa1_corr": acc.get("kappa1_corr"),
            "kappa2_max_abs": acc.get("kappa2_max_abs"),
            "kappa2_corr": acc.get("kappa2_corr"),
            "profile": r["profile"].replace("\n", " | "),
        })
    return results


def git_bisect_quick(vtp_path, batch_size=256):
    """Time OMM at two commits to confirm regression source."""
    repo = Path(__file__).resolve().parent
    commits = {"3db8a79": "fast (pre-regression)", "1dd0f90": "slow (regression)"}
    timings = {}

    for commit, label in commits.items():
        script = f"""
import sys, time, os
os.environ['PYCURV_PROFILE'] = '0'
sys.path.insert(0, '{repo}')
import torch
from core import (TriangleGraphGPU, build_from_vtp, build_adjacency,
                  compute_edge_distances, clean_mesh, build_csr, run_voting)
vtp = '{vtp_path}'
t0 = time.perf_counter()
tg = TriangleGraphGPU(device='cuda')
build_from_vtp(vtp, tg)
build_adjacency(tg)
compute_edge_distances(tg)
clean_mesh(tg, pixel_size=1.0, min_component=30)
build_csr(tg)
run_voting(tg, 9, {batch_size}, cache_sssp=True)
print(f'VOTING_TIME={{time.perf_counter()-t0:.2f}}')
"""
        cmd = ["git", "stash", "-q"]
        subprocess.run(cmd, cwd=repo, capture_output=True)
        subprocess.run(["git", "checkout", commit, "-q"], cwd=repo, capture_output=True)
        proc = subprocess.run(
            [PYTHON, "-c", script], cwd=repo, capture_output=True, text=True,
            env={**os.environ, "PYCURV_PROFILE": "0"},
        )
        subprocess.run(["git", "checkout", "perf/sssp-runtime-opt", "-q"], cwd=repo)
        subprocess.run(["git", "stash", "pop", "-q"], cwd=repo, capture_output=True)

        out = proc.stdout + proc.stderr
        t = None
        for line in out.splitlines():
            if line.startswith("VOTING_TIME="):
                t = float(line.split("=")[1])
        timings[commit] = {"label": label, "seconds": t, "ok": proc.returncode == 0}
        print(f"  {commit} ({label}): {t:.1f}s" if t else f"  {commit}: FAILED")

    return timings


def main():
    parser = argparse.ArgumentParser(description="Benchmark GPU voting vs CPU")
    parser.add_argument("--batch-size", type=int, nargs="+", default=[256, 1024])
    parser.add_argument("--radius-hit", type=float, default=9)
    parser.add_argument("--output", default="benchmark_results.csv")
    parser.add_argument("--out-dir", default=str(OUT_DIR))
    parser.add_argument("--sssp", choices=["spfa", "delta"], default="spfa")
    parser.add_argument("--bisect", action="store_true",
                        help="Quick timing at 3db8a79 vs 1dd0f90 on OMM")
    parser.add_argument("--mesh", choices=["OMM", "IMM", "both"], default="both")
    args = parser.parse_args()

    if args.bisect:
        print("Bisect confirm: OMM @ batch_size=256")
        timings = git_bisect_quick(OMM_VTP, batch_size=256)
        fast = timings.get("3db8a79", {}).get("seconds")
        slow = timings.get("1dd0f90", {}).get("seconds")
        if fast and slow:
            print(f"  Finding: 1dd0f90 is {slow/fast:.2f}x slower than 3db8a79")
        return

    meshes = []
    if args.mesh in ("OMM", "both"):
        meshes.append(("OMM", OMM_VTP))
    if args.mesh in ("IMM", "both"):
        meshes.append(("IMM", IMM_VTP))

    all_results = []
    for key, vtp in meshes:
        if not vtp.exists():
            print(f"WARNING: {vtp} not found, skipping {key}")
            continue
        all_results.extend(
            benchmark_mesh(key, vtp, args.batch_size, args.out_dir,
                           args.radius_hit, args.sssp))

    if not all_results:
        print("No results.")
        return

    out_path = Path(args.output)
    fieldnames = list(all_results[0].keys())
    write_header = not out_path.exists() or out_path.stat().st_size == 0
    with open(out_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerows(all_results)

    print(f"\n{'='*60}")
    print(f"Results appended to {out_path}")
    print(f"\n{'mesh':<6} {'B':>6} {'sssp':<6} {'time':>8} {'target':>8} {'speed':<6} {'acc'}")
    print("-" * 60)
    for r in all_results:
        print(f"{r['mesh']:<6} {r['batch_size']:>6} {r['sssp']:<6} "
              f"{r['total_seconds']:>8.1f} {r['cpu_target']:>8.1f} "
              f"{'PASS' if r['speed_ok'] else 'FAIL':<6} {r['accuracy_status']}")


if __name__ == "__main__":
    main()
