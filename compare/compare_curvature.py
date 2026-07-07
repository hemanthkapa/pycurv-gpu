#!/usr/bin/env python
"""Six-column curvature accuracy comparison: GPU pycurv vs CPU pycurv reference.

For each mesh (OMM, IMM) and each of the six curvature quantities, aligns the
GPU output triangle-by-triangle against the CPU reference (merged on triangle
index) and reports agreement metrics.
"""
import os
import numpy as np
import pandas as pd

GPU_DIR = "/home/kapa/pycurv-gpu/compare/test1"
CPU_DIR = "/home/kapa/surface_morphometrics/morphometrics/test1/results"

MESHES = ["OMM", "IMM"]

# (label, GPU column, CPU column)
COLUMNS = [
    ("kappa_1",         "kappa1",          "kappa_1"),
    ("kappa_2",         "kappa2",          "kappa_2"),
    ("gauss_curvature", "gauss_curvature", "gauss_curvature_VV"),
    ("mean_curvature",  "mean_curvature",  "mean_curvature_VV"),
    ("shape_index",     "shape_index",     "shape_index_VV"),
    ("curvedness",      "curvedness",      "curvedness_VV"),
]


def load_gpu(mesh):
    path = os.path.join(GPU_DIR, f"YTC042_2_lam10_ts_004_labels_{mesh}.surface.AVV_rh9.csv")
    df = pd.read_csv(path, sep=";")
    df = df.rename(columns={df.columns[0]: "index"})
    df["index"] = df["index"].astype(int)
    return df, path


def load_cpu(mesh):
    path = os.path.join(CPU_DIR, f"YTC042_2_lam10_ts_004_labels_{mesh}.AVV_rh9.csv")
    df = pd.read_csv(path)
    df["index"] = df["index"].astype(int)
    return df, path


def stats(gpu, cpu):
    """Agreement metrics between two aligned 1-D arrays."""
    gpu = np.asarray(gpu, dtype=np.float64)
    cpu = np.asarray(cpu, dtype=np.float64)
    mask = np.isfinite(gpu) & np.isfinite(cpu)
    n = int(mask.sum())
    g, c = gpu[mask], cpu[mask]
    diff = g - c
    abs_diff = np.abs(diff)
    mae = float(abs_diff.mean())
    rmse = float(np.sqrt((diff ** 2).mean()))
    max_abs = float(abs_diff.max())
    bias = float(diff.mean())
    denom = np.abs(c)
    rel_mask = denom > 1e-12
    rel = float((abs_diff[rel_mask] / denom[rel_mask]).mean()) if rel_mask.any() else float("nan")
    if g.std() > 0 and c.std() > 0:
        pearson = float(np.corrcoef(g, c)[0, 1])
    else:
        pearson = float("nan")
    scale = float(np.abs(c).mean())
    nrmse = rmse / scale if scale > 0 else float("nan")
    return dict(n=n, pearson=pearson, mae=mae, rmse=rmse, nrmse=nrmse,
                max_abs=max_abs, bias=bias, mean_rel=rel,
                gpu_mean=float(g.mean()), cpu_mean=float(c.mean()))


def main():
    all_rows = []
    for mesh in MESHES:
        gpu_df, gpu_path = load_gpu(mesh)
        cpu_df, cpu_path = load_cpu(mesh)
        print("=" * 100)
        print(f"MESH: {mesh}")
        print(f"  GPU: {gpu_path}  ({len(gpu_df)} triangles)")
        print(f"  CPU: {cpu_path}  ({len(cpu_df)} triangles)")

        # Select and rename only the needed columns to avoid name collisions.
        gpu_sel = gpu_df[["index"] + [g for _, g, _ in COLUMNS]].rename(
            columns={g: f"gpu__{lbl}" for lbl, g, _ in COLUMNS})
        cpu_sel = cpu_df[["index"] + [c for _, _, c in COLUMNS]].rename(
            columns={c: f"cpu__{lbl}" for lbl, _, c in COLUMNS})

        merged = gpu_sel.merge(cpu_sel, on="index",
                               how="inner", validate="one_to_one")
        print(f"  Matched triangles (inner join on index): {len(merged)}")

        # sanity: how many index values are unique to one side
        only_gpu = len(set(gpu_df["index"]) - set(cpu_df["index"]))
        only_cpu = len(set(cpu_df["index"]) - set(gpu_df["index"]))
        if only_gpu or only_cpu:
            print(f"  WARNING: index-only rows -> GPU-only {only_gpu}, CPU-only {only_cpu}")

        header = (f"    {'quantity':<16}{'n':>8}{'pearson':>10}{'MAE':>13}"
                  f"{'RMSE':>13}{'nRMSE':>10}{'max|Δ|':>13}{'bias':>13}{'mean_rel':>11}")
        print(header)
        print("    " + "-" * (len(header) - 4))
        for label, gcol, ccol in COLUMNS:
            s = stats(merged[f"gpu__{label}"], merged[f"cpu__{label}"])
            print(f"    {label:<16}{s['n']:>8}{s['pearson']:>10.5f}"
                  f"{s['mae']:>13.5e}{s['rmse']:>13.5e}{s['nrmse']:>10.4f}"
                  f"{s['max_abs']:>13.5e}{s['bias']:>13.5e}{s['mean_rel']:>11.4f}")
            all_rows.append(dict(mesh=mesh, quantity=label, **s))
        print()

    out = pd.DataFrame(all_rows)
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "curvature_comparison_metrics.csv")
    out.to_csv(out_path, index=False)
    print("=" * 100)
    print(f"Wrote metrics table: {out_path}")


if __name__ == "__main__":
    main()
