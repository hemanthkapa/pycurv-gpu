#!/usr/bin/env python
"""
Edge-case and robustness validation for the GPU curvature pipeline.

Complements test_sphere.py (single radius_hit analytic check) by covering:

  1. A radius_hit sweep on an analytic sphere (curvature should stay close
     to the analytic value across a range of neighborhood sizes).
  2. A tiny mesh (a handful of triangles) — checks the pipeline doesn't
     crash or produce NaNs when there's barely enough geometry for voting.
  3. A mesh with several small disconnected components — checks
     min_component correctly removes the small pieces and leaves the large
     one intact.
  4. A mesh containing degenerate (near-zero-area) triangles — checks these
     are dropped at load time and don't poison neighboring curvature values
     with NaN/Inf.

Usage:
    python compare/validate_edge_cases.py
"""
import math
import os
import sys
import tempfile

import numpy as np
import torch
import vtk

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.triangle_graph_gpu import TriangleGraphGPU
from core.mesh_io import build_from_vtp, build_adjacency, compute_edge_distances
from core.preprocessing import clean_mesh
from core.geodesic import build_csr
from core.voting import run_voting

FAILURES = []


def _check(name, condition, detail=""):
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {name}" + (f" -- {detail}" if detail else ""))
    if not condition:
        FAILURES.append(name)


def _write_vtp(poly, path):
    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(path)
    writer.SetInputData(poly)
    writer.Write()


def _triangulated_sphere(radius, resolution):
    sphere = vtk.vtkSphereSource()
    sphere.SetRadius(radius)
    sphere.SetThetaResolution(resolution)
    sphere.SetPhiResolution(resolution)
    sphere.Update()
    tri = vtk.vtkTriangleFilter()
    tri.SetInputData(sphere.GetOutput())
    tri.Update()
    return tri.GetOutput()


def pick_device():
    if torch.cuda.is_available():
        return 'cuda'
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'


def run_on_vtp(path, radius_hit, min_component=0, batch_size=256, no_clean=False):
    device = pick_device()
    tg = TriangleGraphGPU(device=device)
    build_from_vtp(path, tg)
    build_adjacency(tg)
    compute_edge_distances(tg)
    if not no_clean:
        clean_mesh(tg, pixel_size=1.0, min_component=min_component)
    build_csr(tg)
    if tg.num_triangles > 0:
        run_voting(tg, radius_hit, batch_size=batch_size)
    return tg


# ---------------------------------------------------------------------------
# 1. Radius_hit sweep on an analytic sphere
# ---------------------------------------------------------------------------

def test_radius_hit_sweep():
    print("\n=== 1. radius_hit sweep on sphere (R=10, expected curvedness=0.1) ===")
    R = 10.0
    expected = 1.0 / R
    with tempfile.NamedTemporaryFile(suffix='.vtp', delete=False) as f:
        path = f.name
    try:
        _write_vtp(_triangulated_sphere(R, 32), path)
        for rh in (5.0, 7.0, 10.0, 15.0):
            tg = run_on_vtp(path, rh, min_component=0)
            curv = tg.curvedness.cpu().numpy()
            finite = np.isfinite(curv)
            rel_err = abs(float(curv[finite].mean()) - expected) / expected
            _check(f"radius_hit={rh}: curvedness within 15% of analytic",
                  finite.all() and rel_err < 0.15,
                  f"mean={curv[finite].mean():.4f} rel_err={rel_err*100:.1f}%")
    finally:
        os.unlink(path)


# ---------------------------------------------------------------------------
# 2. Tiny mesh
# ---------------------------------------------------------------------------

def test_tiny_mesh():
    print("\n=== 2. Tiny mesh (single tetrahedron, 4 triangles) ===")
    points = vtk.vtkPoints()
    for p in [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)]:
        points.InsertNextPoint(*p)
    faces = [(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)]
    polys = vtk.vtkCellArray()
    for f in faces:
        polys.InsertNextCell(3, f)
    poly = vtk.vtkPolyData()
    poly.SetPoints(points)
    poly.SetPolys(polys)

    with tempfile.NamedTemporaryFile(suffix='.vtp', delete=False) as f:
        path = f.name
    try:
        _write_vtp(poly, path)
        tg = run_on_vtp(path, radius_hit=2.0, min_component=0, batch_size=8)
        _check("loads all 4 triangles", tg.num_triangles == 4,
              f"got {tg.num_triangles}")
        if tg.kappa_1 is not None:
            _check("no NaN/Inf in kappa_1/kappa_2",
                  bool(torch.isfinite(tg.kappa_1).all()) and
                  bool(torch.isfinite(tg.kappa_2).all()))
    finally:
        os.unlink(path)


# ---------------------------------------------------------------------------
# 3. Small disconnected components
# ---------------------------------------------------------------------------

def test_small_components_removed():
    print("\n=== 3. Small disconnected components (min_component filtering) ===")
    big = _triangulated_sphere(10.0, 20)  # ~few hundred triangles

    # Tiny separate blob far away (a handful of triangles)
    points = vtk.vtkPoints()
    for p in [(100, 0, 0), (101, 0, 0), (100, 1, 0), (100, 0, 1)]:
        points.InsertNextPoint(*p)
    small_faces = [(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)]
    small_polys = vtk.vtkCellArray()
    for f in small_faces:
        small_polys.InsertNextCell(3, f)
    small = vtk.vtkPolyData()
    small.SetPoints(points)
    small.SetPolys(small_polys)

    append = vtk.vtkAppendPolyData()
    append.AddInputData(big)
    append.AddInputData(small)
    append.Update()

    with tempfile.NamedTemporaryFile(suffix='.vtp', delete=False) as f:
        path = f.name
    try:
        _write_vtp(append.GetOutput(), path)

        device = pick_device()
        tg = TriangleGraphGPU(device=device)
        build_from_vtp(path, tg)
        build_adjacency(tg)
        compute_edge_distances(tg)
        n_before = tg.num_triangles
        clean_mesh(tg, pixel_size=1.0, min_component=30)
        n_after = tg.num_triangles

        _check("small 4-triangle blob removed by min_component=30",
              n_after < n_before and n_after > 0,
              f"{n_before} -> {n_after} triangles")
    finally:
        os.unlink(path)


# ---------------------------------------------------------------------------
# 4. Degenerate (zero-area) triangles
# ---------------------------------------------------------------------------

def test_degenerate_triangles_dropped():
    print("\n=== 4. Degenerate (zero-area) triangles ===")
    sphere = _triangulated_sphere(10.0, 16)

    n_expected = sphere.GetNumberOfCells()  # before adding the degenerate cell

    points = sphere.GetPoints()
    # Add 3 duplicate points to create a zero-area triangle
    p0 = points.GetPoint(0)
    idx = [points.InsertNextPoint(p0) for _ in range(3)]

    polys = sphere.GetPolys()
    polys.InsertNextCell(3, idx)

    poly = vtk.vtkPolyData()
    poly.SetPoints(points)
    poly.SetPolys(polys)

    with tempfile.NamedTemporaryFile(suffix='.vtp', delete=False) as f:
        path = f.name
    try:
        _write_vtp(poly, path)
        device = pick_device()
        tg = TriangleGraphGPU(device=device)
        build_from_vtp(path, tg)
        _check("zero-area triangle dropped at load",
              tg.num_triangles == n_expected,
              f"loaded {tg.num_triangles}, expected {n_expected}")

        build_adjacency(tg)
        compute_edge_distances(tg)
        build_csr(tg)
        run_voting(tg, radius_hit=5.0, batch_size=256)
        _check("no NaN/Inf in curvedness after voting",
              bool(torch.isfinite(tg.curvedness).all()))
    finally:
        os.unlink(path)


def main():
    test_radius_hit_sweep()
    test_tiny_mesh()
    test_small_components_removed()
    test_degenerate_triangles_dropped()

    print(f"\n{'='*60}")
    if FAILURES:
        print(f"FAILED ({len(FAILURES)}): {', '.join(FAILURES)}")
        sys.exit(1)
    else:
        print("All edge-case checks PASSED")


if __name__ == "__main__":
    main()
