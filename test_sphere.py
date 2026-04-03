#!/usr/bin/env python
"""
Validate GPU curvature pipeline on a sphere (known curvature = 1/R).

Creates a VTP sphere, runs the pipeline, and checks results.
"""
import vtk
import torch
from core.triangle_graph_gpu import TriangleGraphGPU
from core.mesh_io import build_from_vtp, build_adjacency, compute_edge_distances
from core.voting import run_voting


def create_sphere_vtp(filename, radius=10.0, resolution=32):
    """Create a sphere VTP file."""
    sphere = vtk.vtkSphereSource()
    sphere.SetRadius(radius)
    sphere.SetThetaResolution(resolution)
    sphere.SetPhiResolution(resolution)
    sphere.Update()

    tri = vtk.vtkTriangleFilter()
    tri.SetInputData(sphere.GetOutput())
    tri.Update()

    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(filename)
    writer.SetInputData(tri.GetOutput())
    writer.Write()
    return radius


def main():
    import tempfile, os

    R = 10.0
    expected_kappa = 1.0 / R

    with tempfile.NamedTemporaryFile(suffix='.vtp', delete=False) as f:
        vtp_path = f.name

    try:
        create_sphere_vtp(vtp_path, radius=R, resolution=32)

        device = 'cpu'
        if torch.cuda.is_available():
            device = 'cuda'
        elif torch.backends.mps.is_available():
            device = 'mps'

        tg = TriangleGraphGPU(device=device)
        build_from_vtp(vtp_path, tg)
        build_adjacency(tg)
        compute_edge_distances(tg)

        radius_hit = 5.0
        run_voting(tg, radius_hit, batch_size=128)

        k1 = tg.kappa_1.cpu()
        k2 = tg.kappa_2.cpu()
        si = tg.shape_index.cpu()
        curved = tg.curvedness.cpu()

        print(f"\n{'='*50}")
        print(f"Sphere R={R}, expected kappa = {expected_kappa:.4f}")
        print(f"  kappa_1:  mean={k1.mean():.4f}  std={k1.std():.4f}  "
              f"range=[{k1.min():.4f}, {k1.max():.4f}]")
        print(f"  kappa_2:  mean={k2.mean():.4f}  std={k2.std():.4f}  "
              f"range=[{k2.min():.4f}, {k2.max():.4f}]")
        print(f"  shape_i:  mean={si.mean():.4f}  "
              f"range=[{si.min():.4f}, {si.max():.4f}]")
        print(f"  curved:   mean={curved.mean():.4f}  "
              f"range=[{curved.min():.4f}, {curved.max():.4f}]")

        k1_err = abs(k1.mean().item() - expected_kappa) / expected_kappa
        k2_err = abs(k2.mean().item() - expected_kappa) / expected_kappa
        print(f"\n  kappa_1 rel error: {k1_err*100:.1f}%")
        print(f"  kappa_2 rel error: {k2_err*100:.1f}%")

        # Mean curvature and curvedness are better metrics for triangle graph
        mean_curv = (k1.mean().item() + k2.mean().item()) / 2
        mean_err = abs(mean_curv - expected_kappa) / expected_kappa
        curv_err = abs(curved.mean().item() - expected_kappa) / expected_kappa
        print(f"  mean_curv rel error: {mean_err*100:.1f}%")
        print(f"  curvedness rel error: {curv_err*100:.1f}%")

        if mean_err < 0.1 and curv_err < 0.15:
            print("\n  PASS: mean curvature and curvedness within tolerance")
        else:
            print("\n  FAIL: curvatures too far from expected")

    finally:
        os.unlink(vtp_path)


if __name__ == "__main__":
    main()
