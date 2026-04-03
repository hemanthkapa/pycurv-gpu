# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

GPU-accelerated reimplementation of [pycurv](https://github.com/kalemaria/pycurv) — surface morphometrics for cryo-electron tomography. Computes principal curvatures on triangle meshes via tensor voting, using PyTorch for GPU acceleration.

## Environment & Dependencies

Conda environment (`gpu`). Key deps: **PyTorch**, **VTK**, **numpy**, **pandas**, **PyYAML**.

No pyproject.toml or build system — run scripts directly.

```bash
# Full pipeline
python run_gpu.py config.yaml

# Validation test (procedural sphere with known curvature)
python test_sphere.py
```

## Architecture

### Processing Pipeline

1. **Mesh I/O** (`mesh_io.py`) — Load VTP via VTK → triangle-level tensors + adjacency graph
2. **Preprocessing** (`preprocessing.py`) — Scale by pixel_size, remove small components (GPU BFS), remove border triangles
3. **Vertex Graph** (`mesh_io.py:build_vertex_graph`) — Remap to local vertex IDs, build vertex adjacency + CSR vertex→triangle mapping. **Must run after preprocessing.**
4. **Voting** (`voting.py`) — Pass 1: normal estimation via geodesic-weighted tensor voting. Pass 2: curvature estimation via Tong-Tang formula.
5. **Geodesics** (`geodesic.py`) — Batched frontier-relaxation SSSP on triangle/vertex graphs

### Central Data Structure: `TriangleGraphGPU`

Defined in `triangle_graph_gpu.py`. Mutable bag of tensors — all pipeline stages read/write fields on a single `tg` instance. Three layers:

- **Triangle-level**: `centers [T,3]`, `normals [T,3]`, `areas [T]`, `edge_src/dst/dist` (adjacency)
- **Vertex-level**: `vertex_positions [P,3]`, `vertex_normals [P,3]`, `v_edge_src/dst/dist` (mesh edges)
- **Curvature outputs**: `kappa_1/kappa_2 [P]`, `t_1/t_2 [P,3]`, `n_v [P,3]`, `gauss_curvature`, `mean_curvature`, `curvedness`, `shape_index`, `orientation_class`

### GPU Patterns

- Pure PyTorch tensor ops — no custom CUDA kernels
- `scatter_add_` / `scatter_reduce_` for parallel aggregation
- Auto-tuned batch sizes based on free GPU memory (`_get_free_memory`)
- Device support: `cuda` > `mps` > `cpu` fallback
- Temporary numpy caches (`_face_point_ids`, `_all_vertices`) used during graph construction, then discarded

## Algorithm Reference

Curvature voting follows the Tong-Tang method (same as CPU pycurv):
- **g_max** = π · radius_hit / 2 (geodesic neighborhood radius)
- **Pass 1**: Accumulate area/distance-weighted normal outer products → eigenvector = estimated normal
- **Pass 2**: Project neighbor directions onto tangent plane, compute κᵢ = 2cos((π-φ)/2) / ‖d‖, accumulate into B matrix → eigenvalues give κ₁, κ₂ via: κ₁ = 3b₁ - b₂, κ₂ = 3b₂ - b₁
- **Invariant**: κ₁ ≥ κ₂ always (enforced by swap)

## Lessons Learned

When an approach fails and the user corrects course, **append the lesson here immediately** — what was tried, why it failed, and what worked instead. This section is the living record of what works and what doesn't in this codebase. Format: `- **topic**: what failed → what works (date)`

<!-- Add entries below this line -->
- **Normal convention**: VTK gives outward-pointing normals; pycurv uses inward-pointing. Must negate normals in build_from_vtp. Without this, all curvatures have wrong sign. (2026-03-28)
- **Pass 2 eigenvector ordering**: With inward/outward normals, the eigenvalue ≈ 0 (normal direction) can be at any position in eigh output. Must identify which eigenvector aligns with n_v and use the other two for curvature — can't assume eigenvalue ordering. (2026-03-28)
- **Area weighting matters**: Previous GPU impl had no area weighting in Pass 2, producing RVV instead of AVV. surface_morphometrics uses AVV (area2=True). (2026-03-28)
