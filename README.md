# pycurv-gpu

GPU (PyTorch) reimplementation of the [pycurv](https://github.com/kalemaria/pycurv) tensor-voting
curvature pipeline used to estimate principal curvatures on triangle meshes segmented from
cryo-electron tomograms. It targets the specific AVV (area-weighted vector voting) code path that
the [surface_morphometrics](https://github.com/grotjahnlab/surface_morphometrics) pipeline runs in
production, and produces output that is a drop-in replacement for CPU pycurv's `.vtp`/`.csv`/`.gt`
files (same property names: `n_v`, `kappa_1`, `curvedness_VV`, etc.).

## What this does (and doesn't) implement

pycurv-gpu reimplements exactly the slice of pycurv that surface_morphometrics uses:

| Stage                              | CPU pycurv                                   | pycurv-gpu                                         |
| ---------------------------------- | -------------------------------------------- | -------------------------------------------------- |
| Mesh load + triangle graph         | `TriangleGraph.build_graph_from_vtk_surface` | `core.mesh_io.build_from_vtp` + `build_adjacency`  |
| Small component removal            | `find_small_connected_components`            | `core.preprocessing.remove_small_components`       |
| Border purge (segmentation meshes) | `find_vertices_near_border(purge=True)`      | `core.preprocessing.remove_wrong_border_triangles` |
| Pass 1: normal vector voting       | `normals_estimation`                         | `core.voting.normal_vector_voting`                 |
| Pass 2: AVV curvature voting       | `curvature_estimation` (`area2=True`)        | `core.voting.curvature_voting`                     |
| Geodesic neighborhoods             | graph-tool `shortest_distance` per vertex    | `core.geodesic.sssp_triangle_batch` (batched SSSP) |
| Output                             | `.vtp` / `.csv` / `.gt`                      | same formats, same property names                  |

**Not implemented** (not used by surface_morphometrics' production pipeline): SSVV/RVV/NVV/VCTV
voting method variants, the vertex-based `PointGraph` workflow, `run_gen_surface` (marching-cubes /
Hoppe surface reconstruction from a segmentation), and pycurv's inter-surface distance module.
surface_morphometrics reimplements distance/verticality/thickness itself (VTK/scipy) on top of this
pipeline's `n_v`/`xyz` outputs, rather than calling additional pycurv modules.

## Install

```bash
# In a conda env with a CUDA-enabled torch build (see https://pytorch.org/get-started/locally/):
pip install -e .
# or, without installing as a package:
pip install -r requirements.txt
```

Optional: `.gt` (graph-tool) output requires `conda install -c conda-forge graph-tool` (no pip
wheel exists) in the same environment. Everything else (`.vtp`, `.csv`) has no graph-tool
dependency. graph-tool's compiled extension needs a newer libstdc++ than torch otherwise pulls in
first when both are installed together, so `core/__init__.py` imports graph-tool before torch to
avoid that -- no action needed on your end, just install it into the same env as torch.

## Usage

```bash
# Single mesh
python run_gpu.py surface.vtp --radius-hit 10

# From a surface_morphometrics config.yml (reads radius_hit/pixel_size/min_component/exclude_borders)
python run_gpu.py surface.vtp --config config.yml

# Segmentation-derived mesh with a jagged border (screened-Poisson meshes should leave this off)
python run_gpu.py surface.vtp --radius-hit 10 --remove-wrong-borders

# Also write a graph-tool .gt file for surface_morphometrics' downstream steps
python run_gpu.py surface.vtp --radius-hit 10 --gt
```

Key flags (`python run_gpu.py --help` for the full list): `--radius-hit`, `--pixel-size`,
`--min-component`, `--exclude-borders`, `--remove-wrong-borders`, `--epsilon`/`--eta`
(orientation-classification parameters, default 0/0 = all triangles treated as surface patches,
matching surface_morphometrics), `--batch-size`, `--sssp {spfa,delta}`, `--device`.

### Programmatic API

```python
from core.api import run_pipeline

outputs = run_pipeline(
    "surface.vtp", radius_hit=10, min_component=30, exclude_borders=1,
)
# outputs: {"vtp": ..., "csv": ..., "runtimes_csv": ..., "num_triangles": ..., "total_seconds": ...}
```

`run_pipeline()` mirrors CPU pycurv's `normals_directions_and_curvature_estimation` /
surface_morphometrics' `curvature.run_pycurv(filename, folder, ...)` signature so it can be called
in-process instead of shelling out to `run_gpu.py`.

### Caching

Pass 1 (normal voting) output is cached to `{basename}.NVV_rh{radius_hit}.gpu_normals.npz`
(plain numpy, no graph-tool dependency) and reused on the next run against the same mesh, mirroring
CPU pycurv's `.NVV_rh{radius_hit}.gt` skip-if-exists behavior. Delete this file (or pass
`cache_normals=False` / `--no-cache-normals`) to force a full recompute.

## Repository layout

```
core/
  triangle_graph_gpu.py   TriangleGraphGPU: per-triangle geometry + adjacency tensors
  mesh_io.py              VTP load/save, adjacency construction, .gt export
  preprocessing.py        scaling, small-component removal, border purge
  geodesic.py             batched SSSP on the triangle dual graph
  geodesic_delta.py        alternative delta-stepping SSSP backend (--sssp delta)
  voting.py               Pass 1 (normals) + Pass 2 (AVV curvature) tensor voting
  api.py                  run_pipeline(): the importable, high-level entry point
run_gpu.py                CLI wrapper around core.api.run_pipeline
test_sphere.py            analytic validation on a sphere (known curvature = 1/R)
benchmark_geodesic.py     geodesic-step-only CPU vs GPU benchmark
compare/                  edge-case/robustness validation suite
```

## Development

Active development happens on feature branches off `main`; see `git log --graph --all` for the
current state. Requires a CUDA GPU for realistic performance, but falls back to `cpu` automatically
if none is available.
