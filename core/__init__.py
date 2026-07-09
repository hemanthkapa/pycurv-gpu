# graph-tool is optional (only used for --gt output) but must be imported before
# torch if both are installed in the same environment: graph-tool's compiled
# extension needs a newer libstdc++ (GLIBCXX) than torch otherwise pulls in first,
# causing an ImportError at `save_gt()` time. Importing it here first (before the
# `torch`-importing modules below) avoids that -- harmless no-op if not installed.
try:
    import graph_tool  # noqa: F401
except ImportError:
    pass

from .triangle_graph_gpu import TriangleGraphGPU
from .mesh_io import build_from_vtp, build_adjacency, compute_edge_distances, save_vtp, save_gt
from .preprocessing import clean_mesh, find_border_triangles, remove_wrong_border_triangles
from .geodesic import sssp_triangle_batch, build_csr, set_sssp_mode
from .voting import run_voting
from .api import run_pipeline, extract_curvatures, shape_index_classify, format_radius_hit
