"""Click entry point for the `morphometrics pycurv_gpu` plugin command.

Import-light on purpose: only `click` at module load so
`morphometrics --help` (which loads plugins for short help) stays fast.
Heavy deps (torch / core.api) are imported inside the callback.
"""
from __future__ import annotations

import glob
import os
import sys

import click


@click.command(name="pycurv_gpu")
@click.argument("configfile", type=click.Path(exists=True))
@click.argument("surface", required=False, default=None)
@click.option("-f", "--force", is_flag=True, default=False,
              help="Skip interactive confirmation prompts.")
@click.option("--no-gt", is_flag=True, default=False,
              help="Skip graph-tool .gt output (needed for downstream "
                   "morphometrics steps; default is to write it).")
@click.option("--device", default=None,
              help="Torch device (e.g. cuda, cuda:0, cpu). Default: auto.")
def pycurv_gpu(configfile, surface, force, no_gt, device):
    """GPU curvature analysis (pycurv-gpu) on surface meshes.

    Drop-in alternative to `morphometrics pycurv` for the curvature step.
    Reads radius_hit / pixel_size / min_component / exclude_borders from
    CONFIGFILE and writes the usual .AVV_rh* .vtp/.csv/.gt outputs into
    work_dir.

    CONFIGFILE: path to config.yml.
    SURFACE: optional single .surface.vtp (basename or path; recommended for
    cluster parallelization). If omitted, all surfaces in work_dir are
    processed.
    """
    import yaml

    from core.api import run_pipeline

    with open(configfile) as file:
        config = yaml.safe_load(file)

    work_dir = config.get("work_dir") or ""
    if not work_dir:
        seg_dir = config.get("seg_dir") or ""
        if not seg_dir:
            raise click.ClickException(
                "No working directory is specified in the config file. "
                "Please specify a working directory or a data directory."
            )
        click.echo(
            "No working directory is specified in the config file. "
            "The data directory will be used for input and output."
        )
        work_dir = seg_dir
        config["work_dir"] = work_dir

    if not work_dir.endswith(os.sep):
        work_dir_slash = work_dir + os.sep
    else:
        work_dir_slash = work_dir

    cm = config.get("curvature_measurements") or {}
    radius_hit = cm.get("radius_hit", 10)
    pixel_size = cm.get("pixel_size", 1.0)
    min_component = cm.get("min_component", 30)
    exclude_borders = cm.get("exclude_borders", 0)
    remove_wrong_borders = bool(cm.get("remove_wrong_borders", False))

    if surface is None:
        click.echo(
            "No input file specified - will run on all VTP files in the "
            "working directory"
        )
        click.echo(
            "Recommended usage: morphometrics pycurv_gpu config.yml "
            "<meshname.surface.vtp>"
        )
        if not force:
            if not click.confirm("Continue?", default=False):
                sys.exit(1)
        mesh_files = [
            os.path.basename(f)
            for f in glob.glob(work_dir_slash + "*.surface.vtp")
        ]
        if not mesh_files:
            raise click.ClickException(
                f"No *.surface.vtp files found in {work_dir}"
            )
    else:
        click.echo("Input file specified - will run on this file only")
        mesh_files = [surface]

    os.makedirs(work_dir, exist_ok=True)

    failed_surfaces = []
    for i, surface_file in enumerate(mesh_files):
        click.echo(
            f"Processing {surface_file} ({i + 1}/{len(mesh_files)})"
        )
        vtp_path = _resolve_surface_path(surface_file, work_dir)
        if not vtp_path.endswith(".surface.vtp"):
            raise click.ClickException(
                f"Surface must end with .surface.vtp, got: {surface_file}"
            )
        try:
            run_pipeline(
                vtp_path,
                output_dir=work_dir,
                radius_hit=radius_hit,
                pixel_size=pixel_size,
                min_component=min_component,
                exclude_borders=exclude_borders,
                remove_wrong_borders=remove_wrong_borders,
                device=device,
                write_vtp=True,
                write_gt=not no_gt,
            )
            click.echo(f"Completed {surface_file}\n")
        except Exception as exc:
            click.echo(
                f"WARNING: Skipping {surface_file} due to error: {exc}\n"
            )
            failed_surfaces.append(surface_file)

    if failed_surfaces:
        click.echo("The following surfaces failed and were skipped:")
        for name in failed_surfaces:
            click.echo(f"  - {name}")

    click.echo("-------------------------------------------------------")
    click.echo(
        "pycurv_gpu complete. Check the AVV .vtp in Paraview, then continue "
        "with morphometrics distances_orientations (or refine_mesh)."
    )


def _resolve_surface_path(surface_file, work_dir):
    """Accept a basename or absolute/relative path; prefer an existing file."""
    if os.path.isfile(surface_file):
        return os.path.abspath(surface_file)
    candidate = os.path.join(work_dir, os.path.basename(surface_file))
    if os.path.isfile(candidate):
        return os.path.abspath(candidate)
    raise click.ClickException(
        f"Surface not found: {surface_file} (also tried {candidate})"
    )
