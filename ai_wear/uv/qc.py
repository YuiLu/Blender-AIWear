"""UV quality control.

Checks: existence, degenerate (zero-area) UV triangles, overlap (raster
collision), flipped (negative area), 0..1 / UDIM bbox, utilization, modifier
risk (base vs evaluated topology). Problems are locatable to faces and produce a
structured report so the operator can warn (Mode A) or auto-unwrap (Mode B).
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .rasterizer import build_uv_field, _find_uv_layer


# Below this point a nominally valid atlas has too few texels to preserve the
# projected image detail.  The old 2% gate let a 1024px gaming-console unwrap
# through with only 22k occupied texels; seam/padding then operated on islands
# only a few pixels wide and the material looked blocky and blurred.
MIN_UTILIZATION = 0.10
MAX_OVERLAP_RATIO = 0.02
# Real production meshes occasionally contain a handful of collapsed sliver
# triangles (the dining-chair asset has 16 / 22,013).  The rasterizer already
# skips them safely, so rejecting the entire atlas is counterproductive.  Keep
# a strict ratio gate so materially broken unwraps still fail loudly.
MAX_DEGENERATE_RATIO = 0.001


def quality_issues(report: dict) -> list[str]:
    """Return fatal UV-QC reasons; presentation code should show all of them."""
    issues = []
    if not report.get("has_uv", False):
        issues.append("missing UV layer")
        return issues
    if report.get("zero_area_ratio", 0.0) > MAX_DEGENERATE_RATIO:
        issues.append(
            f"degenerate UV triangles={report.get('zero_area_count', 0)}/"
            f"{report.get('ntri', 0)} ({report.get('zero_area_ratio', 0.0):.3%})")
    if report.get("overlap_ratio", 0.0) >= MAX_OVERLAP_RATIO:
        issues.append(f"overlap={report.get('overlap_ratio', 0.0):.2%}")
    if report.get("out_of_01", False):
        issues.append("UV coordinates outside 0..1")
    if report.get("utilization", 0.0) < MIN_UTILIZATION:
        issues.append(f"utilization={report.get('utilization', 0.0):.1%}")
    return issues


def quality_warnings(report: dict) -> list[str]:
    """Return non-fatal conditions worth exposing to the artist."""
    warnings = []
    count = int(report.get("zero_area_count", 0))
    if count and report.get("zero_area_ratio", 0.0) <= MAX_DEGENERATE_RATIO:
        warnings.append(
            f"skipping {count} minor degenerate UV triangles "
            f"({report.get('zero_area_ratio', 0.0):.3%})")
    if report.get("flipped_count", 0):
        warnings.append(
            f"{report.get('flipped_count')} mirrored/flipped UV triangles")
    if report.get("modifier_risk") == "topology_change":
        warnings.append("evaluated modifiers change mesh topology")
    return warnings


def failure_summary(report: dict) -> str:
    issues = quality_issues(report)
    return "; ".join(issues) if issues else "no fatal UV-QC issue"


def has_uv(mesh, layer_name: Optional[str] = None) -> bool:
    return _find_uv_layer(mesh, layer_name) is not None


def _uv_triangle_stats(mesh, layer_name: Optional[str]) -> dict:
    layer = _find_uv_layer(mesh, layer_name)
    if layer is None:
        return {"has_uv": False}
    nloops = len(mesh.loops)
    uv = np.empty(nloops * 2, dtype=np.float32)
    layer.data.foreach_get("uv", uv)
    uv = uv.reshape(nloops, 2)
    ntri = len(mesh.loop_triangles)
    tri_loops = np.empty(ntri * 3, dtype=np.int32)
    mesh.loop_triangles.foreach_get("loops", tri_loops)
    tri_loops = tri_loops.reshape(ntri, 3)
    tri_uv = uv[tri_loops]

    # signed area per triangle
    a = tri_uv[:, 0]; b = tri_uv[:, 1]; c = tri_uv[:, 2]
    signed = 0.5 * ((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1])
                    - (c[:, 0] - a[:, 0]) * (b[:, 1] - a[:, 1]))
    zero = np.abs(signed) < 1e-10
    flipped = signed < -1e-12
    return {
        "has_uv": True,
        "ntri": int(ntri),
        "zero_area_count": int(zero.sum()),
        "flipped_count": int(flipped.sum()),
        "flipped_ratio": float(flipped.mean()) if ntri else 0.0,
        "bbox": (float(uv[:, 0].min()), float(uv[:, 1].min()),
                 float(uv[:, 0].max()), float(uv[:, 1].max())),
        "out_of_01": bool((uv[:, 0].min() < -1e-6 or uv[:, 0].max() > 1 + 1e-6
                            or uv[:, 1].min() < -1e-6 or uv[:, 1].max() > 1 + 1e-6)),
        "tri_uv": tri_uv,
        "signed_area": signed,
    }


def compute_uv_qc(obj, layer_name: Optional[str], depsgraph=None,
                   low_res: int = 256) -> dict:
    """Return a structured QC report dict."""
    import bpy
    dg = depsgraph or bpy.context.evaluated_depsgraph_get()
    eval_obj = obj.evaluated_get(dg)
    emesh = eval_obj.to_mesh()
    try:
        stats = _uv_triangle_stats(emesh, layer_name)
        report = {
            "object": obj.name,
            "uv_layer": layer_name or "(active)",
            "has_uv": stats.get("has_uv", False),
            "ntri": stats.get("ntri", 0),
            "zero_area_count": stats.get("zero_area_count", 0),
            "zero_area_ratio": (
                float(stats.get("zero_area_count", 0))
                / max(1, int(stats.get("ntri", 0)))),
            "flipped_count": stats.get("flipped_count", 0),
            "flipped_ratio": stats.get("flipped_ratio", 0.0),
            "bbox": stats.get("bbox", None),
            "out_of_01": stats.get("out_of_01", False),
            "overlap_ratio": 0.0,
            "utilization": 0.0,
            "modifier_risk": _modifier_risk(obj, dg),
            "ok": False,
        }
        if stats.get("has_uv"):
            field = build_uv_field(obj, layer_name, low_res, depsgraph=dg,
                                   track_coverage=True)
            if field is not None:
                cov = field.coverage
                valid = field.valid
                report["overlap_ratio"] = field.overlap_ratio
                report["degenerate_count"] = field.degenerate_count
                report["utilization"] = float(valid.sum()) / float(low_res * low_res)
        report["issues"] = quality_issues(report)
        report["warnings"] = quality_warnings(report)
        report["ok"] = not report["issues"]
        return report
    finally:
        eval_obj.to_mesh_clear()


def _modifier_risk(obj, depsgraph) -> str:
    """Compare base vs evaluated topology to flag risky modifiers."""
    base = obj.data
    eval_obj = obj.evaluated_get(depsgraph)
    emesh = eval_obj.to_mesh()
    try:
        if len(base.vertices) != len(emesh.vertices):
            return "topology_change"
        # If a triangulate/subsurf changes face/vert counts, write-back is unsafe
        if len(base.polygons) != len(emesh.polygons):
            return "topology_change"
    finally:
        eval_obj.to_mesh_clear()
    # Non-destructive modifiers only
    return "none"


def format_report(report: dict) -> str:
    if not report.get("has_uv"):
        return "UV QC: No UV layer found."
    lines = [
        f"UV QC — {report.get('object')}",
        f"  Layer:        {report.get('uv_layer')}",
        f"  Triangles:    {report.get('ntri')}",
        f"  Zero-area:    {report.get('zero_area_count')} "
        f"({report.get('zero_area_ratio', 0.0)*100:.3f}%)",
        f"  Degenerate:   {report.get('degenerate_count', 0)}",
        f"  Flipped:      {report.get('flipped_count')} ({report.get('flipped_ratio')*100:.1f}%)",
        f"  Overlap:      {report.get('overlap_ratio')*100:.2f}%",
        f"  Utilization:  {report.get('utilization')*100:.1f}%",
        f"  Out of 0..1:  {report.get('out_of_01')}",
        f"  BBox:         {report.get('bbox')}",
        f"  Modifier risk:{report.get('modifier_risk')}",
        f"  BAKE-READY:   {report.get('ok')}",
    ]
    if report.get("issues"):
        lines.append("  Blocking:     " + "; ".join(report["issues"]))
    if report.get("warnings"):
        lines.append("  Warnings:     " + "; ".join(report["warnings"]))
    return "\n".join(lines)
