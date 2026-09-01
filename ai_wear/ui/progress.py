"""Progress panel + helper that reads the live job from the in-memory registry."""

from __future__ import annotations

import bpy

from ..operators import pipeline
from ..cache.job_cache import JobState


def _state_color(state: str) -> tuple:
    return {
        "IDLE": (0.6, 0.6, 0.6, 1),
        "RENDER": (0.5, 0.7, 1.0, 1),
        "AI": (0.9, 0.7, 0.3, 1),
        "BUILD": (0.3, 0.8, 0.8, 1),
        "BAKE": (0.3, 0.9, 0.5, 1),
        "DONE": (0.3, 0.9, 0.4, 1),
        "ERROR": (0.95, 0.3, 0.3, 1),
        "CANCEL": (0.8, 0.6, 0.2, 1),
    }.get(state, (0.8, 0.8, 0.8, 1))


class AIWEAR_PT_progress(bpy.types.Panel):
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "AI Wear"
    bl_label = "Progress"
    # No DEFAULT_CLOSED: this small panel (bar + one status line) is the only
    # in-UI surface for errors, so it stays visible. Full detail (coverage, seam
    # p95, traceback) goes to the System Console via _console_log().

    def draw(self, context):
        layout = self.layout
        # one-click access to the in-UI Text log: every progress line, AI retry
        # message, and the full error+traceback live in 'ai_wear.log'.
        layout.operator("ai_wear.open_log", icon="CONSOLE")
        job = pipeline.get_active_job(context)
        if job is None:
            layout.label(text="Idle.", icon="PAUSE")
            return
        prog = max(0.0, min(1.0, job.progress))
        layout.progress(factor=prog, type="BAR", text=f"{prog*100:.0f}%")
        # one simple status line — full detail (seam QA, coverage, traceback)
        # goes to the System Console via the timer's _console_log().
        if job.state == JobState.ERROR and job.error:
            layout.label(text=f"Error: {job.error[:80]}", icon="ERROR")
        elif job.state == JobState.DONE:
            layout.label(text="Done.", icon="CHECKMARK")
        elif job.message:
            layout.label(text=job.message, icon="INFO")
        else:
            layout.label(text=job.stage.value, icon="RENDER_RESULT")


def register():
    bpy.utils.register_class(AIWEAR_PT_progress)


def unregister():
    bpy.utils.unregister_class(AIWEAR_PT_progress)
