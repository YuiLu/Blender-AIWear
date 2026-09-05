"""Blender-side regression tests for monotonic progress and the compact run UI."""

from types import SimpleNamespace

import bpy

from ai_wear.cache import job_cache
from ai_wear.cache.job_cache import JobState
from ai_wear.operators import pipeline
from ai_wear.ui.panels import AIWEAR_PT_main


# Provider callbacks from successive views must never move global progress back.
values = []
job = job_cache.create_job()
job.state = JobState.AI
for view in range(4):
    job.progress = pipeline._capture_view_progress(view, 4, 0.0)
    values.append(job.progress)
    job.progress = pipeline._capture_view_progress(view, 4, 0.20)
    values.append(job.progress)
    for provider_progress in (0.3, 0.8):
        pipeline._set_ai_progress(job, provider_progress, "AI", view, 4)
        values.append(job.progress)
    job.progress = pipeline._capture_view_progress(view, 4, 0.88)
    values.append(job.progress)
    job.progress = pipeline._capture_view_progress(view, 4, 1.0)
    values.append(job.progress)
assert values == sorted(values), values
assert abs(values[-1] - 0.55) < 1e-9


class FakeLayout:
    def __init__(self, events):
        object.__setattr__(self, "events", events)

    def __setattr__(self, name, value):
        if name in {"enabled", "scale_y"}:
            self.events.append((name, value))
        object.__setattr__(self, name, value)

    def column(self, **_kwargs):
        return self

    def row(self, **_kwargs):
        return self

    def prop(self, *_args, **_kwargs):
        pass

    def label(self, *_args, **_kwargs):
        pass

    def operator(self, operator_id, **_kwargs):
        self.events.append(("operator", operator_id))
        return SimpleNamespace()

    def progress(self, **kwargs):
        self.events.append(("progress", kwargs))


# While running, the Generate slot becomes a percentage BAR; Open Log follows
# Replay and no separate panel is involved.
events = []
job.progress = 0.42
bpy.context.scene.ai_wear.active_job_id = job.id
AIWEAR_PT_main.draw(SimpleNamespace(layout=FakeLayout(events)), bpy.context)
kinds = [event[0] for event in events]
operators = [event[1] for event in events if event[0] == "operator"]
assert "progress" in kinds
assert "ai_wear.run_pipeline" not in operators
assert operators[-2:] == ["ai_wear.replay_downstream", "ai_wear.open_log"]
bar = next(event[1] for event in events if event[0] == "progress")
assert bar["type"] == "BAR" and "42%" in bar["text"]

print("Progress UI and monotonic mapping: PASS")
