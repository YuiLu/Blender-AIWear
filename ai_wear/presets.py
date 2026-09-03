"""Built-in experiment presets matching EXPERIMENTS.md.

``seed_experiment_presets`` fills an empty scene preset list with the experiment
arms so the Presets panel isn't blank and a whole arm can be loaded in one click.
Each arm sets only the fields it varies; every other field keeps the PropertyGroup
default (which mirrors the scene default), so Load restores a consistent baseline.

The preset fields covered are the full AIWearPreset schema (prompt, camera/context,
weights, ablation switches, seam/padding, amount/feather), so Save/Load round-trips
an entire experiment arm — see ai_wear/operators/runner.py.
"""

from __future__ import annotations

# (name, overrides). Only the varied field(s) per arm; the rest use defaults.
DEFAULT_EXPERIMENT_PRESETS = (
    # baseline / main result
    ("baseline", {}),
    # Experiment 1: camera count (Counted Auto, 1 / 6 / 8)
    ("cams_01", {"camera_preset": "AUTO_COUNT", "camera_count": 1}),
    ("cams_06", {"camera_preset": "AUTO_COUNT", "camera_count": 6}),
    ("cams_08", {"camera_preset": "AUTO_COUNT", "camera_count": 8}),
    # Experiment 2: per-view context
    ("context_none", {"view_context_mode": "NONE"}),
    ("context_first", {"view_context_mode": "FIRST_ANCHOR"}),
    ("context_previous", {"view_context_mode": "PREVIOUS_VIEW"}),
    # Experiment 3: geometry prior on / off
    ("geometry_off", {"use_geometry_prior": False}),
    # Experiment 4: topology growth on / off
    ("topology_off", {"use_topology_growth": False}),
    # Experiment 5: seam fusion on / off (padding stays on)
    ("seam_off", {"seam_fuse": False}),
    # Experiment 6: wear amount 30 / 60 / 100
    ("amount_30", {"wear_amount": 30.0}),
    ("amount_60", {"wear_amount": 60.0}),
    ("amount_100", {"wear_amount": 100.0}),
    # Experiment 7: extra prompt with / without
    ("extra_on", {"prompt_extra": "add fine micro-scratches and edge chipping; "
                                  "keep large flat faces mostly clean"}),
)

# Names shipped by 0.3.2/0.3.3 that duplicate the full-feature ``baseline``.
# Remove them from scenes that already persisted the old 18-item collection.
DEPRECATED_EXPERIMENT_PRESETS = frozenset((
    "geometry_on", "topology_on", "seam_on", "extra_off",
))


def seed_experiment_presets(s) -> None:
    """Populate ``s.presets`` with the experiment arms (idempotent: only when empty)."""
    for index in range(len(s.presets) - 1, -1, -1):
        if s.presets[index].name in DEPRECATED_EXPERIMENT_PRESETS:
            s.presets.remove(index)
    if len(s.presets) > 0:
        return
    for name, overrides in DEFAULT_EXPERIMENT_PRESETS:
        p = s.presets.add()
        p.name = name
        for key, value in overrides.items():
            setattr(p, key, value)


def restore_experiment_presets(s) -> None:
    """Clear and re-seed the preset list (the panel's 'Restore' button)."""
    s.presets.clear()
    seed_experiment_presets(s)


def apply_preset(s, p) -> None:
    """Copy every preset field onto the scene settings.

    Shared by the Load button and the preset-list selection callback so both
    paths apply the identical set of fields (and any per-field update callbacks,
    e.g. the Wear Amount/Feather shader refresh, fire exactly once).
    """
    s.prompt_material = p.material
    s.prompt_wear_type = p.wear_type
    s.max_wear_state = p.max_state
    s.prompt_extra = p.prompt_extra
    s.camera_preset = p.camera_preset
    s.camera_count = p.camera_count
    s.view_context_mode = p.view_context_mode
    s.w_ai = p.w_ai; s.w_convex = p.w_convex
    s.w_expose = p.w_expose; s.w_cavity = p.w_cavity
    s.alpha = p.alpha; s.noise_amp = p.noise_amp; s.noise_scale = p.noise_scale
    s.use_ai_mask = p.use_ai_mask
    s.use_geometry_prior = p.use_geometry_prior
    s.use_topology_growth = p.use_topology_growth
    s.seam_fuse = p.seam_fuse
    s.use_padding = p.use_padding
    s.wear_amount = p.wear_amount
    s.feather = p.feather
