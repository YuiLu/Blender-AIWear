"""Shader subpackage.

Re-exports the overlay-shader entry points so the property update callbacks
in ``properties.py`` (``from . import shader``) resolve ``on_wear_amount`` /
``on_feather`` and drive the real-time 0..100 preview with no AI re-run, and so
the pipeline can call ``attach_wear_overlay``.
"""

from .wear_nodegroup import (  # noqa: F401
    ensure_node_group,
    attach_wear_overlay,
    set_amount,
    set_feather,
    update_active,
    on_wear_amount,
    on_feather,
    GROUP_NAME,
    PREVIEW_MAT,
    WEARTIME_NODE,
    WORNTEX_NODE,
    MASKGROUP_NODE,
    UVMAP_NODE,
    OVERLAY_MIX,
    ALPHA_MUL,
    BASE_SRC,
    DELTA_SUB,
    DELTA_SCALE,
    BASE_ADD,
)
