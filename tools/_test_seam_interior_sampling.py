"""Regression test: seam fusion must not blend island values with empty UV space."""

import numpy as np

from ai_wear.uv.seam_registry import SeamPair, fuse_seam, seam_qa


resolution = 16
field = np.zeros((resolution, resolution), dtype=np.float32)
valid = np.zeros_like(field, dtype=bool)

# Two UV islands with their corresponding seam edges at u=.25 and u=.75.
# Their first interior rows contain .8 and .6; island exterior is zero.
valid[4:12, 4:8] = True
field[4:12, 4:8] = 0.8
valid[4:12, 8:12] = True
field[4:12, 8:12] = 0.6
pair = SeamPair(
    edge_index=0, v0=0, v1=1,
    uv_a0=np.array([0.25, 0.25]), uv_a1=np.array([0.25, 0.75]),
    uv_b0=np.array([0.75, 0.25]), uv_b1=np.array([0.75, 0.75]),
    uv_a_inward=np.array([1.0, 0.0]),
    uv_b_inward=np.array([-1.0, 0.0]),
)

before = seam_qa(field, [pair], resolution)
fused = fuse_seam(field, [pair], resolution, diffuse_texels=1, valid=valid)
after = seam_qa(fused, [pair], resolution)

# Correct pair average is .7. Boundary-contaminated sampling produced .35.
assert np.allclose(fused[4:12, 4], 0.7, atol=1e-6)
assert np.allclose(fused[4:12, 11], 0.7, atol=1e-6)
assert float(fused[4:12, 4].min()) >= 0.6
assert not fused[~valid].any()
assert before["p95"] > 0.19
assert after["p95"] < 1e-6

print("Seam interior sampling: PASS")
