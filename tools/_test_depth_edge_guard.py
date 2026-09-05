"""Regression: edited-image payload must not cross clean depth boundaries."""

import numpy as np

from ai_wear.surface.projection import reject_depth_edge_payload


size = 21
coverage = np.ones((size, size), dtype=bool)
depth = np.ones((size, size), dtype=np.float32)
depth[:, 11:] = 2.0
mask = np.ones((size, size), dtype=np.float32)
rgb = np.ones((size, size, 3), dtype=np.float32)

guarded_mask, guarded_rgb, safe = reject_depth_edge_payload(
    mask, rgb, depth, coverage, radius=10.0, guard_pixels=2)

# Both sides of the discontinuity are protected; stable interiors remain.
assert not safe[:, 9:14].any()
assert np.allclose(guarded_mask[:, 9:14], 0.0)
assert np.allclose(guarded_rgb[:, 9:14], 0.5)
assert guarded_mask[10, 5] == 1.0
assert np.allclose(guarded_rgb[10, 5], 1.0)

# A background silhouette receives the same protection on the covered side.
coverage2 = np.zeros((size, size), dtype=bool)
coverage2[4:17, 4:17] = True
depth2 = np.where(coverage2, 1.0, 0.0).astype(np.float32)
_, _, safe2 = reject_depth_edge_payload(
    mask, rgb, depth2, coverage2, radius=10.0, guard_pixels=1)
assert safe2[10, 10]
assert not safe2[4:6, 4:17].any()

print("Depth-edge payload guard: PASS")
