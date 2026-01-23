import numpy as np

from spatteraug.composite import blend_roi


def test_blend_roi_simple():
    base = np.zeros((2, 2, 3), dtype=np.uint8)
    overlay = np.zeros((2, 2, 4), dtype=np.uint8)
    overlay[..., :3] = 255
    overlay[..., 3] = 128
    blend_roi(base, overlay, 0, 0)
    assert base[0, 0, 0] in (127, 128)
    assert base[1, 1, 1] in (127, 128)
