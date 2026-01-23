import numpy as np

from overlayaug.occlusion import bbox_covered_fraction


def test_bbox_covered_fraction():
    coverage = np.zeros((4, 4), dtype=bool)
    coverage[1:3, 1:3] = True
    box = np.array([1, 1, 3, 3], dtype=np.float32)
    assert bbox_covered_fraction(coverage, box) == 1.0
    box2 = np.array([0, 0, 4, 4], dtype=np.float32)
    assert bbox_covered_fraction(coverage, box2) == 4 / 16
