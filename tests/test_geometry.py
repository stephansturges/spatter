import numpy as np

from spatteraug.geometry import flip_boxes, scale_boxes, translate_boxes


def test_scale_boxes():
    boxes = np.array([[0, 0, 10, 20]], dtype=np.float32)
    out = scale_boxes(boxes, 0.5)
    assert np.allclose(out, [[0, 0, 5, 10]])


def test_flip_boxes():
    boxes = np.array([[0, 0, 10, 20]], dtype=np.float32)
    out = flip_boxes(boxes, 20, 30, hflip=True, vflip=False)
    assert np.allclose(out, [[10, 0, 20, 20]])


def test_translate_boxes():
    boxes = np.array([[1, 2, 3, 4]], dtype=np.float32)
    out = translate_boxes(boxes, 5, 6)
    assert np.allclose(out, [[6, 8, 8, 10]])
