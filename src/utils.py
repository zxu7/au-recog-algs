import numpy as np


def points_to_np(points):
    """convert dlib.points to numpy array"""
    out = np.zeros((len(points), 2))
    for i, p in enumerate(points):
        out[i] = (p.x, p.y)
    return out