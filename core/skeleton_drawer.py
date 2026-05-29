import cv2
import numpy as np
from core.pose_utils import SKELETON_EDGES, get_xy, is_visible

_EDGE_COLOR = (0, 255, 0)
_POINT_COLOR = (0, 120, 255)
_LOW_CONF_COLOR = (80, 80, 80)
_VIS_THRESHOLD = 0.3


def draw_skeleton(frame: np.ndarray, keypoints: np.ndarray) -> np.ndarray:
    """Draw pose skeleton on frame in-place. Returns the frame."""
    h, w = frame.shape[:2]

    for i, j in SKELETON_EDGES:
        kp_a = keypoints[i]
        kp_b = keypoints[j]
        if is_visible(kp_a, _VIS_THRESHOLD) and is_visible(kp_b, _VIS_THRESHOLD):
            pt_a = tuple(get_xy(kp_a, w, h).astype(int))
            pt_b = tuple(get_xy(kp_b, w, h).astype(int))
            cv2.line(frame, pt_a, pt_b, _EDGE_COLOR, 2, cv2.LINE_AA)

    for kp in keypoints:
        color = _POINT_COLOR if is_visible(kp, _VIS_THRESHOLD) else _LOW_CONF_COLOR
        pt = tuple(get_xy(kp, w, h).astype(int))
        cv2.circle(frame, pt, 5, color, -1, cv2.LINE_AA)

    return frame
