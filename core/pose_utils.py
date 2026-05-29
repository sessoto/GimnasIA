import numpy as np

# MoveNet keypoint indices (COCO format)
NOSE = 0
LEFT_EYE = 1
RIGHT_EYE = 2
LEFT_EAR = 3
RIGHT_EAR = 4
LEFT_SHOULDER = 5
RIGHT_SHOULDER = 6
LEFT_ELBOW = 7
RIGHT_ELBOW = 8
LEFT_WRIST = 9
RIGHT_WRIST = 10
LEFT_HIP = 11
RIGHT_HIP = 12
LEFT_KNEE = 13
RIGHT_KNEE = 14
LEFT_ANKLE = 15
RIGHT_ANKLE = 16

# Skeleton connections for drawing
SKELETON_EDGES = [
    (NOSE, LEFT_EYE), (NOSE, RIGHT_EYE),
    (LEFT_EYE, LEFT_EAR), (RIGHT_EYE, RIGHT_EAR),
    (LEFT_SHOULDER, RIGHT_SHOULDER),
    (LEFT_SHOULDER, LEFT_ELBOW), (LEFT_ELBOW, LEFT_WRIST),
    (RIGHT_SHOULDER, RIGHT_ELBOW), (RIGHT_ELBOW, RIGHT_WRIST),
    (LEFT_SHOULDER, LEFT_HIP), (RIGHT_SHOULDER, RIGHT_HIP),
    (LEFT_HIP, RIGHT_HIP),
    (LEFT_HIP, LEFT_KNEE), (LEFT_KNEE, LEFT_ANKLE),
    (RIGHT_HIP, RIGHT_KNEE), (RIGHT_KNEE, RIGHT_ANKLE),
]


def calculate_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Angle in degrees at point b, between vectors ba and bc."""
    ba = a - b
    bc = c - b
    norm = np.linalg.norm(ba) * np.linalg.norm(bc)
    if norm < 1e-6:
        return 0.0
    cos_angle = np.dot(ba, bc) / norm
    return float(np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0))))


def is_visible(keypoint: np.ndarray, threshold: float = 0.3) -> bool:
    """True if keypoint confidence exceeds threshold."""
    return float(keypoint[2]) >= threshold


def get_xy(keypoint: np.ndarray, frame_w: int, frame_h: int) -> np.ndarray:
    """Convert normalized [y, x] to pixel coordinates."""
    y, x = float(keypoint[0]), float(keypoint[1])
    return np.array([x * frame_w, y * frame_h])


def keypoints_visible(kps: np.ndarray, indices: list, threshold: float = 0.3) -> bool:
    """True if all specified keypoints are visible."""
    return all(is_visible(kps[i], threshold) for i in indices)
