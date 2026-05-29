import numpy as np
from analyzers.base_analyzer import BaseAnalyzer, AnalysisResult
from core.pose_utils import (
    calculate_angle, keypoints_visible,
    NOSE,
    LEFT_SHOULDER, RIGHT_SHOULDER,
    LEFT_ELBOW, RIGHT_ELBOW,
    LEFT_WRIST, RIGHT_WRIST,
)
from core.scoring_engine import Violation, compute_score, smooth_score
from core.rep_counter import RepCounter

_ELBOW_UP_MAX = 50       # elbow angle when chin is above bar
_ELBOW_DOWN_MIN = 155    # elbow angle at full hang


class PullupAnalyzer(BaseAnalyzer):
    def __init__(self):
        # For pull-ups: going UP means elbow angle decreasing
        # State machine: full hang (up_state) → top (down_state in terms of angle)
        # We invert: "down_threshold" = small angle (top), "up_threshold" = large angle (hang)
        self._rep_counter = RepCounter(down_threshold=_ELBOW_UP_MAX, up_threshold=_ELBOW_DOWN_MIN)
        self._prev_score = 100.0

    def analyze(self, keypoints: np.ndarray) -> AnalysisResult:
        required = [LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST, RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST]
        if not keypoints_visible(keypoints, required):
            return AnalysisResult(score=self._prev_score, feedback=['No te veo bien. Ajusta la cámara.'])

        elbow_l = calculate_angle(
            keypoints[LEFT_SHOULDER][:2], keypoints[LEFT_ELBOW][:2], keypoints[LEFT_WRIST][:2]
        )
        elbow_r = calculate_angle(
            keypoints[RIGHT_SHOULDER][:2], keypoints[RIGHT_ELBOW][:2], keypoints[RIGHT_WRIST][:2]
        )
        elbow_angle = (elbow_l + elbow_r) / 2.0

        # Chin over bar: nose y < wrist y (in normalized coords, smaller y = higher)
        chin_over = False
        if keypoints_visible(keypoints, [NOSE, LEFT_WRIST]):
            chin_over = float(keypoints[NOSE][0]) < float(keypoints[LEFT_WRIST][0])

        violations = []
        feedback = []

        if elbow_angle < _ELBOW_DOWN_MIN and not chin_over and elbow_angle < 80:
            violations.append(Violation('chin_not_over', 0.5, weight=1.5))
            feedback.append('Sube más, lleva el mentón sobre la barra')

        score = compute_score(violations)
        score = smooth_score(score, self._prev_score)
        self._prev_score = score

        rep_done = self._rep_counter.update(elbow_angle)
        phase = 'SUBE' if elbow_angle > 120 else ('BAJA' if elbow_angle < 50 else 'MANTÉN')

        if not feedback:
            feedback.append('¡Buena ejecución!')

        return AnalysisResult(
            score=score,
            violations=violations,
            feedback=feedback,
            rep_completed=rep_done,
            rep_count=self._rep_counter.count,
            phase=phase,
        )
