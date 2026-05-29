import numpy as np
from analyzers.base_analyzer import BaseAnalyzer, AnalysisResult
from core.pose_utils import (
    calculate_angle, keypoints_visible,
    LEFT_SHOULDER, RIGHT_SHOULDER,
    LEFT_HIP, RIGHT_HIP,
    LEFT_KNEE, RIGHT_KNEE,
)
from core.scoring_engine import Violation, compute_score, smooth_score
from core.rep_counter import RepCounter

_CRUNCH_MAX = 60      # torso angle at top of crunch
_REST_MIN = 140       # torso angle at rest (lying down)


class AbsAnalyzer(BaseAnalyzer):
    def __init__(self):
        self._rep_counter = RepCounter(down_threshold=_CRUNCH_MAX, up_threshold=_REST_MIN)
        self._prev_score = 100.0

    def analyze(self, keypoints: np.ndarray) -> AnalysisResult:
        required = [LEFT_SHOULDER, LEFT_HIP, LEFT_KNEE, RIGHT_SHOULDER, RIGHT_HIP, RIGHT_KNEE]
        if not keypoints_visible(keypoints, required):
            return AnalysisResult(score=self._prev_score, feedback=['No te veo bien. Ajusta la cámara.'])

        # Torso angle: shoulder → hip → knee
        torso_l = calculate_angle(
            keypoints[LEFT_SHOULDER][:2], keypoints[LEFT_HIP][:2], keypoints[LEFT_KNEE][:2]
        )
        torso_r = calculate_angle(
            keypoints[RIGHT_SHOULDER][:2], keypoints[RIGHT_HIP][:2], keypoints[RIGHT_KNEE][:2]
        )
        torso_angle = (torso_l + torso_r) / 2.0

        violations = []
        feedback = []

        score = compute_score(violations)
        score = smooth_score(score, self._prev_score)
        self._prev_score = score

        rep_done = self._rep_counter.update(torso_angle)
        phase = 'SUBE' if torso_angle > 100 else ('BAJA' if torso_angle < 50 else 'MANTÉN')

        if not feedback:
            feedback.append('¡Bien! Sigue el ritmo')

        return AnalysisResult(
            score=score,
            violations=violations,
            feedback=feedback,
            rep_completed=rep_done,
            rep_count=self._rep_counter.count,
            phase=phase,
        )
