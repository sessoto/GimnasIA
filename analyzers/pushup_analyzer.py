import numpy as np
from analyzers.base_analyzer import BaseAnalyzer, AnalysisResult
from core.pose_utils import (
    calculate_angle, keypoints_visible,
    LEFT_SHOULDER, RIGHT_SHOULDER,
    LEFT_ELBOW, RIGHT_ELBOW,
    LEFT_WRIST, RIGHT_WRIST,
    LEFT_HIP, RIGHT_HIP,
    LEFT_ANKLE, RIGHT_ANKLE,
)
from core.scoring_engine import Violation, compute_score, smooth_score
from core.rep_counter import RepCounter

_ELBOW_DOWN_MAX = 95     # elbow angle at bottom position
_ELBOW_UP_MIN = 155      # elbow angle at top position
_BODY_ALIGN_TOL = 20     # degrees from straight line tolerance


class PushupAnalyzer(BaseAnalyzer):
    def __init__(self):
        self._rep_counter = RepCounter(down_threshold=_ELBOW_DOWN_MAX, up_threshold=_ELBOW_UP_MIN)
        self._prev_score = 100.0

    def analyze(self, keypoints: np.ndarray) -> AnalysisResult:
        required = [LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST, RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST]
        if not keypoints_visible(keypoints, required):
            return AnalysisResult(score=self._prev_score, feedback=['No te veo bien. Ajusta la cámara.'])

        # Elbow angle: shoulder → elbow → wrist
        elbow_l = calculate_angle(
            keypoints[LEFT_SHOULDER][:2], keypoints[LEFT_ELBOW][:2], keypoints[LEFT_WRIST][:2]
        )
        elbow_r = calculate_angle(
            keypoints[RIGHT_SHOULDER][:2], keypoints[RIGHT_ELBOW][:2], keypoints[RIGHT_WRIST][:2]
        )
        elbow_angle = (elbow_l + elbow_r) / 2.0

        # Body alignment: shoulder → hip → ankle should be ~180°
        body_angle = None
        if keypoints_visible(keypoints, [LEFT_HIP, LEFT_ANKLE]):
            body_angle = calculate_angle(
                keypoints[LEFT_SHOULDER][:2], keypoints[LEFT_HIP][:2], keypoints[LEFT_ANKLE][:2]
            )

        violations = []
        feedback = []

        if body_angle is not None:
            deviation = abs(180.0 - body_angle)
            if deviation > _BODY_ALIGN_TOL:
                severity = min(1.0, (deviation - _BODY_ALIGN_TOL) / 30.0)
                violations.append(Violation('body_misalign', severity, weight=2.0))
                if body_angle < 160:
                    feedback.append('Levanta las caderas, mantén el cuerpo en línea recta')
                else:
                    feedback.append('Baja las caderas, no dejes que se hundan')

        score = compute_score(violations)
        score = smooth_score(score, self._prev_score)
        self._prev_score = score

        rep_done = self._rep_counter.update(elbow_angle)
        phase = 'BAJA' if elbow_angle > 140 else ('SUBE' if elbow_angle < 70 else 'MANTÉN')

        if not feedback:
            feedback.append('¡Excelente forma!')

        return AnalysisResult(
            score=score,
            violations=violations,
            feedback=feedback,
            rep_completed=rep_done,
            rep_count=self._rep_counter.count,
            phase=phase,
        )
