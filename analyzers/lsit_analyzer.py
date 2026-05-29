import numpy as np
from analyzers.base_analyzer import BaseAnalyzer, AnalysisResult
from core.pose_utils import (
    calculate_angle, keypoints_visible,
    LEFT_SHOULDER, RIGHT_SHOULDER,
    LEFT_ELBOW, RIGHT_ELBOW,
    LEFT_WRIST, RIGHT_WRIST,
    LEFT_HIP, RIGHT_HIP,
    LEFT_KNEE, RIGHT_KNEE,
    LEFT_ANKLE, RIGHT_ANKLE,
)
from core.scoring_engine import Violation, compute_score, smooth_score

_HIP_TARGET = 90.0        # ideal hip angle (shoulder→hip→knee)
_HIP_TOL = 20.0
_LEG_STRAIGHT_MIN = 165.0 # hip→knee→ankle must be near straight
_ARM_STRAIGHT_MIN = 165.0 # shoulder→elbow→wrist must be near straight


class LSitAnalyzer(BaseAnalyzer):
    def __init__(self):
        self._prev_score = 100.0

    def analyze(self, keypoints: np.ndarray) -> AnalysisResult:
        required = [
            LEFT_SHOULDER, LEFT_HIP, LEFT_KNEE, LEFT_ANKLE,
            LEFT_ELBOW, LEFT_WRIST,
        ]
        if not keypoints_visible(keypoints, required):
            return AnalysisResult(score=self._prev_score, feedback=['No te veo bien. Ajusta la cámara.'])

        # Hip angle
        hip_l = calculate_angle(
            keypoints[LEFT_SHOULDER][:2], keypoints[LEFT_HIP][:2], keypoints[LEFT_KNEE][:2]
        )
        hip_r = calculate_angle(
            keypoints[RIGHT_SHOULDER][:2], keypoints[RIGHT_HIP][:2], keypoints[RIGHT_KNEE][:2]
        )
        hip_angle = (hip_l + hip_r) / 2.0

        # Leg straightness
        leg_l = calculate_angle(
            keypoints[LEFT_HIP][:2], keypoints[LEFT_KNEE][:2], keypoints[LEFT_ANKLE][:2]
        )
        leg_angle = leg_l

        # Arm straightness
        arm_l = calculate_angle(
            keypoints[LEFT_SHOULDER][:2], keypoints[LEFT_ELBOW][:2], keypoints[LEFT_WRIST][:2]
        )
        arm_angle = arm_l

        violations = []
        feedback = []

        hip_dev = abs(hip_angle - _HIP_TARGET)
        if hip_dev > _HIP_TOL:
            severity = min(1.0, (hip_dev - _HIP_TOL) / 30.0)
            violations.append(Violation('hip_angle', severity, weight=2.0))
            if hip_angle > _HIP_TARGET:
                feedback.append('Levanta más las piernas, buscá los 90°')
            else:
                feedback.append('Baja un poco las piernas hacia los 90°')

        if leg_angle < _LEG_STRAIGHT_MIN:
            severity = min(1.0, (_LEG_STRAIGHT_MIN - leg_angle) / 30.0)
            violations.append(Violation('bent_legs', severity, weight=1.5))
            feedback.append('Estirá bien las piernas')

        if arm_angle < _ARM_STRAIGHT_MIN:
            severity = min(1.0, (_ARM_STRAIGHT_MIN - arm_angle) / 30.0)
            violations.append(Violation('bent_arms', severity, weight=1.5))
            feedback.append('Estirá los brazos completamente')

        score = compute_score(violations)
        score = smooth_score(score, self._prev_score)
        self._prev_score = score

        if not feedback:
            feedback.append('¡L-Sit perfecto! Mantenelo')

        return AnalysisResult(
            score=score,
            violations=violations,
            feedback=feedback,
            rep_completed=False,
            rep_count=0,
            phase='HOLD',
        )
