import numpy as np
from analyzers.base_analyzer import BaseAnalyzer, AnalysisResult
from core.pose_utils import (
    calculate_angle, keypoints_visible, get_xy,
    LEFT_SHOULDER, RIGHT_SHOULDER,
    LEFT_HIP, RIGHT_HIP,
    LEFT_KNEE, RIGHT_KNEE,
    LEFT_ANKLE, RIGHT_ANKLE,
)
from core.scoring_engine import Violation, compute_score, smooth_score
from core.rep_counter import RepCounter

# Angles in degrees
_GOOD_DEPTH_MAX = 105       # knee angle below this = good depth
_SHALLOW_MIN = 130          # knee angle above this = too shallow
_TORSO_LEAN_MIN = 45        # hip angle below this = torso too forward
_KNEE_CAVE_TOL = 0.05       # knee x must be within this fraction of ankle x


class SquatAnalyzer(BaseAnalyzer):
    def __init__(self):
        self._rep_counter = RepCounter(down_threshold=_GOOD_DEPTH_MAX, up_threshold=160)
        self._prev_score = 100.0

    def analyze(self, keypoints: np.ndarray) -> AnalysisResult:
        required = [LEFT_HIP, LEFT_KNEE, LEFT_ANKLE, RIGHT_HIP, RIGHT_KNEE, RIGHT_ANKLE]
        if not keypoints_visible(keypoints, required):
            return AnalysisResult(score=self._prev_score, feedback=['No te veo bien. Aléjate un poco.'])

        # Average left/right for robustness
        l_hip   = keypoints[LEFT_HIP]
        l_knee  = keypoints[LEFT_KNEE]
        l_ankle = keypoints[LEFT_ANKLE]
        r_hip   = keypoints[RIGHT_HIP]
        r_knee  = keypoints[RIGHT_KNEE]
        r_ankle = keypoints[RIGHT_ANKLE]

        # Knee angle: hip → knee → ankle
        knee_angle_l = calculate_angle(l_hip[:2], l_knee[:2], l_ankle[:2])
        knee_angle_r = calculate_angle(r_hip[:2], r_knee[:2], r_ankle[:2])
        knee_angle = (knee_angle_l + knee_angle_r) / 2.0

        # Torso angle (shoulder → hip → knee) — check if shoulders visible
        torso_angle = None
        if keypoints_visible(keypoints, [LEFT_SHOULDER, RIGHT_SHOULDER]):
            l_sh = keypoints[LEFT_SHOULDER]
            r_sh = keypoints[RIGHT_SHOULDER]
            # Use side facing camera (higher confidence)
            if l_sh[2] >= r_sh[2]:
                torso_angle = calculate_angle(l_sh[:2], l_hip[:2], l_knee[:2])
            else:
                torso_angle = calculate_angle(r_sh[:2], r_hip[:2], r_knee[:2])

        # Knee cave: knee x should track ankle x
        knee_cave = abs(float(l_knee[1]) - float(l_ankle[1])) > _KNEE_CAVE_TOL or \
                    abs(float(r_knee[1]) - float(r_ankle[1])) > _KNEE_CAVE_TOL

        # Build violations
        violations = []
        feedback = []

        if knee_angle > _SHALLOW_MIN:
            severity = min(1.0, (knee_angle - _SHALLOW_MIN) / 30.0)
            violations.append(Violation('shallow', severity, weight=2.0))
            feedback.append('Baja más, dobla más las rodillas')

        if torso_angle is not None and torso_angle < _TORSO_LEAN_MIN:
            severity = min(1.0, (_TORSO_LEAN_MIN - torso_angle) / 30.0)
            violations.append(Violation('torso_lean', severity, weight=1.5))
            feedback.append('Mantén el torso más erguido')

        if knee_cave:
            violations.append(Violation('knee_cave', 0.6, weight=1.0))
            feedback.append('Rodillas hacia afuera, no dejes que colapsen')

        score = compute_score(violations)
        score = smooth_score(score, self._prev_score)
        self._prev_score = score

        rep_done = self._rep_counter.update(knee_angle)
        phase = 'BAJA' if knee_angle > 140 else ('SUBE' if knee_angle < 90 else 'MANTÉN')

        if not feedback:
            feedback.append('¡Buena postura! Sigue así')

        return AnalysisResult(
            score=score,
            violations=violations,
            feedback=feedback,
            rep_completed=rep_done,
            rep_count=self._rep_counter.count,
            phase=phase,
        )
