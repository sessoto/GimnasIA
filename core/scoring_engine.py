from dataclasses import dataclass, field


@dataclass
class Violation:
    name: str
    severity: float   # 0.0 = perfecto, 1.0 = máxima penalización
    weight: float = 1.0


def compute_score(violations: list) -> float:
    """Compute 0-100 score from a list of Violation objects."""
    if not violations:
        return 100.0
    penalty = sum(v.severity * v.weight for v in violations)
    total_weight = sum(v.weight for v in violations)
    score = 100.0 * (1.0 - penalty / total_weight)
    return max(0.0, min(100.0, score))


def smooth_score(current: float, previous: float, alpha: float = 0.3) -> float:
    """Exponential moving average to avoid flickering."""
    return alpha * current + (1.0 - alpha) * previous
