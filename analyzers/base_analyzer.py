from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import numpy as np


@dataclass
class AnalysisResult:
    score: float                        # 0-100
    violations: list = field(default_factory=list)
    feedback: list = field(default_factory=list)   # list of str
    rep_completed: bool = False
    rep_count: int = 0
    phase: str = ''                     # e.g. 'BAJA', 'SUBE', 'HOLD'


class BaseAnalyzer(ABC):
    @abstractmethod
    def analyze(self, keypoints: np.ndarray) -> AnalysisResult:
        """
        Analyze a set of 17 MoveNet keypoints.

        keypoints: np.ndarray shape (17, 3), each row is [y, x, confidence]
                   with y and x normalized to [0, 1].
        """
