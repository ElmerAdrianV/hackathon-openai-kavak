from __future__ import annotations
from typing import Tuple, List
import numpy as np

class OnlineCalibrator:
    """Tiny online linear regressor with optional variance head.
    yhat = w^T x + b, trained via SGD on squared error.
    """

    def __init__(self, dim: int, lr: float = 1e-2, l2: float = 1e-4):
        self.w = np.zeros(dim, dtype=float)
        self.b = 0.0
        self.lr = lr
        self.l2 = l2
        # Uncertainty head as a single scalar s >= 0 mapped from disagreement features.
        self.sigma_base = 0.75

    def predict(self, x: np.ndarray, disagreement: float = 0.0) -> Tuple[float, float]:
        yhat = float(np.dot(self.w, x) + self.b)
        # Simple uncertainty: grow with disagreement
        sigma = float(max(0.1, self.sigma_base * (1.0 + disagreement)))
        # clip to [0,5]
        yhat = min(5.0, max(0.0, yhat))
        return yhat, sigma

    def partial_fit(self, x: np.ndarray, y: float):
        # One-step SGD for squared loss with L2
        yhat = float(np.dot(self.w, x) + self.b)
        grad = (yhat - y)
        self.w -= self.lr * (grad * x + self.l2 * self.w)
        self.b -= self.lr * grad
