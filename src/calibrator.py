from __future__ import annotations
from typing import Tuple
import numpy as np


class OnlineCalibrator:
    """
    Tiny online linear regressor yhat = w^T x + b with SGD + L2.
    Also returns a crude sigma that increases with disagreement.
    """

    def __init__(self, dim: int, lr: float = 1e-2, l2: float = 1e-4):
        self.w = np.zeros(dim, dtype=float)
        self.b = 0.0
        self.lr = lr
        self.l2 = l2
        self.sigma_base = 0.75

    def predict(self, x: np.ndarray, disagreement: float = 0.0) -> Tuple[float, float]:
        yhat = float(np.dot(self.w, x) + self.b)
        yhat = min(5.0, max(0.0, yhat))
        sigma = float(max(0.1, self.sigma_base * (1.0 + disagreement)))
        return yhat, sigma

    def partial_fit(self, x: np.ndarray, y: float):
        yhat = float(np.dot(self.w, x) + self.b)
        grad = yhat - y
        self.w -= self.lr * (grad * x + self.l2 * self.w)
        self.b -= self.lr * grad
