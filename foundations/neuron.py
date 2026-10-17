import numpy as np
from numpy.typing import NDArray


class Solution:
    def forward(self, x: NDArray[np.float64], w: NDArray[np.float64], b: float, activation: str) -> float:
        ans = (x @ w)+ b
        if activation == 'sigmoid':
            ans = (1.0 / (1.0 + np.exp(-ans)))
        elif activation == 'relu':
            ans = np.maximum(0,ans)
        return np.round(ans,5)
