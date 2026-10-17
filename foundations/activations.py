import numpy as np
from numpy.typing import NDArray


class Solution:
    
    def sigmoid(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        ans = np.empty(0)
        for i in range(len(z)):
            if z[i] >= 0:
                ans = np.append(ans, 1.0 / (1.0 + np.exp(-z[i])))
            else:
                ans = np.append(ans, np.exp(z[i]) / (1.0 + np.exp(z[i])))
        return np.round(ans, 5)



    def relu(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array
        # Formula: max(0, z) element-wise
        return np.maximum(0, z)