import numpy as np
from numpy.typing import NDArray
from typing import Tuple


class Solution:
    def backward(self, x: NDArray[np.float64], w: NDArray[np.float64], b: float, y_true: float) -> Tuple[NDArray[np.float64], float]:
        # x: 1D input array
        # w: 1D weight array
        # b: scalar bias
        # y_true: true target value
        #
        # Forward: z = dot(x, w) + b, y_hat = sigmoid(z)
        # Loss: L = 0.5 * (f(x) - y_true)^2
        # grad : grad = (y_hat - y_true) (yPred * (1 - yPred))
        # Return: (dL_dw rounded to 5 decimals, dL_db rounded to 5 decimals)
        # pass
        def dfx(yPred, yTrue):
            return (yPred - yTrue) * (yPred * (1 - yPred))
        
        z = (x @ w)+ b
        yPred = (1.0 / (1.0 + np.exp(-z)))
        grad = dfx(yPred, y_true)
        return (np.round(grad * x, 5), np.round(grad, 5))
        
