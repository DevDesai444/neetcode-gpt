import numpy as np
from numpy.typing import NDArray
from typing import Tuple


class Solution:
    def train(self, x: NDArray[np.float64], y: NDArray[np.float64], epochs: int, lr: float) -> Tuple[NDArray[np.float64], float]:
        # X: (n_samples, n_features)
        # y: (n_samples,) targets
        # epochs: number of training iterations
        # lr: learning rate
        #
        # Model: y_hat = X @ w + b
        # Loss: MSE = (1/n) * sum((y_hat - y)^2)
        # Initialize w = zeros, b = 0
        # return (np.round(w, 5), round(b, 5))
        # pass
        w = np.zeros(len(X[0])) # (n_features, )
        b = 0.0

        for e in range(epochs):
            y_ = x @ w + b # (n_samples, )
            # loss = np.mean((y_ - y)**2)

            dl_dw = (2/len(x)) * x.T @ (y_ - y)  # 2/n X (y_ - y) # (n_features, )
            dl_db = (2/len(x)) * np.sum((y_ - y))

            w -= lr * dl_dw
            b -= lr * dl_db
        
        return (np.round(w,5), round(b,5))
