import numpy as np
from typing import List


class Solution:
    def forward_and_backward(self,
                              x: List[float],
                              W1: List[List[float]], b1: List[float],
                              W2: List[List[float]], b2: List[float],
                              y_true: List[float]) -> dict:
        # Loss: MSE = mean((predictions - y_true)^2)
        #
        # Return dict with keys:
        #   'loss':  float (MSE loss, rounded to 4 decimals)
        #   'dW1':   2D list (gradient w.r.t. W1, rounded to 4 decimals)
        #   'db1':   1D list (gradient w.r.t. b1, rounded to 4 decimals)
        #   'dW2':   2D list (gradient w.r.t. W2, rounded to 4 decimals)
        #   'db2':   1D list (gradient w.r.t. b2, rounded to 4 decimals)
        # pass
        x, w1, w2, yTrue, b1, b2 = np.array(x), np.array(W1), np.array(W2), np.array(y_true), np.array(b1), np.array(b2)
        
        z1 = x @ w1.T + b1
        a1 = np.maximum(0, z1)
        z2 = a1 @ w2.T + b2
        loss = np.mean((z2 - yTrue)**2)

        # dL / dw2 = (dL/dz2) (dz2/dw2)
        dL_dz2 = 2 * ((z2 - yTrue)/len(z2))
        dz2_dw = a1
        dL_dw2 = dL_dz2.reshape(-1, 1) @ a1.reshape(1, -1)

        # dL / dw1 = (dL/dz2) (dz2/da1) (da1/dz1) (dz1/dw1)
        dz2_da1 = w2
        da1_dz1 = (z1 > 0)
        dz1_dw1 = x

        dL_dw1 = ((dL_dz2) @ (dz2_da1) * (da1_dz1)).reshape(-1,1) @ (dz1_dw1).reshape(1,-1)

        dL_db1 = (dL_dz2 @ dz2_da1) * da1_dz1


        return {'loss': np.round(loss,4), 'dW1': np.round(dL_dw1,4), 'db1': np.round(dL_db1,4), 'dW2': np.round(dL_dw2,4), 'db2': np.round(dL_dz2,4)}



        