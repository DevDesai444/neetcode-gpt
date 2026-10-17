import numpy as np
from numpy.typing import NDArray
from typing import List


class Solution:
    def forward(self, x: NDArray[np.float64], weights: List[NDArray[np.float64]], biases: List[NDArray[np.float64]]) -> NDArray[np.float64]:
        # x: 1D input array
        # weights: list of 2D weight matrices
        # biases: list of 1D bias vectors
        # Apply ReLU after each hidden layer, no activation on output layer
        # return np.round(your_answer, 5)
        # pass

        '''
        x = [1.0, 2.0]
        weights = [[[0.1, 0.2], 
                    [0.3, 0.4]], 

                    [[0.5], 
                    [0.6]]]
        biases = [[0.1, 0.1], 
                [0.0]]
        Output:  [1.06]
        '''
        h = x
        for l in range(len(weights)):
            h = h @ weights[l] + biases[l]

            if l < len(weights)-1 :
                h = np.maximum(h,0)
        return np.round(h, 5)
            




