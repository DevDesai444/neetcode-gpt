import numpy as np
from numpy.typing import NDArray


class Solution:

    def softmax(self, z):
        m = -np.inf
        d = 0.0
        for i in range(len(z)):
            mPrev = m
            m = max(m, z[i])
            d = d * np.exp(mPrev - m) + np.exp(z[i] - m)

        out = np.exp(z - m) / d
        return np.round(out, 4)