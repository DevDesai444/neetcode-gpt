import numpy as np
from numpy.typing import NDArray


class Solution:

    def softmax(self, z):
        shifted = z - np.max(z)
        e = np.exp(shifted)
        return np.round(e / np.sum(e), 4)