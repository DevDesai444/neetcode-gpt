class Solution:
    def get_minimizer(self, iterations: int, learning_rate: float, init: int) -> float:
        # Objective function: f(x) = x^2
        # Derivative:         f'(x) = 2x
        # Update rule:        x = x - learning_rate * f'(x)
        # Round final answer to 5 decimal places

        def dfx(x):
            return 2*x
        i=1
        x = init
        while i<=iterations:
            x = x - (learning_rate * dfx(x))
            i += 1
        return round(x,5)