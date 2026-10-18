import torch
import torch.nn as nn
from typing import List, Dict


class Solution:

    def compute_activation_stats(self, model: nn.Module, x: torch.Tensor) -> List[Dict[str, float]]:
        # Forward pass through model layer by layer
        # After each nn.Linear, record: mean, std, dead_fraction
        # Run with torch.no_grad(). Round to 4 decimals.
        # pass
        ans = []
        with torch.no_grad():
            for layer in model.children():
                x = layer(x)  # (n of samples, neurons out)
                if type(layer) == nn.Linear:
                    nd = 0
                    for n in range(len(x[0])):
                        if (x[:, n] <= 0).all():
                            nd += 1
                    ans.append({"mean": round(float(x.mean()), 4), "std": round(float(x.std()), 4), "dead_fraction": float(nd/len(x[0]))})
        return ans

    def compute_gradient_stats(self, model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> List[Dict[str, float]]:
        # Forward + backward pass with nn.MSELoss
        # For each nn.Linear layer's weight gradient, record: mean, std, norm
        # Call model.zero_grad() first. Round to 4 decimals.
        # pass
        model.zero_grad()
        y_ = model(x)
        lossFunc = nn.MSELoss()
        loss = lossFunc(y_, y)
        loss.backward()

        ans = []
        for layer in model.children():
            if type(layer) == nn.Linear:
                grad = layer.weight.grad # (out, in)
                ans.append({"mean": round(float(grad.mean()), 4), "std": round(float(grad.std()), 4), "norm": round(float(torch.norm(grad)), 4)})
        return ans


    def diagnose(self, activation_stats: List[Dict[str, float]], gradient_stats: List[Dict[str, float]]) -> str:
        # Classify network health based on the stats
        # Return: 'dead_neurons', 'exploding_gradients', 'vanishing_gradients', or 'healthy'
        # Check in priority order (see problem description for thresholds)
        # pass
        # activation_stats, gradient_stats

        for layer in activation_stats:
            if layer["dead_fraction"] > 0.5:
                return 'dead_neurons'

        for layer in gradient_stats:
            if layer["norm"] > 10:
                return 'exploding_gradients'

        for layer in gradient_stats:
            if layer["norm"] < 1e-5:
                return 'vanishing_gradients'

        return 'healthy'