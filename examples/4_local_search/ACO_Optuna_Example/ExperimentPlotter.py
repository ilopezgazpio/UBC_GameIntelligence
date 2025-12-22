import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional


class ExperimentPlotter:

    """
    Handles the visualization of the experiment results.
    """

    @staticmethod
    def plot_convergence(history: List[float], title: str = "ACO Convergence"):
        """
        Plots the fitness score over iterations.
        """
        plt.figure(figsize=(10, 6))
        plt.plot(history, label="Best Score")
        plt.xlabel("Iteration")
        plt.ylabel("Objective Function Value")
        plt.title(title)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend()
        plt.show()
