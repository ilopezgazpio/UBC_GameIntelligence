import numpy as np
from typing import List, Tuple, Optional

class LopDataGenerator:
    """
    Responsible for generating or loading data for the Linear Ordering Problem (LOP).
    Follows SRP: Only handles data generation.
    """

    @staticmethod
    def generate_random_instance(size: int, seed: Optional[int] = None) -> np.ndarray:
        """
        Generates a random square matrix representing a LOP instance.

        Args:
            size (int): The number of items/nodes (size of the matrix side).
            seed (int, optional): Random seed for reproducibility.

        Returns:
            np.ndarray: A (size x size) matrix with random values.
        """
        if seed is not None:
            np.random.seed(seed)

        # Generate random values between 0 and 100
        matrix = np.random.randint(0, 100, size=(size, size))

        # Zero out the diagonal (no self-loops in LOP typically)
        np.fill_diagonal(matrix, 0)

        return matrix
