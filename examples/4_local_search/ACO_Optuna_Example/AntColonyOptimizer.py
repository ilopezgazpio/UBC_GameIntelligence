import numpy as np
import time
from typing import List, Tuple, Optional

class AntColonyOptimizer:
    """
    Encapsulates the Ant Colony Optimization logic for LOP.
    Vectorized using NumPy for performance where applicable.
    """

    def __init__(self, matrix: np.ndarray, n_ants: int = 10, alpha: float = 1.0, beta: float = 2.0, evaporation: float = 0.1, q: float = 100.0):
        """
        Initializes the ACO solver.

        Args:
            matrix (np.ndarray): The  matrix of the problem.
            n_ants (int): Number of ants in the colony.
            alpha (float): Importance of pheromone.
            beta (float): Importance of heuristic information.
            evaporation (float): Pheromone evaporation rate (0.0 to 1.0).
            q (float): Pheromone deposit factor.
        """

        self.matrix = matrix
        self.size = matrix.shape[0]
        self.n_ants = n_ants
        self.alpha = alpha
        self.beta = beta
        self.evaporation = evaporation
        self.q = q

        # Initialize pheromones with a small constant value
        self.pheromones = np.ones((self.size, self.size)) * 0.1

        # Heuristic information (greedy preference): matrix values directly
        # In LOP, M[i][j] usually means preference of i before j.
        self.heuristic = self.matrix.astype(float)

        self.best_solution = None
        self.best_score = -np.inf
        self.history = []

    def _calculate_score(self, solution: np.ndarray) -> float:
        """
        Calculates the objective function for a given permutation (solution).
        Score = Sum of upper triangle of the matrix permuted by 'solution'.

        Args:
            solution (np.ndarray): Array of indices representing the permutation.
        """

        # Create the permuted matrix efficiently
        permuted_matrix = self.matrix[solution][:, solution]

        # Sum only elements above the diagonal (i < j)
        return np.triu(permuted_matrix, k=1).sum()

    def _construct_solutions(self) -> List[np.ndarray]:
        """
        Constructs solutions for all ants.
        """

        solutions = []
        for _ in range(self.n_ants):
            visited = np.zeros(self.size, dtype=bool)
            path = []

            # Start with a random node or logic (here we pick random first)
            current = np.random.randint(0, self.size)
            path.append(current)
            visited[current] = True

            for _ in range(self.size - 1):
                # Calculate probabilities for unvisited nodes
                unvisited_indices = np.where(~visited)[0]

                # Vectorized probability calculation
                # P(i|j) ~ (tau_ij)^alpha * (eta_ij)^beta
                tau = self.pheromones[current, unvisited_indices] ** self.alpha
                eta = self.heuristic[current, unvisited_indices] ** self.beta

                probs = tau * eta

                # Handle edge case where probs sum to 0
                if probs.sum() == 0:
                    probs = np.ones_like(probs)

                probs = probs / probs.sum()

                # Select next node based on probability
                next_node = np.random.choice(unvisited_indices, p=probs)
                path.append(next_node)
                visited[next_node] = True
                current = next_node

            solutions.append(np.array(path))
        return solutions



    def _update_pheromones(self, solutions: List[np.ndarray], scores: List[float]):
        """
        Updates the pheromone matrix based on solutions found (Evaporation + Deposit).
        """

        # 1. Evaporation
        self.pheromones *= (1 - self.evaporation)

        # 2. Deposit
        # We enforce "Elitist" strategy or simple ACO. Here: all ants deposit.
        for solution, score in zip(solutions, scores):
            # Deposit logic: add pheromone on edges (i, j) appearing in the solution
            # In LOP, the solution is an ordering. We can reinforce 'i comes immediately before j'
            # or 'i comes anywhere before j'.
            # Simplified approach: Reinforce direct edges in the path.
            for i in range(len(solution) - 1):
                u, v = solution[i], solution[i+1]
                self.pheromones[u, v] += self.q / (100000.0 / (score + 1e-5)) # Normalization trick



    def run(self, iterations: int = 100, verbose: bool = True):
        """
        Runs the ACO algorithm for a specified number of iterations.
        Added 'verbose' flag to silence output during Optuna trials.
        """
        print(f"Starting ACO optimization for {iterations} iterations...")
        start_time = time.time()

        for i in range(iterations):
            solutions = self._construct_solutions()
            scores = [self._calculate_score(sol) for sol in solutions]

            # Find iteration best
            max_idx = np.argmax(scores)
            if scores[max_idx] > self.best_score:
                self.best_score = scores[max_idx]
                self.best_solution = solutions[max_idx]

            self._update_pheromones(solutions, scores)
            self.history.append(self.best_score)

            if verbose and (i + 1) % 10 == 0:
                print(f"Iteration {i+1}/{iterations} | Best Score: {self.best_score:.2f}")

        elapsed = time.time() - start_time

        if verbose:
            print(f"Optimization finished in {elapsed:.4f} seconds.")

        return self.best_solution, self.best_score
