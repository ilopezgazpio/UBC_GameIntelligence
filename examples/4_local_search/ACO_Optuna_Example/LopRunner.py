from LopDataGenerator import LopDataGenerator
from AntColonyOptimizer import AntColonyOptimizer
from ExperimentPlotter import ExperimentPlotter

class LopRunner:

    """
    Main runner class (Client).
    Orchestrates the data loading, algorithm execution, and plotting.
    """

    def __init__(self, size: int = 20, iterations: int = 50, n_ants: int = 15):
        self.size = size
        self.iterations = iterations
        self.n_ants = n_ants


    def run(self):
        # 1. Generate Data
        print("--- Step 1: Generating Data ---")
        generator = LopDataGenerator()
        matrix = generator.generate_random_instance(self.size, seed=42)
        print(f"Generated matrix of size {self.size}x{self.size}")

        # 2. Initialize Algorithm
        print("--- Step 2: Initializing ACO ---")
        aco = AntColonyOptimizer(matrix, n_ants=self.n_ants, alpha=1.0, beta=2.0, evaporation=0.1)

        # 3. Run Optimization
        print("--- Step 3: Running Optimization ---")
        best_sol, best_score = aco.run(iterations=self.iterations)

        print(f"\nFinal Result: Best Score = {best_score}")
        print(f"Best Ordering: {best_sol}")

        # 4. Plot Results
        print("--- Step 4: Plotting Results ---")
        plotter = ExperimentPlotter()
        plotter.plot_convergence(aco.history)


# --- Entry Point ---
if __name__ == "__main__":
    runner = LopRunner(size=30, iterations=100, n_ants=20)
    runner.run()
