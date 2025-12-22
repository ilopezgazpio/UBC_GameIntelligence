from LopDataGenerator import LopDataGenerator
from AntColonyOptimizer import AntColonyOptimizer
from ExperimentPlotter import ExperimentPlotter

import optuna  # Requires: pip install optuna

import matplotlib.pyplot as plt


class LopOptunaTuner:
    """
    Responsible for tuning hyperparameters using Optuna.
    Saves state to SQLite and isolates the optimization process.
    """

    def __init__(self, problem_size: int, n_trials: int = 50, fixed_seed: int = 42):
        """
        Args:
            problem_size: The size of the matrix (N x N).
            n_trials: How many different parameter combinations to try.
            fixed_seed: Seed to generate the PROBLEM INSTANCE.
                        Crucial: We must use the SAME matrix for all trials
                        to make scores comparable.
        """
        self.problem_size = problem_size
        self.n_trials = n_trials

        # Generate the benchmark matrix ONCE.
        # Every Optuna trial will try to solve THIS specific matrix.
        self.benchmark_matrix = LopDataGenerator.generate_random_instance(
            size=problem_size, seed=fixed_seed
        )

        # Dynamic study name based on size
        # This allows you to have "lop_opt_size_20", "lop_opt_size_50" in the same DB.
        self.study_name = f"lop_optimization_size_{problem_size}"
        self.storage_url = "sqlite:///lop_optuna_experiments.db"



    def objective(self, trial: optuna.Trial) -> float:
        """
        The objective function to be maximized by Optuna.
        It defines the search space for hyperparameters.
        """

        # 1. Suggest Hyperparameters
        # We define ranges based on typical ACO literature.

        # Alpha: Influence of Pheromone (0.0 means ignore pheromone, just greedy)
        alpha = trial.suggest_float("alpha", 0.1, 5.0)

        # Beta: Influence of Heuristic (0.0 means ignore heuristic, pure pheromone)
        beta = trial.suggest_float("beta", 0.1, 5.0)

        # Evaporation: How fast trails disappear (0.01 = slow, 0.99 = instant)
        evaporation = trial.suggest_float("evaporation", 0.01, 0.5)

        # Q: Amount of pheromone deposited
        q = trial.suggest_float("q", 10.0, 1000.0)

        # n_ants: Number of agents (Integer)
        # More ants = better exploration but slower per iteration
        n_ants = trial.suggest_int("n_ants", 10, 50)

        # 2. Instantiate the Algorithm with these parameters
        # We use a FIXED number of iterations for fairness (e.g., 50).
        # We pass the pre-generated benchmark_matrix.
        aco = AntColonyOptimizer(
            matrix=self.benchmark_matrix,
            n_ants=n_ants,
            alpha=alpha,
            beta=beta,
            evaporation=evaporation,
            q=q
        )

        # 3. Run the algorithm
        _, best_score = aco.run(iterations=50, verbose=False)

        # 4. Return the value we want to MAXIMIZE
        return best_score



    def run_study(self):
        """
        Sets up the study and executes the optimization.
        """
        print(f"--- Starting Optuna Study: {self.study_name} ---")
        print(f"Storage: {self.storage_url}")

        # Create or load the study
        # direction="maximize" because higher LOP score is better
        study = optuna.create_study(
            study_name=self.study_name,
            storage=self.storage_url,
            direction="maximize",
            load_if_exists=True
        )

        # Execute the optimization
        # n_jobs=1: Sequential execution (safest for simple scripts)
        # n_jobs=-1: Use all CPU cores (faster, but harder to debug print statements)
        study.optimize(self.objective, n_trials=self.n_trials)

        # Report results
        print("\n--- Study Complete ---")
        print(f"Best parameters found for Size {self.problem_size}:")
        print(study.best_params)
        print(f"Best Value achieved: {study.best_value}")

        return study.best_params

# ==========================================
# PART 3: Execution Logic
# ==========================================

if __name__ == "__main__":
    import argparse

    # Simple CLI to choose between Standard Run and Optuna Run
    print("Select mode:")
    print("1. Standard Single Run (Visualize Convergence)")
    print("2. Hyperparameter Optimization (Optuna + SQLite)")

    mode = input("Enter 1 or 2: ").strip()

    if mode == "1":
        # --- ORIGINAL MODE ---
        # Runs one instance and plots the graph
        size = 30
        print(f"Running Standard Experiment (Size {size})...")

        gen = LopDataGenerator()
        mat = gen.generate_random_instance(size, seed=42)
        aco = AntColonyOptimizer(mat, n_ants=20, alpha=1.0, beta=2.0, evaporation=0.1)
        best_sol, best_score = aco.run(iterations=100, verbose=True)
        print(f"Final Score: {best_score}")

        # Simple plot inside main for brevity in this mode
        plt.plot(aco.history)
        plt.title("ACO Convergence")
        plt.show()

    elif mode == "2":
        # --- OPTUNA MODE ---
        # Runs many trials to find best parameters

        # You can change this size to experiment with different problem difficulties
        target_size = int(input("Enter problem size to optimize for (e.g., 20, 50): "))
        trials = int(input("Enter number of trials (e.g., 50): "))

        tuner = LopOptunaTuner(problem_size=target_size, n_trials=trials)
        best_params = tuner.run_study()

        print("\nYou can verify results using `optuna-dashboard sqlite:///lop_optuna_experiments.db`")

    else:
        print("Invalid selection.")
