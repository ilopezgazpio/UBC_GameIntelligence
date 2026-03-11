from __future__ import annotations


"""
Generic Optuna runner for RL algorithms on Gymnasium environments.

Design goals
------------
- Same experimental settings across many algorithms.
- Algorithm-specific hyperparameter search spaces.
- One SQLite database for everything, but one Optuna study per algorithm.
- Summary table with best/top-K statistics saved back into the same SQLite file.
- Minimal coupling to the concrete RL algorithm implementation.

Assumptions
-----------
Every agent class registered here follows roughly the same contract:
- constructor accepts: env=..., seed=..., and the algorithm kwargs
- exposes reset_env(seed=...)
- exposes play(max_steps=..., seed=...)
- exposes final_state.cumulative_reward

If one algorithm does not follow this contract, do NOT modify the runner.
Instead, add a small adapter/factory for that algorithm.
"""


"""
EXAMPLE RUN:

From source of the repo ensure there is a results folder

source .venv/bin/activate

PYTHONPATH=. python Optuna_LunarLander_Runner.py \
  --algorithms dueling_det_dqn \
  --study-name lunar_benchmark \
  --db-path results/optuna_rl.db \
  --n-trials 100 \
  --runs-per-trial 3 \
  --episodes 500 \
  --max-steps 1000 \
  --top-k 10 \
  --gravity -10.0 \
  --enable-wind false \
  --wind-power 0.0 \
  --turbulence-power 0.0



python rl_optuna_runner.py \
  --algorithms all \
  --study-name lunar_benchmark \
  --db-path results/optuna_rl.db \
  --n-trials 200 \
  --runs-per-trial 5 \
  --episodes 600 \
  --top-k 20
"""

import argparse
import importlib
import numpy as np
import torch.nn as nn
import torch.optim as optim
import json
import math
import sqlite3
from dataclasses import asdict, dataclass, field
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Callable, Dict, Iterable, Mapping, Protocol

import gymnasium as gym
import optuna
from optuna.pruners import MedianPruner, NopPruner
from optuna.samplers import TPESampler
from optuna.trial import Trial
from optuna.trial import TrialState

from SimpleBaselines.agent.rl_agents.DuelingDetDQN_RL_Agent import DuelingDetDQN_RL_Agent
from SimpleBaselines.states.State import State


# ---------------------------------------------------------------------------
# Protocols and core data structures
# ---------------------------------------------------------------------------

class SearchSpaceFn(Protocol):
    """Callable contract for algorithm-specific Optuna search spaces."""

    def __call__(self, trial: Trial) -> Dict[str, Any]:
        ...
        
        
@dataclass(frozen=True)
class EnvConfig:
    """Environment settings that must remain constant across experiments."""
    env_id: str = "LunarLander-v3"
    continuous: bool = False
    gravity: float = -10.0
    enable_wind: bool = False
    wind_power: float = 15.0
    turbulence_power: float = 1.5

    def __post_init__(self) -> None:
        # LunarLander-specific validation.
        # These bounds are intentionally conservative and reflect Gymnasium docs.
        if self.env_id == "LunarLander-v3":
            if not (-12.0 <= self.gravity <= 0.0):
                raise ValueError("For LunarLander-v3, gravity must be in [-12.0, 0.0].")
            if self.wind_power < 0.0:
                raise ValueError("wind_power must be >= 0.0.")
            if self.turbulence_power < 0.0:
                raise ValueError("turbulence_power must be >= 0.0.")

    def make_env(self) -> gym.Env:
        """Factory method. Keeps env creation in one place."""
        return gym.make(
            self.env_id,
            continuous=self.continuous,
            gravity=self.gravity,
            enable_wind=self.enable_wind,
            wind_power=self.wind_power,
            turbulence_power=self.turbulence_power,
        )


@dataclass(frozen=True)
class TrainConfig:
    """Training/evaluation settings shared by all algorithms."""
    episodes: int = 500
    max_steps: int = 1000
    runs_per_trial: int = 3
    base_seed: int = 42
    report_every: int = 25


@dataclass(frozen=True)
class StudyConfig:
    """Optuna/study settings shared by all algorithms."""
    db_path: str = "optuna_rl.db"
    study_name: str = "lunar_lander"
    n_trials: int = 100
    timeout_seconds: int | None = None
    top_k: int = 10
    direction: str = "maximize"
    use_pruner: bool = True

    @property
    def storage_url(self) -> str:
        # sqlite:///relative.db is enough for SQLAlchemy/Optuna
        return f"sqlite:///{self.db_path}"


SearchSpaceFn = Callable[[optuna.trial.Trial], Dict[str, Any]]
ParamResolverFn = Callable[[Mapping[str, Any]], Dict[str, Any]]

@dataclass(frozen=True)
class AlgorithmSpec:
    name: str
    agent_import_path: str
    search_space: SearchSpaceFn
    fixed_kwargs: Dict[str, Any] = field(default_factory=dict)
    resolve_params: ParamResolverFn | None = None

    def agent_cls(self) -> type:
        module_name, class_name = self.agent_import_path.rsplit(".", 1)
        module = importlib.import_module(module_name)
        return getattr(module, class_name)

    def build_agent(self, env: gym.Env, seed: int, agent_params: Mapping[str, Any],) -> Any:
        """
        Build the agent from already-resolved parameters.
        """
        agent_class = self.agent_cls()
        kwargs = {
            "env": env,
            "seed": seed,
            **self.fixed_kwargs,
            **dict(agent_params),
        }
        return agent_class(**kwargs)


@dataclass(frozen=True)
class SummaryRow:
    """One row for the final results table."""
    study_name: str
    algorithm: str
    best_value: float | None
    top_k: int
    top_k_mean: float | None
    top_k_std: float | None
    completed_trials: int
    best_trial_number: int | None
    best_params_json: str | None
    
    
    

# ---------------------------------------------------------------------------
# Algorithm-specific search spaces
# Add more functions like these for the rest of your library.
# ---------------------------------------------------------------------------

"""
TODO
"""


"""
DUELING DETERMINISTIC DQN
"""

HIDDEN_LAYER_OPTIONS = {
    "64": [64],
    "128": [128],
    "256": [256],
    "64x64": [64, 64],
    "128x128": [128, 128],
    "256x256": [256, 256],
    "64x64x64": [64, 64, 64],
    "128x128x128": [128, 128, 128],
}

def decode_best_params(algorithm_name: str, best_params: dict) -> dict:
    params = dict(best_params)

    if algorithm_name == "dueling_det_dqn":
        params["hidden_layers_size"] = HIDDEN_LAYER_OPTIONS[params["hidden_layers_size"]]

    return params


def resolve_dueling_det_dqn_params(sampled_params: Mapping[str, Any]) -> Dict[str, Any]:
    params = dict(sampled_params)

    # Decode the storage-safe categorical value into the real architecture.
    hidden_layers_key = params["hidden_layers_size"]
    params["hidden_layers_size"] = HIDDEN_LAYER_OPTIONS[hidden_layers_key]

    return params



def dueling_det_dqn_search_space(trial: Trial) -> Dict[str, Any]:
    """
    Search space for DuelingDetDQN. This contains ONLY tuned hyperparameters.
    Constants belong in fixed_kwargs inside the registry.
    """

    return {
        "gamma": trial.suggest_float("gamma", 0.90, 0.999),
        "nn_learning_rate": trial.suggest_float("nn_learning_rate", 1e-5, 1e-2, log=True),
        "egreedy": trial.suggest_float("egreedy", 0.80, 1.0),
        "egreedy_final": trial.suggest_float("egreedy_final", 0.001, 0.05),
        "egreedy_decay": trial.suggest_int("egreedy_decay", 500, 5000),
        "hidden_layers_size": trial.suggest_categorical(
            "hidden_layers_size",
            tuple(HIDDEN_LAYER_OPTIONS.keys()),   # strings only
        ),
        "dropout": trial.suggest_float("dropout", 0.0, 0.5),
        "memory_size": trial.suggest_int("memory_size", 1000, 10000),
        "batch_size": trial.suggest_int("batch_size", 16, 128),
        "target_net_update_steps": trial.suggest_int("target_net_update_steps", 100, 1000),
    }


"""
General Algorithm Registry
"""

ALGORITHM_REGISTRY: Dict[str, AlgorithmSpec] = {
    "dueling_det_dqn": AlgorithmSpec(
        name="dueling_det_dqn",
        agent_import_path=(
            "SimpleBaselines.agent.rl_agents.DuelingDetDQN_RL_Agent."
            "DuelingDetDQN_RL_Agent"
        ),
        search_space=dueling_det_dqn_search_space,
        resolve_params=resolve_dueling_det_dqn_params,
        fixed_kwargs={
            "activation_fn": nn.Tanh,
            "use_batch_norm": False,
            "loss_fn": nn.MSELoss,
            "optimizer": optim.Adam,
            "clip_error": True,
        },
    ),
}


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def str2bool(value: str) -> bool:
    """Robust boolean parser for argparse."""
    if isinstance(value, bool):
        return value
    normalized = value.strip().lower()
    if normalized in {"true", "1", "yes", "y", "on"}:
        return True
    if normalized in {"false", "0", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Cannot parse boolean value from: {value!r}")


def resolve_algorithms(requested: Iterable[str]) -> list[AlgorithmSpec]:
    """
    Convert CLI names into registry entries.

    Supports:
    - --algorithms all
    - --algorithms algo1 algo2 ...
    """
    names = list(requested)
    if len(names) == 1 and names[0].lower() == "all":
        return list(ALGORITHM_REGISTRY.values())

    missing = [name for name in names if name not in ALGORITHM_REGISTRY]
    if missing:
        available = ", ".join(sorted(ALGORITHM_REGISTRY.keys()))
        raise ValueError(
            f"Unknown algorithms: {missing}. Available algorithms: {available}"
        )

    return [ALGORITHM_REGISTRY[name] for name in names]


def make_study_name(base_study_name: str, algorithm_name: str) -> str:
    """
    One DB, one study per algorithm.

    Example:
        lunar_lander__dueling_det_dqn
        lunar_lander__double_dqn
    """
    return f"{base_study_name}__{algorithm_name}"


def create_or_load_study(
    study_name: str,
    study_cfg: StudyConfig,
    train_cfg: TrainConfig,
) -> optuna.Study:
    """
    Persistent Optuna study in SQLite.

    load_if_exists=True lets you resume interrupted experiments.
    """
    pruner = (
        MedianPruner(
            n_startup_trials=5,
            n_warmup_steps=train_cfg.report_every,
            interval_steps=train_cfg.report_every,
            n_min_trials=3,
        )
        if study_cfg.use_pruner
        else NopPruner()
    )

    sampler = TPESampler(
        seed=train_cfg.base_seed,
        multivariate=True,
    )

    study = optuna.create_study(
        study_name=study_name,
        storage=study_cfg.storage_url,
        direction=study_cfg.direction,
        sampler=sampler,
        pruner=pruner,
        load_if_exists=True,
    )
    return study


def run_one_seed(spec: AlgorithmSpec, env_cfg: EnvConfig, train_cfg: TrainConfig, agent_params: Mapping[str, Any], seed: int, trial: Trial | None = None, run_index: int = 0,) -> float:
    env = env_cfg.make_env()
    rewards: list[float] = []

    try:
        agent = spec.build_agent(env=env, seed=seed, agent_params=agent_params, )

        for episode in range(train_cfg.episodes):
            episode_seed = seed + episode

            agent.reset_env(seed=episode_seed)
            agent.play(max_steps=train_cfg.max_steps, seed=episode_seed)

            reward = float(agent.final_state.cumulative_reward)
            rewards.append(reward)

            if trial is not None and (episode + 1) % train_cfg.report_every == 0:
                global_step = run_index * train_cfg.episodes + episode
                running_mean = float(np.mean(rewards))
                trial.report(running_mean, step=global_step)

                if trial.should_prune():
                    raise optuna.TrialPruned(
                        f"Pruned {spec.name} at run={run_index}, episode={episode + 1}"
                    )

        return float(np.mean(rewards))

    finally:
        env.close()


def build_objective(spec: AlgorithmSpec, env_cfg: EnvConfig, train_cfg: TrainConfig,) -> Callable[[Trial], float]:

    def objective(trial: Trial) -> float:
        sampled_params = spec.search_space(trial)

        agent_params = (
            spec.resolve_params(sampled_params)
            if spec.resolve_params is not None
            else dict(sampled_params)
        )

        trial.set_user_attr("algorithm", spec.name)
        trial.set_user_attr("env_id", env_cfg.env_id)
        trial.set_user_attr("env_config", asdict(env_cfg))
        trial.set_user_attr("train_config", asdict(train_cfg))
        trial.set_user_attr("sampled_params", dict(sampled_params))
        trial.set_user_attr("agent_params", dict(agent_params))

        per_run_scores: list[float] = []

        for run_index in range(train_cfg.runs_per_trial):
            seed = train_cfg.base_seed + run_index

            score = run_one_seed(
                spec=spec,
                env_cfg=env_cfg,
                train_cfg=train_cfg,
                agent_params=agent_params,
                seed=seed,
                trial=trial,
                run_index=run_index,
            )
            per_run_scores.append(score)
            trial.set_user_attr(f"run_{run_index}_score", score)

        score_mean = float(np.mean(per_run_scores))
        score_std = float(np.std(per_run_scores)) if len(per_run_scores) > 1 else 0.0

        trial.set_user_attr("score_mean_across_runs", score_mean)
        trial.set_user_attr("score_std_across_runs", score_std)

        return score_mean

    return objective


def completed_trials_sorted(study: optuna.Study, direction: str) -> list[optuna.trial.FrozenTrial]:
    """
    Return only completed trials sorted by objective value.
    """
    trials = [
        t for t in study.get_trials(deepcopy=False)
        if t.state == TrialState.COMPLETE and t.value is not None
    ]
    reverse = direction == "maximize"
    trials.sort(key=lambda t: float(t.value), reverse=reverse)
    return trials


def summarize_study(study: optuna.Study, study_cfg: StudyConfig, algorithm_name: str,) -> SummaryRow:
    """
    Build a compact result row from the study.
    top_k_mean/top_k_std are computed directly from the best K completed trial values.
    """

    trials = completed_trials_sorted(study, study_cfg.direction)
    top_trials = trials[: study_cfg.top_k]

    if not trials:
        return SummaryRow(
            study_name=study.study_name,
            algorithm=algorithm_name,
            best_value=None,
            top_k=study_cfg.top_k,
            top_k_mean=None,
            top_k_std=None,
            completed_trials=0,
            best_trial_number=None,
            best_params_json=None,
        )

    top_values = [float(t.value) for t in top_trials]
    best_trial = trials[0]

    return SummaryRow(
        study_name=study.study_name,
        algorithm=algorithm_name,
        best_value=float(best_trial.value),
        top_k=study_cfg.top_k,
        top_k_mean=float(mean(top_values)),
        top_k_std=float(pstdev(top_values)) if len(top_values) > 1 else 0.0,
        completed_trials=len(trials),
        best_trial_number=int(best_trial.number),
        best_params_json=json.dumps(best_trial.params, ensure_ascii=False, sort_keys=True),
    )


def ensure_reporting_table(db_path: str) -> None:
    """
    Create an extra reporting table in the same SQLite file used by Optuna.

    Optuna keeps its own schema. We add one table for convenient leaderboard-like summaries.
    """

    db_file = Path(db_path)
    db_file.parent.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(db_file) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS experiment_summary (
                study_name TEXT NOT NULL,
                algorithm TEXT NOT NULL,
                best_value REAL,
                top_k INTEGER NOT NULL,
                top_k_mean REAL,
                top_k_std REAL,
                completed_trials INTEGER NOT NULL,
                best_trial_number INTEGER,
                best_params_json TEXT,
                PRIMARY KEY (study_name)
            )
            """
        )
        conn.commit()


def save_summary_row(db_path: str, row: SummaryRow) -> None:
    """
    Upsert one result row into the reporting table.
    """
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO experiment_summary (
                study_name,
                algorithm,
                best_value,
                top_k,
                top_k_mean,
                top_k_std,
                completed_trials,
                best_trial_number,
                best_params_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(study_name) DO UPDATE SET
                algorithm = excluded.algorithm,
                best_value = excluded.best_value,
                top_k = excluded.top_k,
                top_k_mean = excluded.top_k_mean,
                top_k_std = excluded.top_k_std,
                completed_trials = excluded.completed_trials,
                best_trial_number = excluded.best_trial_number,
                best_params_json = excluded.best_params_json
            """,
            (
                row.study_name,
                row.algorithm,
                row.best_value,
                row.top_k,
                row.top_k_mean,
                row.top_k_std,
                row.completed_trials,
                row.best_trial_number,
                row.best_params_json,
            ),
        )
        conn.commit()


def print_summary_table(rows: list[SummaryRow]) -> None:
    """
    Lightweight terminal table. No pandas dependency needed.
    """
    if not rows:
        print("No results.")
        return

    headers = [
        "study_name",
        "algorithm",
        "best_value",
        "top_k",
        "top_k_mean",
        "top_k_std",
        "completed_trials",
        "best_trial_number",
    ]

    data = [
        [
            row.study_name,
            row.algorithm,
            f"{row.best_value:.4f}" if row.best_value is not None else "NA",
            str(row.top_k),
            f"{row.top_k_mean:.4f}" if row.top_k_mean is not None else "NA",
            f"{row.top_k_std:.4f}" if row.top_k_std is not None else "NA",
            str(row.completed_trials),
            str(row.best_trial_number) if row.best_trial_number is not None else "NA",
        ]
        for row in rows
    ]

    widths = []
    for col_idx, header in enumerate(headers):
        widths.append(max(len(header), *(len(r[col_idx]) for r in data)))

    def fmt(row_values: list[str]) -> str:
        return " | ".join(value.ljust(widths[i]) for i, value in enumerate(row_values))

    print("\n" + fmt(headers))
    print("-+-".join("-" * w for w in widths))
    for row in data:
        print(fmt(row))



# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_experiments(algorithms: list[AlgorithmSpec], env_cfg: EnvConfig, train_cfg: TrainConfig, study_cfg: StudyConfig,) -> list[SummaryRow]:

    """
    Main orchestration function.
    """
    ensure_reporting_table(study_cfg.db_path)

    rows: list[SummaryRow] = []

    for spec in algorithms:
        study_name = make_study_name(study_cfg.study_name, spec.name)
        study = create_or_load_study(study_name, study_cfg, train_cfg)

        # Study metadata helps later when inspecting the DB or dashboard.
        study.set_user_attr("algorithm", spec.name)
        study.set_user_attr("env_config", asdict(env_cfg))
        study.set_user_attr("train_config", asdict(train_cfg))
        study.set_user_attr("study_config", asdict(study_cfg))

        objective = build_objective(spec, env_cfg, train_cfg)

        print(f"\n=== Running study: {study_name} ===")
        study.optimize(
            objective,
            n_trials=study_cfg.n_trials,
            timeout=study_cfg.timeout_seconds,
            gc_after_trial=True,
            show_progress_bar=True,
            n_jobs=1,  # keep sequential if you care about stronger reproducibility
        )

        row = summarize_study(study, study_cfg, spec.name)
        save_summary_row(study_cfg.db_path, row)
        rows.append(row)

        print(f"Best params for {spec.name}:")
        print(study.best_params if row.best_value is not None else "No completed trials")

    return rows



# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generic Optuna runner for multiple RL algorithms."
    )

    # Which algorithms to run
    parser.add_argument(
        "--algorithms",
        nargs="+",
        default=["dueling_det_dqn"],
        help=(
            "Algorithms to run. Use 'all' to run all registry entries. "
            f"Available: {', '.join(sorted(ALGORITHM_REGISTRY.keys()))}"
        ),
    )

    # Study / DB settings
    parser.add_argument("--study-name", type=str, default="lunar_lander")
    parser.add_argument("--db-path", type=str, default="optuna_rl.db")
    parser.add_argument("--n-trials", type=int, default=100)
    parser.add_argument("--timeout-seconds", type=int, default=None)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--direction", choices=["maximize", "minimize"], default="maximize")
    parser.add_argument("--use-pruner", type=str2bool, default=True)

    # Training settings
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--runs-per-trial", type=int, default=3)
    parser.add_argument("--base-seed", type=int, default=42)
    parser.add_argument("--report-every", type=int, default=25)

    # Environment settings
    parser.add_argument("--env-id", type=str, default="LunarLander-v3")
    parser.add_argument("--continuous", type=str2bool, default=False)
    parser.add_argument("--gravity", type=float, default=-10.0)
    parser.add_argument("--enable-wind", type=str2bool, default=False)
    parser.add_argument("--wind-power", type=float, default=15.0)
    parser.add_argument("--turbulence-power", type=float, default=1.5)

    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    algorithms = resolve_algorithms(args.algorithms)

    env_cfg = EnvConfig(
        env_id=args.env_id,
        continuous=args.continuous,
        gravity=args.gravity,
        enable_wind=args.enable_wind,
        wind_power=args.wind_power,
        turbulence_power=args.turbulence_power,
    )

    train_cfg = TrainConfig(
        episodes=args.episodes,
        max_steps=args.max_steps,
        runs_per_trial=args.runs_per_trial,
        base_seed=args.base_seed,
        report_every=args.report_every,
    )

    study_cfg = StudyConfig(
        db_path=args.db_path,
        study_name=args.study_name,
        n_trials=args.n_trials,
        timeout_seconds=args.timeout_seconds,
        top_k=args.top_k,
        direction=args.direction,
        use_pruner=args.use_pruner,
    )

    rows = run_experiments(
        algorithms=algorithms,
        env_cfg=env_cfg,
        train_cfg=train_cfg,
        study_cfg=study_cfg,
    )
    print_summary_table(rows)


if __name__ == "__main__":
    main()
