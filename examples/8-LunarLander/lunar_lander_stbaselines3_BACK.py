#!/usr/bin/env python3
# Usage: Launch from the root directory of the repository

import os
import time
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

import stable_baselines3 as sb3
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback, ProgressBarCallback
from stable_baselines3.common.utils import set_random_seed


# -------------------------
# Hyperparameters / config
# -------------------------
SEED = 123
N_TIMESTEPS = 100_000
EVAL_EPISODES = 50

LOG_DIR = "./logs/dqn_lunar"
TB_NAME = "DQN_LunarLander"


parameters = {
    "batch_size": 128,
    "buffer_size": 50_000,
    "exploration_final_eps": 0.1,
    "exploration_fraction": 0.12,
    "gamma": 0.99,
    "gradient_steps": -1,
    "learning_rate": 0.00063,
    "learning_starts": 0,
    "policy_kwargs": dict(net_arch=[256, 256]),
    "target_update_interval": 250,
    "train_freq": 4,

    # Logging knobs:
    "verbose": 1,  # set to 2 for even more console logs
    "tensorboard_log": LOG_DIR,
}


# -------------------------
# Custom logging callback
# -------------------------
class InnerTrainingLogger(BaseCallback):
    """
    Logs training internals frequently:
    - exploration rate (epsilon)
    - replay buffer size
    - episode reward/length (requires Monitor)
    - any SB3-reported metrics (loss, q_values, etc.) via self.logger
    """

    def __init__(self, log_every_steps: int = 1000):
        super().__init__()
        self.log_every_steps = log_every_steps
        self.start_time = None

    def _on_training_start(self) -> None:
        self.start_time = time.time()
        print(f"[Callback] Training started. Logging every {self.log_every_steps} steps.")

    def _on_step(self) -> bool:
        # Log periodic step stats
        if self.num_timesteps % self.log_every_steps == 0:
            elapsed = max(1e-9, time.time() - self.start_time)
            fps = int(self.num_timesteps / elapsed)

            # exploration_rate exists for DQN (and some other algos)
            exploration_rate = getattr(self.model, "exploration_rate", np.nan)

            # replay buffer size (DQN has replay_buffer)
            rb_size = np.nan
            if hasattr(self.model, "replay_buffer") and self.model.replay_buffer is not None:
                rb_size = self.model.replay_buffer.size()

            print(
                f"[Step {self.num_timesteps:>8}] fps={fps:<4} "
                f"eps={exploration_rate:.4f} replay={rb_size}"
            )

            # Also record into TensorBoard
            self.logger.record("custom/fps", fps)
            self.logger.record("custom/exploration_rate", float(exploration_rate))
            if not np.isnan(rb_size):
                self.logger.record("custom/replay_buffer_size", float(rb_size))

        return True

    def _on_rollout_end(self) -> None:
        # Monitor wrapper adds "episode" info into infos; SB3 aggregates into ep_info_buffer.
        # We can print a rolling mean when available:
        if len(self.model.ep_info_buffer) > 0:
            ep_rew_mean = np.mean([ep["r"] for ep in self.model.ep_info_buffer])
            ep_len_mean = np.mean([ep["l"] for ep in self.model.ep_info_buffer])
            # record to TB as well
            self.logger.record("custom/ep_rew_mean_window", float(ep_rew_mean))
            self.logger.record("custom/ep_len_mean_window", float(ep_len_mean))


# -------------------------
# Env factory
# -------------------------
def make_env():
    env = gym.make(
        "LunarLander-v3",
        continuous=False,
        gravity=-10.0,
        enable_wind=False,
        wind_power=15.0,
        turbulence_power=1.5,
    )
    env = Monitor(env)  # IMPORTANT: enables episode reward/len logging
    return env


def evaluate(agent, env, n_episodes: int = 20):
    rewards = []
    for _ in range(n_episodes):
        obs, info = env.reset()
        terminated = truncated = False
        ep_reward = 0.0
        while not (terminated or truncated):
            action, _ = agent.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += float(reward)
        rewards.append(ep_reward)
    return np.array(rewards, dtype=np.float32)


def main():
    os.makedirs(LOG_DIR, exist_ok=True)

    # Create envs
    env = make_env()
    env_test = make_env()

    # Reproducibility
    set_random_seed(SEED)
    try:
        env.reset(seed=SEED)
        env.action_space.seed(SEED)
        env.observation_space.seed(SEED)
    except TypeError:
        pass

    # Quick sanity check (catches many subtle issues)
    print("[Info] Checking env API with SB3 env checker...")
    check_env(env.unwrapped, warn=True)

    # Analyze the environment
    print("Observation space:", env.observation_space)
    print("Observation space sample:", env.observation_space.sample())
    print("Action space:", env.action_space)
    print("Action space sample:", env.action_space.sample())

    # Create the agent
    agent = sb3.DQN(
        policy="MlpPolicy",
        env=env,
        **parameters,
    )

    # Evaluation callback (runs during training)
    eval_callback = EvalCallback(
        env_test,
        best_model_save_path=os.path.join(LOG_DIR, "best_model"),
        log_path=os.path.join(LOG_DIR, "eval"),
        eval_freq=10_000,          # evaluate every N steps
        n_eval_episodes=10,
        deterministic=True,
        render=False,
    )

    callbacks = [
        ProgressBarCallback(),                 # visual progress bar in console
        InnerTrainingLogger(log_every_steps=2000),
        eval_callback,
    ]

    # Train
    agent.learn(
        total_timesteps=N_TIMESTEPS,
        tb_log_name=TB_NAME,
        log_interval=10,  # how often SB3 prints its own logs (in rollouts)
        callback=callbacks,
        progress_bar=True,  # (newer SB3) also enables a progress bar
    )

    # Final test run
    print("[Info] Final evaluation...")
    rewards = evaluate(agent, env_test, n_episodes=EVAL_EPISODES)
    print("Mean reward:", float(np.mean(rewards)))
    print("Std reward:", float(np.std(rewards)))

    # Plot rewards
    plt.plot(rewards)
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Evaluation Rewards")
    plt.show()

    env.close()
    env_test.close()


if __name__ == "__main__":
    main()

