# UBC_GameIntelligence: Academic Environments and Agents Library

A student-friendly collection of **academic environments** and **algorithmic agents** for exploring classical AI search and reinforcement learning (RL).
It includes custom Gymnasium-compatible environments such as **River Crossing**, **8-Puzzle**, and **Stock Trading**, plus a set of incremental examples to learn concepts step by step.

**Scope:** educational and limited research purposes.

---

## Dependencies

Install these core libraries inside a virtual environment (see next section):

- `gymnasium`
- `numpy`
- `scipy`
- `matplotlib`
- `pandas`

---

## Installation & Setup

We strongly recommend using a **project-local virtual environment** (`.venv`) to isolate dependencies.

### Linux (and macOS)

```bash
# 1) Make sure you have Python 3
python3 --version

# (Ubuntu/Debian only, if needed)
sudo apt-get install -y python3-venv

# 2) Create a virtual env in the project folder
python3 -m venv .venv

# 3) Activate it (Bash/Zsh/Fish)
source .venv/bin/activate
# Fish shell:
source .venv/bin/activate.fish

# 4) Upgrade pip and install required packages
pip install --upgrade pip
pip install gymnasium numpy scipy matplotlib pandas

# 5) Deactivate when done
deactivate
```

### Windows

```
# 1) Check Python
py --version   # preferred on Windows
# or
python --version

# 2) Create the env (use a project-local ".venv" folder)
py -3 -m venv .venv
# or
python -m venv .venv

# 3) Activate
# PowerShell:
.\.venv\Scripts\Activate.ps1
# Command Prompt (cmd.exe):
.\.venv\Scripts\activate.bat
# Git Bash:
source .venv/Scripts/activate

# 4) Upgrade pip and install required packages
pip install --upgrade pip
pip install gymnasium numpy scipy matplotlib pandas

# 5) Deactivate
deactivate
```
---

### Getting Started

1. Clone the repository

```
git clone <your-repo-url>
cd <your-repo-name>
```

2. Create & activate the virtual environment (see above).

3. Run an example
```
# Example: Blind Search – Breadth-First Search
PYTHONPATH=. python examples/2_blind_search/Breadth_First_Search.py

```

or you can also install SimpleBaselines library with (from inside ./SimpleBaselines folder)
```
pip install -e .
```

or with (from the root of the git repo)

```
pip install -e SimpleBaselines
```

or with (from colab or other online editors)
```
pip install git+https://github.com/ilopezgazpio/UBC_GameIntelligence.git
```


4. Explore the tutorials in examples/ (see the full list below).


## Available Agents

The library includes both **classical blind search agents** and **reinforcement learning agents**.

### Blind Search (Graph) — `blindsearch_graph_agents/`

- `Blind_BFS_Graph_Agent.py` — Breadth-First Search
- `Blind_DFS_Graph_Agent.py` — Depth-First Search
- `Blind_DLS_Graph_Agent.py` — Depth-Limited Search
- `Blind_IDS_Graph_Agent.py` — Iterative Deepening Search
- `Blind_UCS_Graph_Agent.py` — Uniform Cost Search

### Blind Search (Tree) — `blindsearch_tree_agents/`

- `Blind_BFS_Tree_Agent.py` — Breadth-First Search
- `Blind_DFS_Tree_Agent.py` — Depth-First Search
- `Blind_DLS_Tree_Agent.py` — Depth-Limited Search
- `Blind_IDS_Tree_Agent.py` — Iterative Deepening Search
- `Blind_UCS_Tree_Agent.py` — Uniform Cost Search

### Reinforcement Learning — `rl_agents/`

- `EpsilonGreedyQLearning_RL_Agent.py` — Tabular Q-Learning (ε-greedy)
- **DQN variants**
  - `DeterministicDQN_RL_Agent.py`
  - `DoubleDetDQN_RL_Agent.py`
  - `DuelingDetDQN_RL_Agent.py`
  - `ExperienceReplayDetDQN_RL_Agent.py`
  - `MixedExperienceDQN_RL_Agent.py`
  - `DeterministicDQConvN_RL_Agent.py`
  - `MultiDQN_liftoff.py`

---

## Incremental Examples

A curated path of scripts that build concepts progressively.
> **Note:** folders marked `*_todo` are ongoing work and can be ignored for now.

```text
examples/
├── 0_introduction
│   └── first_steps_python.py
├── 1_frontier_expansion
│   ├── frontier_expansion_8Puzzle.py
│   └── frontier_expansion_RiverCrossing.py
├── 2_blind_search
│   ├── Blind_Search_discussion.py
│   ├── Breadth_First_Search.py
│   ├── Depth_First_Search.py
│   ├── Depth_Limited_Search.py
│   ├── Iterative_Deepening_Search.py
│   └── Uniform_Cost_Search.py
├── 3_informed_search_todo
│   └── Informed_Search_discussion.txt
├── 4_local_search_todo
├── 5_Population_based_search_todo
├── 6_MultiObjective_search_todo
├── 7_RL
│   ├── 01_tutorial_cartPole.py
│   ├── 02_tutorial_frozenLakeRandom.py
│   ├── 03_tutorial_deterministicFrozenLakeRandom.py
│   ├── 04_tutorial_deterministicFrozenLake_QLearning.py
│   ├── 05_tutorial_StochasticFrozenLake_QLearning.py
│   ├── 06_tutorial_StochasticFrozenLake_StochasticQLearning.py
│   ├── 07_tutorial_StochasticFrozenLake_EpsilonGreedyQLearning.py
│   ├── 08_tutorial_FrozenLake_valueIteration.py
│   ├── 09_tutorial_taxi_EpsilonGreedyQLearning.py
│   ├── 10_tutorial_cartPole_DeterministicDQN.py
│   ├── 10_tutorial_cartPole_StochasticDQN.py
│   ├── 11_tutorial_pongMakeAtari_DetDQConvNet.py
│   ├── 12_tutorial_cartPole_ExperienceReplayDQN.py
│   ├── 13_tutorial_cartPole_StableTargetNetDQN.py
│   ├── 14_tutorial_cartPole_DoubleDQN.py
│   └── 15_tutorial_cartPole_DuelingDQN.py
└── 8-Dronak
    ├── best_params
    │   └── best_params.json
    ├── lunar_lander_comp.py
    ├── lunar_lander.py
    ├── lunar_lander_stbaselines3.py
    └── optuna_lunarlander.py


### Commands for debug

```
# one-time
python -m pip install --upgrade build

# build sdist + wheel into dist/
python -m build

# install the wheel (what users do)
python -m pip install dist/mylib-0.1.0-py3-none-any.whl

# OR dev/editable install for local hacking
python -m pip install -e .
```






