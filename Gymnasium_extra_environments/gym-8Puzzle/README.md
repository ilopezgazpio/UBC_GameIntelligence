# gym-8Puzzle

Gymnasium environment for the 8-puzzle AI problem.

For a complete description of the environment and its rules check [8-Puzzle problem](http://en.wikipedia.org/wiki/Fifteen_puzzle) link on Wikipedia.

## Instalation
From outside the folder, run:
```
pip3 install -e gym-8Puzzle
```
or, from inside the folder, run:
```
cd gym-8Puzzle
pip3 install -e .

```

## Usage
create an instance of the environment with 
```
import gymnasium as gym
env = gym.make('gym_8Puzzle:8Puzzle-v0')
```

Analyzing the environment search space, observation space and action space:
```
env.action_space
env.action_space.n
env.observation_space
env.observation_space.low
env.observation_space.high
```

Internal state can also be accessed:
```
env.env.state
```

Sample the action space:
```
env.action_space.sample()
```

Play with the environment
```
env.reset()
action = env.action_space.sample()
env.render()
env.step(action)
```