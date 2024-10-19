# gym-RiverCrossing

Gymnasium environment for the River Crossing AI problem.

For a complete description of the environment and its rules check [River crossing puzzle](https://en.wikipedia.org/wiki/River_crossing_puzzle#:~:text=A%20river%20crossing%20puzzle%20is,may%20be%20safely%20left%20together) link on Wikipedia.

## Instalation
From outside the folder, run:
```
pip3 install -e gym-RiverCrossing
```
or, from inside the folder, run:
```
cd gym-RiverCrossing
pip3 install -e .

```

## Usage
create an instance of the environment with 
```
import gymnasium as gym
env = gym.make('gym_RiverCrossing:RiverCrossing-v0')
```
Analyzing the environment search space, observation space and action space:
```
env.action_space
env.action_space.n
env.observation_space
env.observation_space.low
env.observation_space.high
```
Only for debugging purposes, the internal state can also be accessed (we need to go through all wrappers):
```
env.env.env.env.state
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