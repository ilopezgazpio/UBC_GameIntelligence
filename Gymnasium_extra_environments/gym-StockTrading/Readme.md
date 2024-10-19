# gym-StockTrading

Openai-gym environment for Stock Trading. The environment can make use of personalized stock trading data.

## Instalation
From outside the folder, run:
```
pip3 install -e gym-StockTrading
```
or, from inside the folder, run:
```
cd gym-StockTrading
pip3 install -e .
```

## Usage
create an instance of the environment with 
```
import gymnasium as gym
env = gym.make('gym_StockTrading:StockTrading-v0')
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
env.env.state
```

Sample the action space:
```
env.action_space.sample()
```

Play with the environment
```
env.reset()
env.render()
```

You can find a main.py file in the gym-StockTrading folder that shows how to use the environment with stable-baselines3.