import numpy as np
import gym
from cartpole_Q import QSHAPE, ACTIONS, OUTFILE as INFILE
from discretizer import Discretizer
from typing import Tuple

class QExpert():

    def __init__(self, infile: str = INFILE, state_shape=QSHAPE,
     action_shape=ACTIONS, env="CartPole-v1"):
        self.env = gym.make(env)
        self.model = self._read_model(infile, state_shape, action_shape)
        limits = (
            (self.env.observation_space.low[0],self.env.observation_space.high[0]),
            (-10,10),
            (self.env.observation_space.low[2], self.env.observation_space.high[2]),
            (-5,5))
        self.disc = Discretizer(limits, self.model.shape)
        pass

    def _read_model(self, infile, states, actions):
        return np.fromfile(infile).reshape(states + (actions,))

    def get_action(self, obs: tuple) -> int:
        state = self.disc.map(obs)
        return np.argmax(self.model[state])

    def sample(self, episodes: int) -> Tuple[np.ndarray, np.ndarray]:
        states, actions = [], []
        for _ in range(episodes):
            if _ % 50 == 0:
                print(f"expert episode {_} of {episodes}")
            done = False
            obs = self.env.reset()
            while not done:
                action = self.get_action(obs)
                states.append(obs)
                actions.append(action)
                obs, __, done, ___ = self.env.step(action)
        return np.asarray(states), np.asarray(actions, dtype=np.float64)