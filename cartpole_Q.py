import gym
import math
import numpy as np
from typing import Tuple, List
OUTFILE="model.txt"
env = gym.make("CartPole-v1")

class Discretizer:

    def __init__(self, bounds: Tuple[Tuple], steps: Tuple[int]):
        self.bounds = bounds
        self.steps = steps
    
    def index(self, value, low, high, steps):
        lim_norm = high - low
        val_norm = value - low
        l = int(val_norm * steps // lim_norm)
        return l

    def map(self, values: Tuple) -> Tuple[int]:
        point = [self.index(values[i],*self.bounds[i],self.steps[i]) for i in range(len(values))]
        return tuple(point)

    def get_zeros(self, dim: tuple):
        return np.zeros(self.steps + dim)

def get_action(qmat: np.ndarray, state: tuple) -> int:
    return np.argmax(qmat[state]) # 0 or 1 with max reward

# This together with the expected reward is known as the BELLMAN EQUATION
def update_reward(qmat, expected_reward, oldstate, action, learn_rate):
    index = oldstate + (action,)
    qmat[index] = (1-learn_rate) * qmat[index] + learn_rate*expected_reward

# If discount rate == 1, rewards propagate backwards directly. 
# Set discount rate < 1 to weigh them in time
def expected_reward(qmat, newstate, reward, discount_rate=1):
    updated = reward + discount_rate * np.max(qmat[newstate])
    return updated

def get_rate(t: int, gamma: float, min_rate = 0.01, init = 1) -> float:
    return max(min_rate, min(init, math.exp(-gamma*t)))

def randomize(space, prob, action):
    if np.random.rand(1) < prob:
        action = space.sample()
    return action

def state(x, v, rot, omega) -> tuple:
    return (x, v, rot, omega)

ACTIONS=2
QSHAPE=(4,8,4,8)
LIMITS = ((env.observation_space.low[0], env.observation_space.high[0]),
            (-10,10),
            (env.observation_space.low[2], env.observation_space.high[2]),
            (-5,5))

if __name__ == "__main__":
    discrete = Discretizer(LIMITS,QSHAPE)
    model = discrete.get_zeros((2,))    
    for t in range(4000):
        learn_rate = get_rate(t, 0.001)
        explo_rate = get_rate(t, 0.001)
        done = False
        obs = env.reset()
        newstate = discrete.map(state(*obs))
        tt = 0
        if t % 50 == 0:
            print("iteration {}".format(t))
        while not done and tt < 20000:
            tt = tt + 1
            oldstate = newstate
            action = randomize(env.action_space, explo_rate, get_action(model, oldstate))
            obs, reward, done, _ = env.step(action)
            newstate = discrete.map(state(*obs))
            expected = expected_reward(model,newstate,reward,0.98)
            update_reward(model,expected,oldstate,action,learn_rate)
    with open(OUTFILE, 'w') as f:
        model.tofile(f)
    # benchmark
    rsums = []
    for t in range(100):
        done = False
        rsum = 0
        obs = env.reset()
        newstate = discrete.map(state(*obs))
        while not done:
            oldstate = newstate
            action = get_action(model, oldstate)
            obs, reward, done, _ = env.step(action)
            newstate = discrete.map(state(*obs))
            rsum = rsum + reward
        rsums.append(rsum)
    print("Evaluated over 100 episodes: {}".format(np.average(rsums)))
