from cartpole_Q import QSHAPE
import tensorflow as tf
import numpy as np
from cartpole_DAgger import Policy, HIDDEN, ACTIONS, OUTFILE as INFILE, ENV

NUM_EPISODES = 50
if __name__ == "__main__":
    model = Policy(HIDDEN, ACTIONS)
    model.build((1,len(QSHAPE)))
    model.load_weights(INFILE)
    for e in range(NUM_EPISODES):
        t = 0
        obs = ENV.reset()
        done = False
        while not done:
            t += 1
            action = model.action(obs)
            obs, _, done, __ = ENV.step(action)
            ENV.render()
        print(f"died at {t}")
