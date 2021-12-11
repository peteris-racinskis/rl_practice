from cartpole_Q import *
import numpy as np
INFILE=OUTFILE

if __name__ == "__main__":
    with open(INFILE, 'r') as f:
        model = np.fromfile(f).reshape(QSHAPE + (ACTIONS,))
    xlim = (env.observation_space.low[0], env.observation_space.high[0])
    vlim = (-10,10)
    philim = (env.observation_space.low[2], env.observation_space.high[2])
    wlim = (-5,5)
    discrete = Discretizer((xlim,vlim,philim,wlim),(4,8,4,8))
    tt = 0
    
    for t in range(1000):
        done = False
        obs = env.reset()
        newstate = discrete.map(state(*obs))
        print("died at {}".format(tt))
        tt = 0
        while not done:
            tt = tt + 1
            oldstate = newstate
            action = get_action(model, oldstate)
            obs, reward, done, _ = env.step(action)
            newstate = discrete.map(state(*obs))
            env.render()