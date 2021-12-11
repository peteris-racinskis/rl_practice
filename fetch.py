import numpy as np
import gym


env = gym.make("FetchPickAndPlace-v1")
obs = env.reset()
done = False

def policy(prev):
    # Here you would implement your smarter policy. In this case,
    # we just sample random actions.
    return np.asarray([0,0,0,-0.005*prev])


test = policy(1)
t=0
prev=1
while not done:
    t+=1
    if t % 10 == 0:
        prev = -prev
    action = policy(prev)
    obs, reward, done, _ = env.step(action)
    env.render()

print()