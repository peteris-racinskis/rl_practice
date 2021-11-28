import gym

env = gym.make("CartPole-v1")

def simple_pd(p, _, rot, omega, kp, kd, ki):
    command = round(0.5 + kp*rot + kd*omega + ki*p) 
    return command

# pi : obs -> action
kp = 0.5
kd = 0.1
ki = 0.05
for _ in range(1000):
    done = False
    obs = env.reset()
    while not done:
        action = simple_pd(*obs,kp,kd, ki)
        obs, reward, done, _ = env.step(action)
        env.render()

