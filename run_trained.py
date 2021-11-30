from cartpole_deep_Q import CombinedNetwork, ACTIONS, HIDDEN
import tensorflow as tf
import numpy as np
import gym

def get_action(logits: tf.Tensor) -> int:
    return int(tf.argmax(tf.nn.softmax(logits[0])))

def get_state(state: np.ndarray) -> tf.Tensor:
    return tf.expand_dims(tf.constant(state, dtype=tf.float32), (0))
    pass

EPISODES = 100

if __name__ == "__main__":    
    model = CombinedNetwork(ACTIONS, HIDDEN)
    env = gym.make("CartPole-v1")
    model.build((1,4))
    model.load_weights('eval-weights')
    for e in range(EPISODES):
        state = env.reset()
        done = False
        rsum = 0
        while not done:
            state = get_state(state)
            action = get_action(model(state)[0])
            state, reward, done, _ = env.step(action)
            rsum = rsum + reward
            env.render()
        print(f"Died at {rsum}")
