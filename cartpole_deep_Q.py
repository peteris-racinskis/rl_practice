import tensorflow as tf
from tensorflow.keras import layers, Model
from typing import Tuple, List
import numpy as np
import gym
env = gym.make("CartPole-v1")
# single network for both providing action probabilities and 
# predicting expected rewards of each state
class CombinedNetwork(Model):

    def __init__(self, actions, hidden) -> None:
        super().__init__()

        self.hidden = layers.Dense(hidden,activation="relu")
        self.action = layers.Dense(actions)
        self.reward = layers.Dense(1)

    # override the model call method
    def call(self, state: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        # layers should be callable and produce output tensors
        intermediate = self.hidden(state)
        return self.action(intermediate), self.reward(intermediate)

# get output form single step, wrap inside np arrays
def step(action: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    state, reward, done, _ = env.step(action)
    return (state.astype(np.float32),
            np.asarray(reward, dtype=np.float32),
            np.asarray(done, dtype=np.int32))

# tf wrapper for functions that operate on numpy arrays, allows integration
# with computation graphs
def tf_step(action: tf.Tensor) -> List[tf.Tensor]:
    # takes: func | list of tensors (input) | list of tf data types (output)
    return tf.numpy_function(step, [action], [tf.float32, tf.float32, tf.int32])

def run_episode():
    pass

def expected_reward():
    pass

def update_model():
    pass

def get_loss():
    pass

test = CombinedNetwork(2, 100)
state = tf. convert_to_tensor([1,2,3,4], dtype=np.float64)
state = tf.expand_dims(state, 0) # tensor shape needs at least 2 dims (even if it's just a vector, give it shape (1,n))
(at, Gt) = test.call(state)
x = at.numpy()
print(test(state))
model = None

if __name__ == "__main__":
    for e in range(1000):
        states, actions, rewards = run_episode()
        expected = expected_reward(states, rewards)
        loss = get_loss(states, actions, expected, model)
        update_model(model, loss)
        pass