from cartpole_Q import Discretizer, QSHAPE, ACTIONS, LIMITS, get_action
from cartpole_Q import OUTFILE as INFILE, env as ENV
import numpy as np
import gym
from typing import Dict, Tuple, List
from tensorflow.keras import Model, layers, losses, optimizers, activations
import tensorflow as tf

class KerasModel(Model):

    def __init__(self, hidden, actions):
        super().__init__()
        self.hidden1 = layers.Dense(hidden, activation=activations.relu)
        self.hidden2 = layers.Dense(hidden, activation=activations.relu)
        self.action = layers.Dense(actions)
    
    def call(self, state: tf.Tensor) -> tf.Tensor:
        return self.action(self.hidden2(self.hidden1(state)))

def read_model() -> np.ndarray:
    return np.fromfile(INFILE).reshape(QSHAPE + (ACTIONS,))

def sample_expert(model: np.ndarray, episodes) -> Tuple[np.ndarray, np.ndarray]:
    discrete = Discretizer(LIMITS,model.shape)
    states, actions = [], []
    for _ in range(episodes):
        if _ % 50 == 0:
            print(f"expert episode {_}")
        done = False
        obs = ENV.reset()
        while not done:
            state = discrete.map(obs)
            action = get_action(model, state)
            states.append(obs)
            actions.append(action)
            obs, __, done, ___ = ENV.step(action)
            #ENV.render()
    return np.asarray(states), np.asarray(actions, dtype=np.float64)

def train_policy(states: np.ndarray, actions: np.ndarray) -> Model:
    t_states = tf.convert_to_tensor(states)
    t_actions = tf.convert_to_tensor(actions, dtype=tf.float64)
    loss = losses.Huber(reduction=losses.Reduction.SUM)
    model = KerasModel(HIDDEN, ACTIONS)
    model.compile(
        optimizer=optimizers.Adam(),
        loss=losses.mean_squared_error,
        run_eagerly=EAGER)
    model.fit(t_states, t_actions,epochs=10)
    return model


NUM_EPISODES=5000
OUTFILE="cloned-upd"
HIDDEN=256
EAGER=False
if __name__ == "__main__":
    expert = read_model()
    states, actions = sample_expert(expert, NUM_EPISODES)
    model = train_policy(states, actions)
    model.save_weights(OUTFILE)