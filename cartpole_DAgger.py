import numpy as np
import tensorflow as tf
import gym
from tensorflow.keras import Model, layers, optimizers, losses, activations
from expert import QExpert

class Policy(Model):
    
    def __init__(self, hidden, actions):
        super().__init__()
        # STOP FORGETTING ABOUT THE ACTIVATION FFS
        self.h1 = layers.Dense(hidden, activation=activations.relu)
        self.h2 = layers.Dense(hidden, activation=activations.relu)
        self.h3 = layers.Dense(hidden, activation=activations.relu)
        self.actions = layers.Dense(actions)

    def call(self, x):
        return self.actions(self.h3(self.h2(self.h1(x))))

    def action(self, obs):
        x = tf.expand_dims(tf.convert_to_tensor(obs),0)
        return int(tf.argmax(tf.nn.softmax(tf.squeeze(self.call(x)))))

def train_policy(expert: QExpert):
    states, actions = expert.sample(NUM_DEMONSTRATION)
    model = Policy(HIDDEN, ACTIONS)
    model.compile(
        optimizer=optimizers.Adam(),
        loss=losses.Huber(),
        run_eagerly=EAGER
    )
    model.fit(states, actions, epochs=DEMO_EPOCHS)
    for i in range(NUM_TRAINING_STEPS):
        print(f"Training step {i+1} of {NUM_TRAINING_STEPS}, sampling environment...")
        obs_i = []
        actions_i = []
        for j in range(NUM_EPISODES):
            done = False
            obs = ENV.reset()
            obs_j = []
            while not done:
                action = model.action(obs)
                obs, _, done, __ = ENV.step(action)
                obs_j.append(obs)
            actions_j = [expert.get_action(x) for x in obs_j]
            obs_i += obs_j
            actions_i += actions_j
        x1 = states.shape[0]
        states = np.concatenate((states, np.asarray(obs_i)))
        x2 = states.shape[0]
        print(f"dataset size {states.shape[0]} delta {x2-x1}")
        actions = np.concatenate((actions, np.asarray(actions_i, dtype=np.float64)))
        print("Retraining policy")
        model.fit(states,actions,epochs=TRAIN_EPOCHS)
    return model

NUM_DEMONSTRATION=50
NUM_EPISODES=30
NUM_TRAINING_STEPS=100
HIDDEN=512
ACTIONS=2
EAGER=False
DEMO_EPOCHS=7
TRAIN_EPOCHS=1
OUTFILE="dagger-updated"
ENV=gym.make("CartPole-v1")
if __name__ == "__main__":
    expert = QExpert()
    model = train_policy(expert)
    model.save_weights(OUTFILE)