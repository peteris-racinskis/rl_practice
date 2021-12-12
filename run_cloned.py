from cartpole_Q import QSHAPE
import tensorflow as tf
import numpy as np
from cartpole_clone import KerasModel, HIDDEN, ACTIONS, OUTFILE as INFILE, ENV

def get_action(logits: tf.Tensor) -> int:
    return int(tf.argmax(tf.nn.softmax(logits[0])))

def get_state(state: np.ndarray) -> tf.Tensor:
    return tf.expand_dims(tf.constant(state, dtype=tf.float32), (0))

NUM_EPISODES = 50
if __name__ == "__main__":
    model = KerasModel(HIDDEN, ACTIONS)
    model.build((1,len(QSHAPE)))
    model.load_weights(INFILE)
    for e in range(NUM_EPISODES):
        t = 0
        state = ENV.reset()
        done = False
        while not done:
            t += 1
            state = get_state(state)
            action = get_action(model(state))
            state, _, done, __ = ENV.step(action)
            ENV.render()
        print(f"died at {t}")

