import tensorflow as tf
from tensorflow.keras import layers, Model, optimizers, losses
from typing import Tuple, List
import statistics
import collections
from tensorflow.python.ops.manip_ops import roll
import tqdm
import numpy as np
import gym
from comparison import compute_loss, get_expected_return


# single network for both providing action probabilities and 
# predicting expected rewards of each state
class CombinedNetwork(Model):

    def __init__(self, actions, hidden) -> None:
        super().__init__()

        self.hidden1 = layers.Dense(hidden,activation=tf.nn.relu)
        self.hidden2 = layers.Dense(hidden,activation=tf.nn.relu)
        self.action = layers.Dense(actions)
        self.reward = layers.Dense(1)

    # override the model call method
    def call(self, state: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        # layers should be callable and produce output tensors
        intermediate = self.hidden2(self.hidden1(state))
        return self.action(intermediate), self.reward(intermediate)

# get output form single step, wrap inside np arrays
def step(action: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    state, reward, done, _ = env.step(action)
    return (state.astype(np.float32),
            np.asarray(reward, dtype=np.int32),
            np.asarray(done, dtype=np.int32))

# tf wrapper for functions that operate on numpy arrays, allows integration
# with computation graphs
def tf_step(action: tf.Tensor) -> List[tf.Tensor]:
    # takes: func | list of tensors (input) | list of tf data types (output)
    return tf.numpy_function(step, [action], [tf.float32, tf.int32, tf.int32])

# decorate with tf.function to compile this (and everything this calls) into a 
# tensorflow graph.
@tf.function
def run_episode(model: CombinedNetwork, init: tf.Tensor, max_steps: int) \
        -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:

    # tf arrays to collect tensors, dynamic size
    # recorded action probs for each state
    action_probs = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
    # recorded reward estimates for each state 
    est_rewards = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
    # recorded actual local rewards (not integrated!) for each state
    act_rewards = tf.TensorArray(dtype=tf.int32, size=0, dynamic_size=True)

    # the shape of state changes from (4,) to (1,4)
    # save old shape for conversion back (the state will be fed as input 
    # in the next iteration!)
    init_shape = init.shape
    state = init

    # use tf iterator rather than a while loop
    # run a simulated episode (till dies or runs out of time)
    for t in tf.range(max_steps):
        # change state shape to fit with the tensor interface
        # NOTE: state is only used here for model estimates, the simulator
        # doesn't need it - it keeps track of it internally
        state = tf.expand_dims(state, 0) # (4,) -> (1,4)
        # compute model output in state
        action_logits, est_reward = model.call(state)
        # draw 1 sample from vector of unnormalized probabilities
        # returns 2d tensor of shape (1,1) so needs 2 indices to access
        # indexing a tensor object still gives me back a tensor object though,
        # but that's ok since the tf-wrapped step function takes tensor args
        action = tf.random.categorical(action_logits,1)[0, 0]
        # normalize output to probability distribution
        action_prob = tf.nn.softmax(action_logits)
        # run simulation step
        # all returns are already cast to tensors!
        state, reward, done = tf_step(action)
        # shape needs reset to be fed back into the outside code
        state.set_shape(init_shape)
        # put the reward estimates and action probabilities into the buffers
        # tf.squeeze removes any dims with value 1, i.e., [1,2,4] -> [2,4]
        est_rewards = est_rewards.write(t, tf.squeeze(est_reward))
        # store the ACTION ACTUALLY TAKEN - was missing this!
        action_probs = action_probs.write(t, action_prob[0, action])
        act_rewards = act_rewards.write(t, reward)
        # check for termination
        # done is a tensor of type int32 so needs boolean cast
        if tf.cast(done, tf.bool):
            break
    # the tensor lists (dynamic size) need to be converted to stacked tensors (fixed)
    action_probs = action_probs.stack()
    est_rewards = est_rewards.stack()
    act_rewards = act_rewards.stack()
    return action_probs, est_rewards, act_rewards

# iterate backwards over state rewards, summing them up
# get size of the vector
def expected_reward(state_rewards: tf.Tensor, gamma: float) -> tf.Tensor:
    n = state_rewards.shape[0]
    # create tensorarray to hold the result
    # need to use TensorArray because tensors are immutable
    returns = tf.TensorArray(dtype=tf.float32, size=n)
    # is reverse/squeeze the issue?
    #flipped = state_rewards[::-1]
    flipped = tf.cast(state_rewards[::-1], dtype=tf.float32)
    accumulator = tf.constant(0, dtype=tf.float32)
    #accumulator_shape = accumulator.shape
    for i in tf.range(n):
        reward = flipped[i]
        accumulator: tf.Tensor = gamma * accumulator + reward
        returns = returns.write(i, accumulator)
    # turn tensor list into immutable tensor, reverse
    returns = returns.stack()[::-1]
    # epsilon - small constant for stabilizing floating point division opeations
    # when dividing by zero or otherwise very small numbers
    eps = np.finfo(np.float32).eps.item()
    # standardize the rewards to have 0 mean and 1 sd
    # reduce operations return scalars if no axis provided
    return (returns - tf.math.reduce_mean(returns)) / (tf.math.reduce_std(returns) + eps)

# Loss function - this is run on the model under tape and then gets differentiated
# using autodiff on the operation sequence.
# Produces a scalar output that is used to get gradients for all 
# trainable parameters (matrices Wi, vectors Bi) in the model.
def get_loss(expected_rewards: tf.Tensor, inferred_rewards: tf.Tensor, action_probs: tf.Tensor):
    # advantage - compare expected reward with model's estimate
    # G(s_t, a_t) - V_theta(s_t)
    advantage = expected_rewards - inferred_rewards
    # get log probs from action probs, weigh by advantage, invert sign for loss 
    # convention.
    actor_loss = -1 * tf.math.reduce_sum(tf.math.log(action_probs) * advantage)
    # Huber loss - x^2 below delta, abs(x) above delta
    # reduction method = sum along vector
    huber = losses.Huber(reduction=losses.Reduction.SUM)
    critic_loss = huber(expected_rewards, inferred_rewards)
    # return a scalar - this is what the gradient is computed for wrt model params
    return actor_loss + critic_loss

# do single episode and train on collected batch (state-action sequence)
def training_step(start: tf.Tensor,
                    model: CombinedNetwork,
                    gamma: float,
                    optimizer: optimizers.Optimizer,
                    max_steps: int) -> tf.Tensor:
    # tape records tensor operations for differentiation
    with tf.GradientTape() as tape:
        action_probs, est_rewards, act_rewards = run_episode(model, start, max_steps)
        integrated_rewards = expected_reward(act_rewards, gamma)
        # NOTE: !!!!!!!!!
        # Without this step the gradients get all fucked and the model learns to kill itself ASAP
        action_probs, est_rewards, integrated_rewards = [tf.expand_dims(x, 1) for x in
         [action_probs, est_rewards, integrated_rewards]]
        loss = get_loss(integrated_rewards, est_rewards, action_probs)
    # use the tape record of elementary operations to roll back and
    # get the gradients of model parameters wrt loss
    gradients = tape.gradient(loss, model.trainable_variables)
    # give gradients, parameters to optimizer. This is the step that 
    # changes the model weights
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    # get the score for evaluation purposes (see when training is done, how it's doing)
    episode_reward = tf.math.reduce_sum(act_rewards)
    return episode_reward, loss

OFILE="eval-weights-v1"
HIDDEN=24
ACTIONS=2
GAMMA=0.9
MAX_STEPS=500
OPT=optimizers.Adam(learning_rate=0.01)
# ^!!!! dumb fuck, if you don't instantiate a class calling methods will think 
# there is no "self" argument
LOGLEVEL="INFO"
EAGER=False
MAX_EPISODES=10000
BUF_LENGTH=100
FINISHED=495.0

if __name__ == "__main__":

    env = gym.make("CartPole-v1")
    tf.get_logger().setLevel(LOGLEVEL)
    tf.config.run_functions_eagerly(EAGER)

    model = CombinedNetwork(ACTIONS, HIDDEN)

    # keep buffer of episdoes in a deque for rolling average reward
    # a deque is convenient because you can keep it limited to a certain
    # length, and popping is O(1)
    reward_buffer: collections.deque = collections.deque(maxlen=BUF_LENGTH)
    loss_buffer: collections.deque = collections.deque(maxlen=BUF_LENGTH)

    with tqdm.trange(MAX_EPISODES) as t:
        for i in t:
            state = tf.constant(env.reset(), dtype=np.float32)
            episode_reward, loss = training_step(state, model, GAMMA, OPT, MAX_STEPS)
            episode_reward = int(episode_reward)
            loss = float(loss)
            reward_buffer.append(episode_reward)
            loss_buffer.append(loss)
            rolling_avg = statistics.mean(reward_buffer)
            avg_loss = statistics.mean(loss_buffer)
            t.set_description(f'Episode {i}')
            t.set_postfix(avg_loss=avg_loss, rolling_avg=rolling_avg)
            if rolling_avg > FINISHED:
                break

    model.save_weights(OFILE)