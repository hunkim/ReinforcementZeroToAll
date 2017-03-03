'''
This code is based on:
https://github.com/hunkim/DeepRL-Agents
http://karpathy.github.io/2016/05/31/rl/
'''
import numpy as np
import tensorflow as tf
import os
# by Jin Kim (golbin) https://github.com/golbin/TensorFlow-Tutorials
import mini_pacman

env = mini_pacman.Gym()

hidden_layer_neurons = 64
learning_rate = 1e-3
gamma = .99

LOG_DIR = './pacman_log'

# Constants defining our neural network
input_size = 240
output_size = 3

X = tf.placeholder(tf.float32, [None, input_size], name="input_x")
x_image = tf.reshape(X, [-1, 6, 10, 1])
tf.summary.image('input', x_image, 8)

# First layer of weights
with tf.name_scope("layer1"):
    W1 = tf.get_variable("W1", shape=[input_size, hidden_layer_neurons],
                         initializer=tf.contrib.layers.xavier_initializer())
    B1 = tf.Variable(tf.zeros([hidden_layer_neurons]))
    layer1 = tf.matmul(X, W1) + B1
    layer1_act = tf.nn.tanh(layer1)
    tf.summary.histogram("X", X)
    tf.summary.histogram("weights", W1)
    tf.summary.histogram("bias", B1)
    tf.summary.histogram("layer", layer1)
    tf.summary.histogram("activations", layer1_act)


# Second layer of weights
with tf.name_scope("layer2"):
    W2 = tf.get_variable("W2", shape=[hidden_layer_neurons, output_size],
                         initializer=tf.contrib.layers.xavier_initializer())
    B2 = tf.Variable(tf.zeros([output_size]))
    layer2 = tf.matmul(layer1_act, W2) + B2
    action_pred = tf.nn.softmax(layer2)
    tf.summary.histogram("weights", W2)
    tf.summary.histogram("bias", B2)
    tf.summary.histogram("layer", layer2)
    tf.summary.histogram("action_pred", action_pred)

# We need to define the parts of the network needed for learning a policy
Y = tf.placeholder(tf.float32, [None, output_size], name="input_y")
advantages = tf.placeholder(tf.float32, name="reward_signal")

# Loss function
# Sum (Ai*logp(yi|xi))
log_lik = -Y * tf.log(action_pred)
log_like_adv = log_lik * advantages
loss = tf.reduce_mean(tf.reduce_sum(log_like_adv))
tf.summary.scalar("Q", tf.reduce_mean(action_pred))
tf.summary.scalar("Y", tf.reduce_mean(Y))
tf.summary.scalar("log_likelihood", tf.reduce_mean(log_lik))
tf.summary.scalar("loss", loss)

# Learning
train = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

# Some place holders for summary
summary_reward = tf.placeholder(tf.float32, shape=(), name="reward")
tf.summary.scalar("reward", summary_reward)

# Summary
summary = tf.summary.merge_all()


def discount_rewards(r, gamma=0.99):
    """ take 1D float array of rewards and compute discounted reward """
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(range(0, r.size)):
        if r[t] != 0:
            # reset the sum, since this was a game boundary (pong specific!)
            running_add = 0
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add

        # compute the discounted reward backwards through time
        # standardize the rewards to be unit normal (helps control the gradient
        # estimator variance)
        discounted_r -= np.mean(discounted_r)
        discounted_r /= np.std(discounted_r)

    return discounted_r

# Setting up our environment
sess = tf.Session()
rendering = False
sess.run(tf.global_variables_initializer())

# TensorBoard
writer = tf.summary.FileWriter(LOG_DIR)
writer.add_graph(sess.graph)

# Savor and Restore
saver = tf.train.Saver()
checkpoint = tf.train.get_checkpoint_state(LOG_DIR)
if checkpoint and checkpoint.model_checkpoint_path:
    try:
        saver.restore(sess, checkpoint.model_checkpoint_path)
        print("Successfully loaded:", checkpoint.model_checkpoint_path)
    except:
        print("Error on loading old network weights")
else:
    print("Could not find old network weights")

global_step = 0
while True:
    global_step += 1

    # Clear out game variables
    xs = np.empty(0).reshape(0, input_size)
    ys = np.empty(0).reshape(0, output_size)
    rewards = np.empty(0).reshape(0, 1)

    reward_sum = 0
    state = env.reset()

    # Initial 4 frame data
    s_t = np.array([state, state, state, state])

    while True:
        # Append the observations to our batch
        x = np.reshape(s_t, [1, input_size])

        # Run the neural net to determine output
        action_prob = sess.run(action_pred, feed_dict={X: x})
        action_prob = np.squeeze(action_prob)  # shape (?, 2) -> 2
        random_noise = np.random.uniform(0, 1, output_size)
        action = np.argmax(action_prob + random_noise)
        print("Action prediction: ", np.argmax(action_prob), " action taken:", action,
              np.argmax(action_prob) == action)

        # Append the observations and outputs for learning
        xs = np.vstack([xs, x])
        y = np.eye(output_size)[action:action + 1]  # One hot encoding
        ys = np.vstack([ys, y])

        # Determine the outcome of our action
        state, reward, done, _ = env.step(action)

        s_t = np.array([state, s_t[0], s_t[1], s_t[2]])
        reward_sum += reward
        rewards = np.vstack([rewards, reward])

        if done:
            # Determine standardized rewards
            rewards = discount_rewards(rewards)
            reward_mean = np.mean(rewards)
            ll, la, l, s, _ = sess.run([log_lik, log_like_adv, loss, summary, train],
                                       feed_dict={X: xs,
                                                  Y: ys,
                                                  advantages: rewards,
                                                  summary_reward: reward_mean})
            #print(ll, la)
            writer.add_summary(s, global_step)
            break

    # Print status
    print("Average reward for episode {}: {}. Loss: {}".format(
        global_step, reward_sum, l))

    if global_step % 100 == 0:
        print("Saving network...")
        if not os.path.exists(LOG_DIR):
            os.makedirs(LOG_DIR)
        saver.save(sess, os.path.join(LOG_DIR, "model.ckpt"),
                   global_step=global_step)
