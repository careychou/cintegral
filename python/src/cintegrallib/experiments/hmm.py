import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import scipy.stats

tfd = tfp.distributions
tf.enable_eager_execution()
print(tf.__version__)
print(tfp.__version__)

#%%
# fake observation
sample = np.array([[1, 1, 0, 2],
                   [1, 0, 2, 1]])

# shared inductive weights
num_state = 2
num_observe = 3
transit_var = tf.Variable(tf.truncated_normal([num_state, num_state], stddev=0.05), tf.float32)
emission_var = tf.Variable(tf.truncated_normal([num_state, num_observe], stddev=0.05), tf.float32)

for loop in range(500):
    with tf.GradientTape() as g:
        transit = tfd.Categorical(probs=tf.nn.softmax(transit_var))
        emission = tfd.Categorical(probs=tf.nn.softmax(emission_var))
        init_dist = tfd.Categorical(logits=tf.ones(num_state))
        hmm = tfd.HiddenMarkovModel(initial_distribution=init_dist,
                                      transition_distribution=transit,
                                      observation_distribution=emission,
                                      num_steps=4)

        loss = -tf.reduce_mean(hmm.log_prob(sample))

    print('neg logprob: ', loss)
    grads = g.gradient(loss, [transit_var, emission_var])
    optimizer = tf.train.AdamOptimizer(10 ** -2)
    optimizer.apply_gradients(zip(grads, [transit_var, emission_var]))

#%%

print(tf.nn.softmax(transit_var))
print(tf.nn.softmax(emission_var))
