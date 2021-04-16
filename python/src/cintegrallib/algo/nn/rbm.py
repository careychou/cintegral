import tensorflow as tf
import numpy as np
assert  tf.__version__ == '1.15.0'
tf.enable_eager_execution()

# Bernoulli RBM
# https://github.com/meownoid/tensorflow-rbm/blob/master/tfrbm/bbrbm.py
#

class BBRBM:
    def __init__(self, n_visible, n_hidden, lr=10**-2, momentum=0.95):
        # rate
        self.lr = lr
        self.momentum = momentum
        # weights
        self.W = tf.Variable(tf.truncated_normal([n_visible, n_hidden]), name='W', dtype=tf.float32)
        self.bv = tf.Variable(tf.zeros(n_visible), name='bv', dtype=tf.float32)
        self.bh = tf.Variable(tf.zeros(n_hidden), name='bh', dtype=tf.float32)
        # velocities of momentum method
        self.w_v = tf.Variable(tf.zeros([n_visible, n_hidden]), name='w_v', dtype=tf.float32)
        self.bv_v = tf.Variable(tf.zeros([n_visible]), name='bv_v', dtype=tf.float32)
        self.bh_v = tf.Variable(tf.zeros([n_hidden]), name='bh_v', dtype=tf.float32)

    # visible: [batch, visible_units]
    def __call__(self, visible, CDk=10):
        # gibbs sampling
        # v => h => v => h
        h_sample, h = self.sample_h_given_v(visible)

        for i in range(CDk):
            v_sample, v_state = self.sample_v_given_h(h_sample)
            h_sample, h_state = self.sample_h_given_v(v_state)

        positive_grads = tf.matmul(tf.transpose(visible), h)
        negative_grads = tf.matmul(tf.transpose(v_state), h_state)

        # momentum
        self.w_v = self.apply_momentum(self.w_v,  (positive_grads - negative_grads) / tf.to_float(tf.shape(visible)[0]))
        self.bv_v = self.apply_momentum(self.bv_v, tf.reduce_mean(visible - v_state, 0))
        self.bh_v = self.apply_momentum(self.bh_v, tf.reduce_mean(h - h_state, 0))

        # update weights and bias
        self.W.assign_add(self.w_v)
        self.bv.assign_add(self.bv_v)
        self.bh.assign_add(self.bh_v)

        # error
        _, v_recon = self.sample_v_given_h(h_sample)
        return tf.reduce_mean(tf.square(visible - v_recon)).numpy()

    # from h => v
    def sample_h_given_v(self, v_sample):
        h = tf.sigmoid(tf.matmul(v_sample, self.W) + self.bh)
        return self.sample(h), h

    # from v => h
    def sample_v_given_h(self, h_sample):
        v = tf.sigmoid(tf.matmul(h_sample, tf.transpose(self.W)) + self.bv)
        return self.sample(v), v

    # momentum adjustment = momentum * old_grads + ((1 - momentum) * lr/n) * new_grads
    def apply_momentum(self, old, new):
        n = tf.cast(new.shape[0], dtype=tf.float32)
        return tf.add(tf.math.scalar_mul(self.momentum, old), tf.math.scalar_mul((1 - self.momentum) * self.lr / n, new))

    # bernoulli sampling
    def sample(self, probs):
        return tf.nn.relu(tf.sign(probs - tf.random.uniform(tf.shape(probs))))

    # free energy in data
    def free_energy(self, data):
        first_term = tf.matmul(data, tf.transpose(self.bv))
        second_term = tf.reduce_sum(tf.log(1 + tf.exp(self.bh + tf.matmul(data, self.W))), axis=1)
        return -first_term - second_term
