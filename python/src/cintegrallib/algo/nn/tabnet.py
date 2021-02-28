import numpy as np
import tensorflow as tf
import pickle

tf.enable_eager_execution()

#%%

def glu(act, n_units):
  """Generalized linear unit nonlinear activation."""
  return act[:, :n_units] * tf.nn.sigmoid(act[:, n_units:])

class TabNet:
    def __init__(self, num_features, feature_dim, output_dim, num_decision_steps):
        self.num_features = num_features
        self.feature_dim = feature_dim
        self.output_dim = output_dim
        self.feature_coeff_dim = feature_dim - output_dim
        self.num_decision_steps = num_decision_steps
        self.epsilon = 1e-3
        self._build()

    def _build(self):
        # encode weights
        self.encode_weights = {}
        self.encode_weights['tf1_w'] = tf.Variable(tf.truncated_normal([self.num_features, self.feature_dim * 2], stddev=0.05), tf.float32, name='tf1_w')
        self.encode_weights['tf2_w'] = tf.Variable(tf.truncated_normal([self.feature_dim, self.feature_dim * 2], stddev=0.05), tf.float32, name='tf2_w')
        self.encode_weights['bn_gamma1'] = tf.Variable(tf.ones([self.feature_dim * 2]), name='bn_gamma1')
        self.encode_weights['bn_beta1'] = tf.Variable(tf.zeros([self.feature_dim * 2]), name='bn_beta1')
        self.encode_weights['bn_gamma2'] = tf.Variable(tf.ones([self.feature_dim * 2]), name='bn_gamma2')
        self.encode_weights['bn_beta2'] = tf.Variable(tf.zeros([self.feature_dim * 2]), name='bn_beta2')

        for i in range(1, self.num_decision_steps+1):
            if i < self.num_decision_steps:
                self.encode_weights['mask_w_' + str(i)] = tf.Variable(tf.truncated_normal([self.feature_coeff_dim, self.num_features], stddev=0.05), tf.float32,name='mask_w_' + str(i))
                self.encode_weights['mask_gamma_' + str(i)] = tf.Variable(tf.ones([self.num_features]), name='mask_gamma_' + str(i))
                self.encode_weights['mask_beta_' + str(i)] = tf.Variable(tf.zeros([self.num_features]), name='mask_beta_' + str(i))
            self.encode_weights['tf3_w_' + str(i)] = tf.Variable(tf.truncated_normal([self.feature_dim, self.feature_dim * 2], stddev=0.05), tf.float32, name='tf3_w_' + str(i))
            self.encode_weights['bn_gamma3_' + str(i)] = tf.Variable(tf.ones([self.feature_dim * 2]), name='bn_gamma3_' + str(i))
            self.encode_weights['bn_beta3_' + str(i)] = tf.Variable(tf.zeros([self.feature_dim * 2]), name='bn_beta3_' + str(i))
            #self.encode_weights['tf4_w_' + str(i)] = tf.Variable(tf.truncated_normal([self.feature_dim, self.feature_dim * 2], stddev=0.05), tf.float32, name='tf4_w_' + str(i))
            #self.encode_weights['bn_gamma4_' + str(i)] = tf.Variable(tf.ones([self.feature_dim * 2]), name='bn_gamma4_' + str(i))
            #self.encode_weights['bn_beta4_' + str(i)] = tf.Variable(tf.zeros([self.feature_dim * 2]), name='bn_beta4_' + str(i))

        # decode weights
        self.decode_weights = {}
        self.decode_weights['tf1_w'] = tf.Variable(tf.truncated_normal([self.output_dim, self.feature_dim * 2], stddev=0.05), tf.float32, name='tf1_w')
        self.decode_weights['tf2_w'] = tf.Variable(tf.truncated_normal([self.feature_dim, self.feature_dim * 2], stddev=0.05), tf.float32, name='tf2_w')
        self.decode_weights['outputfc_w'] = tf.Variable(tf.truncated_normal([self.feature_dim, self.num_features], stddev=0.05), tf.float32, name='outputfc_w')
        self.decode_weights['bn_gamma1'] = tf.Variable(tf.ones([self.feature_dim * 2]), name='bn_gamma1')
        self.decode_weights['bn_beta1'] = tf.Variable(tf.zeros([self.feature_dim * 2]), name='bn_beta1')
        self.decode_weights['bn_gamma2'] = tf.Variable(tf.ones([self.feature_dim * 2]), name='bn_gamma2')
        self.decode_weights['bn_beta2'] = tf.Variable(tf.zeros([self.feature_dim * 2]), name='bn_beta2')

    def decode(self, features, weights):
        encoded_features = features
        output_aggregated = tf.zeros([features.shape[0], self.num_features])

        for ni in range(self.num_decision_steps):
            # residual layers shared weights
            tf1 = tf.einsum('ij, jk -> ik', encoded_features, weights['tf1_w'])
            mean1, var1 = tf.nn.moments(tf1, [0])
            tf1 = tf.nn.batch_normalization(tf1, mean1, var1, weights['bn_beta1'], weights['bn_gamma1'], self.epsilon)
            tf1 = glu(tf1, self.feature_dim)

            tf2 = tf.einsum('ij, jk -> ik', tf1, weights['tf2_w'])
            mean2, var2 = tf.nn.moments(tf2, [0])
            tf2 = tf.nn.batch_normalization(tf2, mean2, var2, weights['bn_beta2'], weights['bn_gamma2'], self.epsilon)
            tf2 = (glu(tf2, self.feature_dim) + tf1) * np.sqrt(0.5)

            # output representations
            out = tf2 = tf.einsum('ij, jk -> ik', tf2, weights['outputfc_w'])
            output_aggregated += out

        return output_aggregated

    def encode(self, features, weights):
        masked_features = features
        total_entropy = 0
        output_aggregated = tf.zeros([features.shape[0], self.output_dim])
        aggregated_mask_values = tf.zeros([features.shape[0], self.num_features])

        for ni in range(self.num_decision_steps):
            # transformer layers shared weights
            tf1 = tf.einsum('ij, jk -> ik', masked_features, weights['tf1_w'])
            mean1, var1 = tf.nn.moments(tf1, [0])
            tf1 = tf.nn.batch_normalization(tf1, mean1, var1, weights['bn_beta1'], weights['bn_gamma1'], self.epsilon)
            tf1 = glu(tf1, self.feature_dim)

            tf2 = tf.einsum('ij, jk -> ik', tf1, weights['tf2_w'])
            mean2, var2 = tf.nn.moments(tf2, [0])
            tf2 = tf.nn.batch_normalization(tf2, mean2, var2, weights['bn_beta2'], weights['bn_gamma2'], self.epsilon)
            tf2 = (glu(tf2, self.feature_dim) + tf1) * np.sqrt(0.5)

            # transformer layers non-shared weights - decision step
            tf3 = tf.einsum('ij, jk -> ik', tf1, weights['tf3_w_' + str(ni+1)])
            mean3, var3 = tf.nn.moments(tf3, [0])
            tf3 = tf.nn.batch_normalization(tf3, mean3, var3, weights['bn_beta3_' + str(ni+1)], weights['bn_gamma3_' + str(ni+1)], self.epsilon)
            tf3 = (glu(tf3, self.feature_dim) + tf2) * np.sqrt(0.5)

            # output representations
            out = tf.nn.relu(tf3[:, :self.output_dim])
            output_aggregated += out

            # aggregated masked for feature importance
            if ni > 0:
                scale_agg = tf.reduce_sum(out, axis=1, keep_dims=True) / (self.num_decision_steps - 1)
                aggregated_mask_values += mask_values * scale_agg

            if ni < self.num_decision_steps - 1:
                # attention masks
                feature_coeff = tf3[:, self.output_dim:]
                mask_values = tf.einsum('ij, jk -> ik', feature_coeff, weights['mask_w_' + str(ni+1)])
                mean_mask, var_mask = tf.nn.moments(mask_values, [0])
                mask_values = tf.nn.batch_normalization(mask_values, mean_mask, var_mask, weights['mask_beta_' + str(ni+1)], weights['mask_gamma_' + str(ni+1)], self.epsilon)
                mask_values = tf.contrib.sparsemax.sparsemax(mask_values)
                masked_features = tf.multiply(mask_values, features)
                total_entropy += tf.reduce_mean(tf.reduce_sum(-mask_values * tf.log(mask_values + self.epsilon), axis=1)) / (self.num_decision_steps - 1)

        return output_aggregated, total_entropy, aggregated_mask_values

    def save_weights(self, outputpath):
        encode_weights = {k: v.numpy() for k, v in self.encode_weights.items()}
        with open(outputpath + '/tabnet_encode_w.p', 'wb') as f:
            pickle.dump(encode_weights, f)

        decode_weights = {k: v.numpy() for k, v in self.decode_weights.items()}
        with open(outputpath + '/tabnet_decode_w.p', 'wb') as f:
            pickle.dump(decode_weights, f)

    def load_weights(self, inputpath):
        with open(inputpath + '/tabnet_encode_w.p', 'rb') as f:
            w = pickle.load(f)
        self.encode_weights = {k: tf.Variable(v, tf.float32) for k, v in w.items()}

        with open(inputpath + '/tabnet_decode_w.p', 'rb') as f:
            w = pickle.load(f)
        self.decode_weights = {k: tf.Variable(v, tf.float32) for k, v in w.items()}