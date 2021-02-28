#%%
import numpy as np
import tensorflow as tf
from cintegrallib.lib.algo.nn.tabnet import TabNet
from fndwechat.utils import load_data

#%%
path = '/misc_lib/fnd_wechat/data/wechat_04262020/'
data = load_data.load_follower_tags(path)
data = data.iloc[:, 4:]
normalized_data = (data-data.min())/(data.max()-data.min())
normalized_data = normalized_data.dropna(axis=1)
data = data[normalized_data.columns.values]

batch_size = 30
tabnet = TabNet(num_features=normalized_data.shape[1], feature_dim=normalized_data.shape[1]*2, output_dim=10, num_decision_steps=4)
l2_reg = tf.contrib.layers.l2_regularizer(10 ** -2)

for iter in range(1000):
    bidx = np.random.choice(normalized_data.shape[0], batch_size, replace=True)
    X = tf.convert_to_tensor(normalized_data.iloc[bidx, :].values, dtype='float32')

    with tf.GradientTape() as g1:
        agg_out, tot_ent, agg_mask = tabnet.encode(X, tabnet.encode_weights)
        feature_out = tabnet.decode(agg_out, tabnet.decode_weights)
        neg_feature_cost = tf.reduce_mean(tf.nn.relu(feature_out) - feature_out) # keep output positive range
        l2 = tf.contrib.layers.apply_regularization(l2_reg, [*tabnet.encode_weights.values(), *tabnet.decode_weights.values()])
        loss = tf.squared_difference(feature_out, X)
        loss = tf.reduce_mean(loss) + neg_feature_cost

    print('iter [' + str(iter) + ' loss: ' + str(loss))
    len(g1.watched_variables())
    vars = [*tabnet.encode_weights.values(), *tabnet.decode_weights.values()]
    grads = g1.gradient(loss, vars)
    optimizer = tf.train.AdamOptimizer(10 ** -2)
    optimizer.apply_gradients(zip(grads, vars))
