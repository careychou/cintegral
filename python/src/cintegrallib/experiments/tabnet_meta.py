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

#%%
def meta_train(sample, tabnet_model, max_iter, batch_idx, meta_train, max_update):
    for iter in range(max_iter):
        X = tf.convert_to_tensor(sample.iloc[batch_idx, 2:].values, dtype='float32')
        Y1 = tf.convert_to_tensor(sample.iloc[batch_idx, 0:1].values, dtype='float32')
        Y2 = tf.convert_to_tensor(sample.iloc[batch_idx, 1:2].values, dtype='float32')

        with tf.GradientTape(persistent=True) as g1:
            agg_out, tot_ent, agg_mask = tabnet_model.encode(X, tabnet_model.encode_weights)
            loss = tf.reduce_mean(tf.squared_difference(agg_out, Y1))

            if meta_train: # meta train
                # update init
                grads = g1.gradient(loss, g1.watched_variables())
                gradients = dict(zip(tabnet_model.encode_weights.keys(), grads))
                fast_weights = {key: tabnet_model.encode_weights[key] - 1e-3 * gradients[key] for key in gradients.keys()}
                agg_out, tot_ent, agg_mask = tabnet_model.encode(X, fast_weights)
                loss_meta = tf.reduce_mean(tf.squared_difference(agg_out, Y2))
                # update loop
                for update in range(max_update):
                    agg_out, tot_ent, agg_mask = tabnet_model.encode(X, fast_weights)
                    loss = tf.reduce_mean(tf.squared_difference(agg_out, Y1))
                    grads = g1.gradient(loss, fast_weights.values())
                    gradients = dict(zip(fast_weights.keys(), grads))
                    fast_weights = {key:fast_weights[key] - 1e-3 * gradients[key] for key in gradients.keys()}
                    agg_out, tot_ent, agg_mask = tabnet_model.encode(X, fast_weights)
                    loss_meta = tf.reduce_mean(tf.squared_difference(agg_out, Y2))

        if meta_train:
            print('iter:', iter+1, ' loss meta:', loss_meta.numpy())
            grads = g1.gradient(loss_meta, g1.watched_variables())
        else:
            print('iter:', iter+1, ' loss:', loss.numpy())
            grads = g1.gradient(loss, g1.watched_variables())

        optimizer = tf.train.AdamOptimizer(10 ** -2)
        optimizer.apply_gradients(zip(grads, g1.watched_variables()))

    return grads

#%%
batch_size = 50
num_features = normalized_data.shape[1]-2
tabnet = TabNet(num_features=num_features, feature_dim=num_features+1, output_dim=1, num_decision_steps=4)
l2_reg = tf.contrib.layers.l2_regularizer(10 ** -2)

#%%
# pretrain
bidx = np.random.choice(normalized_data.shape[0], batch_size, replace=True)
grads = meta_train(normalized_data, tabnet, 5, bidx, False, 5)

# meta-train
bidx = np.random.choice(normalized_data.shape[0], batch_size, replace=True)
grads = meta_train(normalized_data, tabnet, 10, bidx, True, 5)
