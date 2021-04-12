import tensorflow as tf
import numpy as np
import plotly.express as px
from cintegrallib.algo.nn import dream_transformer as dream_tr
from cintegrallib.algo.nn import opt

assert  tf.__version__ == '1.15.0'
tf.enable_eager_execution()

# explain user embedding neuron using direct loss
# this function simulate each slot in the basket, then compute single neuron activation
# in user embedding => concave function
def explain_user_embedding_neuron_by_loss(model_path):
    # restore prior trained model
    dream_model = dream_tr.DreamTransformer.restore_model(model_path, trainable=True)
    activation_map = np.zeros([dream_model.h_dim, dream_model.basket_size])
    for idx in range(dream_model.h_dim):
        # mask activation
        mask = np.zeros([1, 1, dream_model.h_dim])
        # mask to retrive active neuron
        mask[0, 0, idx] = 1
        mask = tf.constant(mask, dtype=tf.float32)

        for bidx in range(dream_model.basket_size-1):
            x = np.zeros([1, 1, dream_model.basket_size]).astype(int)
            x[0, 0, bidx] = bidx+1
            dream_model(x)
            activation_value = tf.reduce_sum(dream_model.outputs * mask).numpy()
            activation_map[idx, bidx] = activation_value

    activation_map = np.transpose(np.array(activation_map)[:, :-1])
    return activation_map

# explain user embedding neuron using 1st order gradient w.r.t inputs
def explain_user_embedding_neuron_by_dx(model_path):
    # restore prior trained model
    dream_model = dream_tr.DreamTransformer.restore_model(model_path, trainable=True)

    # test with a sample
    x = np.array([[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 0]]])
    activation_map = []

    for idx in range(dream_model.h_dim):
        # mask activation
        mask = np.zeros([1, 1, dream_model.h_dim])
        # mask and maximize a neuron at output pos
        mask[0, 0, idx] = -1
        mask = tf.constant(mask, dtype=tf.float32)

        with tf.GradientTape() as g1:
            dream_model(x)
            loss = tf.reduce_sum(dream_model.outputs * mask)

        grads = g1.gradient(loss, dream_model.z)
        grads = tf.boolean_mask(grads.values, tf.math.not_equal(grads.indices, 0))
        update = tf.reduce_sum(grads, axis=1).numpy()
        activation_map.append(update)

    activation_map = np.transpose(np.array(activation_map)[:, :-1])
    return activation_map

# explain user embedding neuron using hessian w.r.t inputs
def explain_user_embedding_neuron_by_d2x(model_path):
    # restore prior trained model
    dream_model = dream_tr.DreamTransformer.restore_model(model_path, trainable=True)

    # test with a sample
    x = np.array([[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 0]]])
    activation_map = []

    for idx in range(dream_model.h_dim):
        # mask activation
        mask = np.zeros([1, 1, dream_model.h_dim])
        # maximize a neuron at output pos
        mask[0, 0, idx] = -1
        mask = tf.constant(mask, dtype=tf.float32)

        with tf.GradientTape() as g1:
            with tf.GradientTape() as g2:
                dream_model(x)
                loss = tf.reduce_sum(dream_model.outputs * mask)

            grads1 = g2.gradient(loss, dream_model.z).values
        grads2 = g1.jacobian(grads1, dream_model.z)

        # slice out the zero embedding index from 1st order and jacobian matrix
        grads1 = grads1[:-1]
        grads2 = grads2[:-1, :, 1:, :]

        update = opt.newton_root(grads1, grads2)
        update = tf.reduce_sum(update, axis=1).numpy()
        activation_map.append(update)

    activation_map = np.transpose(np.array(activation_map)[:, :-1])
    return activation_map, grads2

def explain_user_embedding_neuron_by_projected_embedding(model_path, user_embedding_idx):
    # restore prior trained model
    dream_model = dream_tr.DreamTransformer.restore_model(model_path, trainable=True)
    lop = tf.train.AdamOptimizer(10**-2)
    original_item_embeddings = dream_model.z.numpy().copy()

    # test with a sample
    x = np.array([[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 0]]])
    # mask and maximize a neuron at output pos
    mask = np.zeros([1, 1, dream_model.h_dim])
    mask[0, 0, user_embedding_idx] = -1
    mask = tf.constant(mask, dtype=tf.float32)

    for epoch in range (100):
        with tf.GradientTape() as g1:
            dream_model(x)
            loss = tf.reduce_sum(dream_model.outputs * mask)


        print('loss :', tf.reduce_sum(loss))
        grads = g1.gradient(loss, dream_model.z)
        lop.apply_gradients(zip([grads], [dream_model.z]))


    new_item_embeddings = dream_model.z.numpy().copy()

    return original_item_embeddings, new_item_embeddings





