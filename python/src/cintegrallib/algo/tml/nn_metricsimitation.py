#
# locally embedding metrics imitation using tensorflow
#
import tensorflow as tf
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
tf.enable_eager_execution()

#%%

def _neighborhood_batch(X=None, batch_size=None, k_neighbors=None):
    bidx = np.random.choice(X.shape[0], batch_size, replace=False)
    bneighbors = tf.nn.embedding_lookup(X, k_neighbors[bidx])
    batch = tf.nn.embedding_lookup(X, bidx)
    return batch, bneighbors, bidx

#%% locally embed - shared weights

def local_embed_shared_weights(features=None, batch_size=None, knn_size=None, steps=500):
    nei_size = knn_size
    knn = NearestNeighbors(nei_size + 1, n_jobs=5).fit(features)
    knn_idx = knn.kneighbors(features, return_distance=False)[:, 1:]
    knn_metrics = [cosine_similarity(features[[i]], features[knn_idx[i, :]]) for i in range(features.shape[0])]
    knn_metrics = tf.cast(tf.squeeze(tf.constant(knn_metrics)), tf.float32)
    features = tf.cast(features, tf.float32)

    W = tf.Variable(tf.truncated_normal([nei_size, 1], stddev=0.05))
    print('initial W', W)

    for step in range (steps):
        X, nei_features, bidx = _neighborhood_batch(features, batch_size, knn_idx)
        nei_metrics = tf.gather(knn_metrics, bidx)

        with tf.GradientTape() as g:
            mW = tf.einsum('ij, jk -> ij', nei_metrics, W)
            Y = tf.einsum('ijk, ij -> ik', nei_features, mW)
            loss = tf.squared_difference(Y, X)
            loss = tf.reduce_mean(loss)

        print('step (', step, ') loss: ', loss.numpy())
        grads = g.gradient(loss, W)
        optimizer = tf.train.AdamOptimizer(10**-2)
        optimizer.apply_gradients([(grads, W)])

    W = tf.constant(W.numpy())
    return knn_idx, knn_metrics, W

#%% locally embed - non shared weights
# stable and converged
def local_embed(features=None, batch_size=None, knn_size=None, fm=False, steps=500):
    nei_size = knn_size
    knn = NearestNeighbors(nei_size + 1, n_jobs=5).fit(features)
    knn_idx = knn.kneighbors(features, return_distance=False)[:, 1:]
    features = tf.cast(features, tf.float32)

    # locally embedding weights
    W = tf.Variable(tf.truncated_normal([features.shape[0], nei_size], stddev=0.05), tf.float32, name='W')
    var_list = [W]

    if fm:
        FMW = tf.Variable(tf.truncated_normal([features.shape[0], nei_size], stddev=0.05), tf.float32, name='FMW')
        var_list.append(FMW)

    for step in range(steps):

        X, nei_features, bidx = _neighborhood_batch(features, batch_size, knn_idx)

        with tf.GradientTape() as g:
            v_bW = tf.nn.embedding_lookup(W, bidx)
            Y1 = tf.einsum('ijk, ij -> ik', nei_features, v_bW)

            if fm: # factorization
                v_fW = tf.nn.embedding_lookup(FMW, bidx)
                Y2 = tf.einsum('ijk, ij -> ijk', nei_features, v_fW)
                Y2T = tf.einsum('ijk -> ikj', Y2)
                Y2 = tf.einsum('bij, bjk -> bjik', Y2, Y2T)
                Y2 = tf.matrix_set_diag(Y2, tf.zeros(tf.matrix_diag_part(Y2).shape, tf.float32))
                Y2 = tf.reduce_sum(Y2, axis=(2, 3)) / 2
                Y = Y1 + Y2
            else:
                Y = Y1

            loss = tf.squared_difference(Y, X)
            loss = tf.reduce_sum(loss)

        print('step (', step, ') loss: ', loss.numpy())
        grads = g.gradient(loss, var_list)
        optimizer = tf.train.AdamOptimizer(10 ** -2)
        optimizer.apply_gradients(zip(grads, var_list))

    W = tf.constant(W.numpy())
    if fm:
        FMW = tf.constant(FMW.numpy())
        return knn_idx, W, FMW
    else:
        return knn_idx, W

#%% imitation projection learning

def imitation_projection(features=None,     # target feature domain data
                         knn_idx=None,      # KNN index from weight training
                         W=None,            # local embed trained weights
                         FMW=None,          # local factorization embed trained weights
                         batch_size=None,   # batch size
                         embed_size=None,   # projection size for target domain
                         steps=500):

    imit_embed_size = embed_size
    imit_feature_dim = features.shape[-1]
    features = tf.cast(features, tf.float32)

    A = tf.Variable(tf.truncated_normal([imit_feature_dim, imit_embed_size], stddev=0.05), tf.float32)

    for step in range(steps):
        X, nei_features, bidx = _neighborhood_batch(features, batch_size, knn_idx)
        X = tf.cast(X, tf.float32)
        bW = tf.nn.embedding_lookup(W, bidx)  # fixed trained neighborhood weights

        with tf.GradientTape() as g:
            XA = tf.einsum('ij, jk -> ik', X, A) # projection X
            nei_proj = tf.einsum('ijk, kl', nei_features, A)
            YA1 = tf.einsum('ijl, ij -> il', nei_proj, bW) # projection neighbor features with fixed W

            if FMW != None: # factorization
                v_fW = tf.nn.embedding_lookup(FMW, bidx)
                YA2 = tf.einsum('ijk, ij -> ijk', nei_proj, v_fW)
                YA2T = tf.einsum('ijk -> ikj', YA2)
                YA2 = tf.einsum('bij, bjk -> bjik', YA2, YA2T)
                YA2 = tf.matrix_set_diag(YA2, tf.zeros(tf.matrix_diag_part(YA2).shape, tf.float32))
                YA2 = tf.reduce_sum(YA2, axis=(2, 3)) / 2
                YA = YA1 + YA2
            else:
                YA = YA1


            loss = tf.squared_difference(YA, XA)
            loss = tf.reduce_mean(loss)

        print('step (', step, ') loss: ', loss.numpy())
        grads = g.gradient(loss, A)
        optimizer = tf.train.AdamOptimizer(10 ** -2)
        optimizer.apply_gradients([(grads, A)])

    return A.numpy()