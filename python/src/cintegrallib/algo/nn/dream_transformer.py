import tensorflow as tf
import numpy as np
assert  tf.__version__ == '1.15.0'
tf.enable_eager_execution()

##
## This version extends original paper using MultiHead Transformer attention replacing recurrent neural network
## for the detail how attention and transformer works, refer to the paper from Google Brain:
## https://arxiv.org/pdf/1706.03762.pdf
##
## 2nd order factorization machine is inserted to learn interaction between items embeddings in basket
##
## carey.h.chou@gmail.com
##

class DreamTransformer:

    # restore a prior trained model
    @classmethod
    def restore_model(cls, filepath, trainable=False):
        import pickle as p

        with open(filepath, "rb") as f:
            init_params, model_params = p.load(f)

        model = cls(init_params['n_classes'],
                    init_params['basket_size'],
                    init_params['tr_head'],
                    init_params['tr_depth'],
                    init_params['h_dim'],
                    init_params['lr'],
                    init_params['fm_dim'],
                    init_params['fm_ratio'])

        # optimizer init
        model.optimizer = tf.train.AdamOptimizer(model.lr)

        # initialize training variables
        model.z = tf.Variable(model_params['z'], name="items_embeddings", dtype=tf.float32, trainable=trainable)
        model.tr_q = tf.Variable(model_params['tr_q'], name="tr_q", dtype=tf.float32, trainable=trainable)
        model.tr_k = tf.Variable(model_params['tr_k'], name="tr_k", dtype=tf.float32, trainable=trainable)
        model.tr_v = tf.Variable(model_params['tr_v'], name="tr_v", dtype=tf.float32, trainable=trainable)
        model.tr_hw = tf.Variable(model_params['tr_hw'], name="tr_hw", dtype=tf.float32, trainable=trainable)

        # if enable factorization machine
        if model.fm_ratio > 0:
            model.fm_b = tf.Variable(model_params['fm_b'], name='fm_b', dtype=tf.float32, trainable=trainable)
            model.fm_w = tf.Variable(model_params['fm_w'], name='fm_w', dtype=tf.float32, trainable=trainable)
            model.fm_v = tf.Variable(model_params['fm_v'], name='fm_v', dtype=tf.float32, trainable=trainable)

        return model

    # initialization
    # Params:
    #   n_classes: number of product classes in basket
    #   basket_size: max size of basket
    #   tr_head: number of heads in multi-head transformer
    #   tr_depth: the depth of interaction between transformer learning matrix
    #   h_dim: size of the embedding for product in the basket, this sets as the LSTM state size as well
    #   lr: learning rate for Adam optimization
    #   fm_dim: factorization machine kernel dimension
    def __init__(self, n_classes, basket_size, tr_head=5, tr_depth=16, h_dim=32, lr=10**-2, fm_dim=2, fm_ratio=0.5):
        self.h_dim = h_dim
        self.lr = lr
        self.n_classes = n_classes
        self.basket_size = basket_size
        self.fm_dim = fm_dim
        self.fm_ratio=fm_ratio
        self.tr_head = tr_head
        self.tr_depth = tr_depth
    def build(self):
        # optimizer init
        self.optimizer = tf.train.AdamOptimizer(self.lr)

        # initialize training variables
        self.z = tf.Variable(initial_value=tf.random.truncated_normal([self.n_classes + 1, self.h_dim]), name="items_embeddings", dtype=tf.float32, trainable=True)
        self.tr_q = tf.Variable(initial_value=tf.random.truncated_normal([self.tr_head, self.h_dim, self.tr_depth]), name="tr_q", dtype=tf.float32, trainable=True)
        self.tr_k = tf.Variable(initial_value=tf.random.truncated_normal([self.tr_head, self.h_dim, self.tr_depth]), name="tr_k", dtype=tf.float32, trainable=True)
        self.tr_v = tf.Variable(initial_value=tf.random.truncated_normal([self.tr_head, self.h_dim, self.tr_depth]), name="tr_v", dtype=tf.float32, trainable=True)
        self.tr_hw = tf.Variable(initial_value=tf.random.truncated_normal([self.tr_head * self.tr_depth, self.h_dim]),name="tr_hw", dtype=tf.float32, trainable=True)

        # if enable factorization machine
        if self.fm_ratio > 0:
            self.fm_b = tf.Variable(tf.zeros([1]), name='fm_b', dtype=tf.float32, trainable=True)
            self.fm_w = tf.Variable(initial_value=tf.random.truncated_normal([self.basket_size]), name='fm_w', dtype=tf.float32, trainable=True)
            self.fm_v = tf.Variable(initial_value=tf.random.truncated_normal([self.basket_size, self.fm_dim]), name='fm_v', dtype=tf.float32, trainable=True)


    # inference
    # Param:
    #   x: [batch_size, time_step, basket]
    def __call__(self, x):
        # Embedded labels (products or categories) [batch_size, history_size-1, basket_size, Z]
        zi = tf.nn.embedding_lookup(params=self.z, ids=x)
        # mask embedding on zero slots
        pad_mask = (x > 0).astype(int)
        zi *= pad_mask[:, :, :, np.newaxis]

        # 2nd order factorization machine
        if self.fm_ratio > 0:
            fm_linear_term = self.fm_b + tf.einsum('ijkl, k -> ijl', zi, self.fm_w)
            fm_kernel_term = 0.5 * tf.reduce_sum(
                tf.pow(tf.einsum('ijkl, km -> ijlm', zi, self.fm_v), 2) -
                tf.einsum('ijkl, km -> ijlm', tf.pow(zi, 2), tf.pow(self.fm_v, 2)), axis=-1)


        # Max pooling along labels (products or categories) [batch_size, history_size-1, Z]
        # Here we should mask multiply to avoid max with empty-bucket embeddings labels (as in Dream Attributes)
        zi_max_pool = tf.math.reduce_max(zi, axis=2)

        # max pool + factorization machine terms
        # [batch_size, history_size-1, Z]
        self.basket_embedding = (1 - self.fm_ratio) * zi_max_pool
        if self.fm_ratio > 0: self.basket_embedding += self.fm_ratio * (fm_linear_term + fm_kernel_term)

        # transformer Q, K, V [head, batch, seq, depth]
        Q = tf.einsum('ijk, hkd -> hijd', self.basket_embedding, self.tr_q)
        K = tf.einsum('ijk, hkd -> hijd', self.basket_embedding, self.tr_k)
        V = tf.einsum('ijk, hkd -> hijd', self.basket_embedding, self.tr_v)

        # Q [head, batch, q seq, q depth]
        # K [head, batch, k seq, k depth]
        # QK [head, batch, q seq, k seq]
        QK = tf.einsum('hijd, hild -> hijl', Q, K)
        dk = tf.cast(tf.shape(K)[-1], tf.float32)
        # attn [head, batch, q seq, k seq]
        self.attn = QK / tf.math.sqrt(dk)
        # make sure K is not peeking ahead in autogressive
        attn_mask = tf.linalg.band_part(tf.ones((tf.shape(QK)[-1], tf.shape(QK)[-1])), -1, 0)
        self.attn *= attn_mask
        # softmax sum to K seq dim --> weight distribution on K seq
        self.attn_weights = tf.nn.softmax(self.attn, axis=-1)
        # [head, batch, V seq, V depth] \dot [head, batch, Q seq, K seq] = [batch, Q seq, head, V depth]
        self.outputs = tf.einsum('hild, hijl -> ijhd', V, self.attn_weights)
        # flatten multi-heads and project to item embedding [batch, Q seq, head * depth]
        self.outputs = tf.reshape(self.outputs, [self.outputs.shape[0], self.outputs.shape[1], -1])
        self.outputs = tf.einsum('ijk, kl -> ijl', self.outputs, self.tr_hw)

        # items embeddings truncate first dimensnion (unkwown label) [Z, n_classes]
        w = tf.transpose(self.z[1:])

        # Logits [batch_size, history_size-1, n_classes]
        logits = tf.tensordot(self.outputs, w, axes=[[2], [0]])

        # Multi classification probas [batch_size, history_size-1, n_classes]
        self.probas = tf.sigmoid(logits)

        return self.probas

    # temporal binary cross entropy loss [batch_size, history_size-1]
    def log_loss(self, y_hat, y):
        p_loss = tf.reduce_mean(tf.multiply(y, tf.log(1e-7 + y_hat)), axis=-1)
        n_loss = tf.reduce_mean(tf.multiply(1 - y, tf.log(1e-7 + 1 - y_hat)), axis=-1)
        self.loss = -(0.5 * p_loss + 0.5 * n_loss)
        return self.loss

    # simple training epochs - need to enhance it later
    # this method does not do batch randomization
    # the caller needs to conduct batch randomization
    def train(self, x, y, epoch=100):
        for ep in range(epoch):
            with tf.GradientTape() as g1:
                probas = self(x)
                loss = self.log_loss(probas, y)
                # mask loss recurred thru padding
                mask = y.max(axis=2)
                loss = tf.reduce_mean(tf.boolean_mask(loss, mask))

            print('epoch = ', ep, ' loss = ', loss)
            self.ws = g1.watched_variables()
            self.grads = g1.gradient(loss, self.ws)
            self.optimizer.apply_gradients(zip(self.grads, self.ws))

    # save trained model parameters so we can restored a pretrained model
    def save_model(self, filepath):
        import pickle as p

        init_params = {
            'h_dim':self.h_dim,
            'lr':self.lr,
            'n_classes':self.n_classes,
            'basket_size':self.basket_size,
            'tr_head':self.tr_head,
            'tr_depth':self.tr_depth,
            'fm_dim':self.fm_dim,
            'fm_ratio':self.fm_ratio
        }

        model_params = {
            'z':self.z.numpy(),
            'tr_q':self.tr_q.numpy(),
            'tr_k':self.tr_k.numpy(),
            'tr_v':self.tr_v.numpy(),
            'tr_hw': self.tr_hw.numpy(),
        }

        if self.fm_ratio > 0:
            model_params['fm_b'] = self.fm_b.numpy()
            model_params['fm_w'] = self.fm_w.numpy()
            model_params['fm_v'] = self.fm_v.numpy()

        with open(filepath, "wb") as f:
            p.dump((init_params, model_params), f)

