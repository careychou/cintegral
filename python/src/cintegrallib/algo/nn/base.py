import tensorflow as tf

# originally proposed to use under domain adversarial to maximize the loss of cross domain classification
#
@tf.custom_gradient
def gradient_reverse(x):
    y = tf.identity(x)
    def inv_grad(dy):
        return -dy
    return y, inv_grad