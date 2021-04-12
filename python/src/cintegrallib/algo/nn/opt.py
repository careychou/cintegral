import tensorflow as tf
import numpy as np

###
# utility functions for differential optimization from tensorflow
# website
#

# newton root finding
# Param:
#   grads1: dy/dx
#   grads2: hessian matrix d2y/dx2

def newton_root(grads1, grads2):
    gshape = grads1.shape
    nparam = tf.reduce_prod(gshape)
    grads1 = tf.reshape(grads1, [nparam, 1])
    grads2 = tf.reshape(grads2, [nparam, nparam])
    # regularize for numeric stability
    eps = 1e-3
    eye_eps = tf.eye(grads2.numpy().shape[0])*eps
    # find root: x(k+1) = x(k) - f'(x)/f''(x)
    update = tf.linalg.solve(grads2 + eye_eps, grads1)

    return tf.reshape(update, gshape)
