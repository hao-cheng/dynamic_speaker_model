#!/usr/bin/env python2
"""Model function related helpers."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys

import tensorflow as tf


def build_optimizer(opt_config, learning_rate):
    """Builds optimzier for the TF model.

    Args:
        opt_config: A OptConfig object containing optimization related info.
        learning_rate: A TF.placeholder object.

    Returns a specified TF optimizer.
    """
    if opt_config.opt_method == 'SGD':
        print('Using SGD as the optimizer', file=sys.stderr)
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    elif opt_config.opt_method == 'Adam':
        print('Using Adam as the optimizer', file=sys.stderr)
        optimizer = tf.train.AdamOptimizer(
            learning_rate, beta1=opt_config.adam_beta1,
            beta2=opt_config.adam_beta2, epsilon=opt_config.adam_epsilon
        )
    else:
        raise ValueError(
            'Unknown optimization method {0}!'.format(opt_config.opt_method))
    return optimizer


def compute_and_apply_gradient(loss, optimizer, clip_value=0.0):
    """Computes and applies gradients."""
    grads_and_vars_orig = optimizer.compute_gradients(loss)

    if clip_value > 0:
        print('Carries out gradient clipping!', file=sys.stderr)
        grad_list, var_list = zip(*grads_and_vars_orig)
        clipped_grad_list, _ = tf.clip_by_global_norm(grad_list, clip_value)
        grads_and_vars = zip(clipped_grad_list, var_list)
    else:
        grads_and_vars = grads_and_vars_orig

    opt_op = optimizer.apply_gradients(grads_and_vars)

    return opt_op
