#!/usr/bin/env python2
"""This implements RNN cell with attention.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys

import tensorflow as tf
from tensorflow.python.ops.rnn_cell import RNNCell


class AttentionCell(RNNCell):
    """RNN cell supports attention mechanism.
    """

    def __init__(self, cell, memory, num_head, memory_dim, hidden_dim,
                 hard_attention=False, attention_type='bilinear'):
        """Initializer function.

        Args:
            cell: A RNNCell object.
            num_head: Integer for the number of heads.
            memory_dim: Integer for the memory dimension.
            hidden_dim: Integer for the hidden layer size.
            hard_attention: Bool whether to carry out hard attention.
            attention_type: String for attention score function,
                            {'bilienar', 'tanh'}
        """
        self._cell = cell
        self._memory = memory
        self._num_head = num_head
        self._memory_dim = memory_dim
        self._hidden_dim = hidden_dim
        self._hard_attention = hard_attention
        if attention_type == 'bilinear':
            self._attention_score_func = self.bilinear_score
            self._score_offset = tf.sqrt(tf.to_float(self._hidden_dim))
        elif attention_type == 'tanh':
            self._attention_score_func = self.tanh_score
        else:
            print('Unknown attention score type {0}'.format(attention_type))
            sys.exit(1)

    @property
    def state_size(self):
        """Returns the state size."""
        return self._cell.state_size

    @property
    def output_size(self):
        """Returns the output size."""
        return self._cell.output_size

    def __call__(self, inputs, state, scope=None):
        """Computes a single forward."""
        with tf.variable_scope(scope or 'attention_cell'):
            _, mix_mem = self.single_attention(inputs)
            new_inputs = tf.concat([mix_mem, inputs], axis=1)
            projected_inputs = tf.contrib.layers.fully_connected(
                new_inputs, self._hidden_dim, activation_fn=None
            )
            return self._cell(projected_inputs, state)

    def tanh_score(self, a, b, scope=None):
        """Computes the similarity score in NMT fashion.

        Args:
            a: [batch_size, hidden_dim] sized tensor.
            b: [batch_size, num_head, 1, hidden_dim] sized tensor.

        Returns:
            s: [batch_size, num_head] sized tensor for unnormalized score.
        """
        with tf.variable_scope(scope or 'tanh_score'):
            att_v = tf.get_variable(
                'att_v', shape=[self._hidden_dim], dtype=tf.float32
            )
            a = tf.reshape(a, [-1, 1, 1, self._hidden_dim])
            s = tf.reduce_sum(att_v * tf.tanh(a + b), [2, 3])

        return s

    def bilinear_score(self, a, b, scope=None):
        """Computes the similarity score in bilinear.

        Args:
            a: [batch_size, hidden_dim] sized tensor.
            b: [batch_size, num_head, 1, hidden_dim] sized tensor.

        Returns:
            s: [batch_size, num_head] sized tensor for unnormalized score.
        """
        with tf.variable_scope(scope or 'bilinear_score'):
            a = tf.reshape(a, [-1, 1, 1, self._hidden_dim])
            s = tf.reduce_sum(a * b / self._score_offset, [2, 3])

        return s

    def single_attention(self, query, scope=None):
        with tf.variable_scope(scope or 'attention_mlp'):
            batch_size = tf.shape(query)[0]
            hidden_dim = self._hidden_dim

            project_feat = tf.tile(
                tf.reshape(
                    tf.contrib.layers.fully_connected(
                        self._memory, hidden_dim, activation_fn=None,
                        biases_initializer=None
                    ),
                    [-1, self._num_head, 1, hidden_dim]),
                [batch_size, 1, 1, 1]
            )

            y = tf.contrib.layers.fully_connected(
                query, hidden_dim, activation_fn=None,
                biases_initializer=None
            )
            s = self._attention_score_func(y, project_feat)
            a_score = tf.nn.softmax(s)
            ascore_reshape = tf.reshape(
                a_score,
                [-1, self._num_head, 1, 1]
            )

            # Reshapes for return
            unormalized_ascore = tf.reshape(s, [-1, self._num_head])

            if self._hard_attention:
                indices = tf.argmax(unormalized_ascore, axis=-1)
                weight_sum = tf.nn.embedding_lookup(
                    self._memory, indices)
            else:
                tiled_memory = tf.tile(
                    tf.reshape(self._memory,
                               [1, self._num_head, 1, self._memory_dim]),
                    [batch_size, 1, 1, 1]
                )

                weight_sum = tf.reduce_sum(
                    ascore_reshape * tiled_memory,
                    [1, 2]
                )

            mixture_embeds = tf.reshape(weight_sum, [-1, self._memory_dim])

        return unormalized_ascore, mixture_embeds
