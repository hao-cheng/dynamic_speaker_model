#!/usr/bin/env python2
"""Neural network helper functions."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys

import tensorflow as tf


# Defines global variables.
VERY_NEG = -1e6


def build_embeddings(embed_size, embed_dim, embed_name,
                     pretrain_embed=None, trainable=True):
    """Builds embedding variables.

    Args:
        embed_size: Integer for the row size of the embedding matrix.
        embed_dim: Integer for the col size of the embedding matrix.
        embed_name: String for the prefix of the name scope.
        pretrain_embed: A [embed_size, embed_dim] sized pretrain embedding.
        trainable: Bool indicating whether the embedding is trainable.

    Returns:
        An embedding variable.
    """
    with tf.variable_scope('{0}_embedding'.format(embed_name)):
        if pretrain_embed is None:
            embedding = tf.get_variable(
                '{0}_embedding'.format(embed_name),
                shape=[embed_size, embed_dim],
                dtype=tf.float32,
                trainable=trainable
            )
        else:
            print('Loading pretrained embedding for {0}'.format(embed_name),
                  file=sys.stderr)
            src_shape = pretrain_embed.shape
            if src_shape[0] != embed_size or src_shape[1] != embed_dim:
                print('specified embedding shape [{0}, {1}]'.format(embed_size,
                                                                    embed_dim),
                      file=sys.stderr)
                print('given embedding shape [{0}, {1}]'.format(src_shape[0],
                                                                src_shape[1]),
                      file=sys.stderr)
                raise ValueError('{0}_embedding size mispecified')

            embedding = tf.get_variable(
                '{0}_embedding'.format(embed_name),
                initializer=tf.constant(pretrain_embed),
                dtype=tf.float32,
                trainable=trainable
            )

        return embedding


# class RNNConfig(object):
#     """Configuration object for Recurrent Neural Network."""
#
#     # Those slots values are determined by function single_directional_lstm.
#     __slots__ = ('num_hidden', 'hidden_dim', 'residual_rnn', 'residual_method',
#                  'keep_prob', 'input_keep_prob', 'state_keep_prob',
#                  'forget_bias', 'carry_bias_init',
#                  'couple_carry_transform_gates', 'couple_gate_lstm',
#                  'vr_recurrent')
#
#     def __init__(self):
#         """Initialization."""
#         self.num_hidden = 0
#         self.hidden_dim = 0
#
#         self.residual_rnn = True
#         self.residual_method = None
#         self.keep_prob = 1.0
#         self.input_keep_prob = 1.0
#         self.state_keep_prob = 1.0
#         self.forget_bias = 1.0
#         self.carry_bias_init = 1.0
#         self.couple_carry_transform_gates = True
#         self.couple_gate_lstm = False
#         self.vr_recurrent = True
#
#     def parse_from_model_config(self, model_config):
#         """Parses parameters from a ModelConfig object."""
#         for attr in self.__slots__:
#             if attr in model_config.__slots__:
#                 setattr(self, attr, getattr(model_config, attr))
#             else:
#                 raise ValueError('RNNConfig.{0} not found!'.format(attr))
#         return True


def single_directional_lstm(num_hidden, hidden_dim, residual_rnn,
                            residual_method, keep_prob, mode,
                            input_keep_prob=None, state_keep_prob=None,
                            output_keep_prob=None,
                            forget_bias=1.0, carry_bias_init=1.0,
                            couple_carry_transform_gates=True,
                            couple_gate_lstm=False, vr_recurrent=True,
                            scope=None):
    """Builds a single directional LSTM recurrent neuralnet.

    Args:
        num_hidden: Integer for the num hidden layers.
        hidden_dim: Integer for the hidden layer dimension.
        residual_rnn: Bool whether to use residual RNN.
        residual_method: String for the residual method {'highway', 'residual'}.
        keep_prob: Float for the dropout keep rate if specified.
        mode: String for the mode {'TRAIN', 'TRAIN_EVAL', 'EVAL', 'INFER'}.
        input_keep_prob: Float for the RNN input dropout keep rate if specified.
        state_keep_prob: Float for the RNN state dropout keep rate if specified.
        output_keep_prob: Float for the RNN output dropout keep rate if specified.
        carry_bias_init: Float for the the Highway wrapper.
        couple_carry_transform_gates: Bool for the Highway Wrapper.
        couple_gate_lstm: Bool whether to use CoupledInputForgetGateLSTMCell.
        vr_recurrent: Bool whether to use variational Dropout.
        scope: String for the variable scope.

    Returns:
        recurrent_net: The specified recurrent neural net.
    """
    keep_prob = keep_prob if mode == 'TRAIN' else 1.0

    if input_keep_prob:
        input_keep_prob = input_keep_prob if mode == 'TRAIN' else 1.0
    else:
        input_keep_prob = keep_prob if mode == 'TRAIN' else 1.0

    if state_keep_prob:
        state_keep_prob = state_keep_prob if mode == 'TRAIN' else 1.0
    else:
        state_keep_prob = keep_prob if mode == 'TRAIN' else 1.0

    if output_keep_prob:
        output_keep_prob = output_keep_prob if mode == 'TRAIN' else 1.0
    else:
        output_keep_prob = keep_prob if mode == 'TRAIN' else 1.0

    if not vr_recurrent:
        state_keep_prob = 1.0

    basic_cell = tf.contrib.rnn.BasicLSTMCell
    if couple_gate_lstm:
        basic_cell = tf.contrib.rnn.CoupledInputForgetGateLSTMCell

    def _build_highway_rnn_cell():
        """Builds highway RNN cell."""
        if mode == 'TRAIN':
            return tf.contrib.rnn.DropoutWrapper(
                tf.contrib.rnn.HighwayWrapper(
                    basic_cell(hidden_dim, forget_bias=forget_bias),
                    couple_carry_transform_gates=couple_carry_transform_gates,
                    carry_bias_init=carry_bias_init
                ),
                input_keep_prob=input_keep_prob,
                output_keep_prob=output_keep_prob,
                state_keep_prob=state_keep_prob,
                dtype=tf.float32,
                variational_recurrent=vr_recurrent,
                input_size=hidden_dim
            )

        return tf.contrib.rnn.HighwayWrapper(
            basic_cell(hidden_dim, reuse=True if mode == 'TRAIN_EVAL' else False),
            couple_carry_transform_gates=couple_carry_transform_gates,
            carry_bias_init=carry_bias_init
        )

    def _build_residual_rnn_cell():
        """Builds Residual RNN cell."""
        if mode == 'TRAIN':
            return tf.contrib.rnn.DropoutWrapper(
                tf.contrib.rnn.ResidualWrapper(
                    basic_cell(hidden_dim, forget_bias=forget_bias)
                ),
                input_keep_prob=input_keep_prob,
                output_keep_prob=output_keep_prob,
                state_keep_prob=state_keep_prob,
                dtype=tf.float32,
                variational_recurrent=vr_recurrent,
                input_size=hidden_dim
            )

        return tf.contrib.rnn.ResidualWrapper(
            basic_cell(hidden_dim, reuse=True if mode == 'TRAIN_EVAL' else False)
        )

    def _build_rnn_cell():
        """Builds RNN cell."""
        if mode == 'TRAIN':
            return tf.contrib.rnn.DropoutWrapper(
                basic_cell(hidden_dim, forget_bias=forget_bias),
                input_keep_prob=input_keep_prob,
                output_keep_prob=output_keep_prob,
                state_keep_prob=state_keep_prob,
                dtype=tf.float32,
                variational_recurrent=vr_recurrent,
                input_size=hidden_dim
            )

        return basic_cell(
            hidden_dim, reuse=True if mode == 'TRAIN_EVAL' else False)

    def _build_rnn_cell_model():
        """Builds RNN cell based on the residual_method."""
        if residual_rnn:
            if residual_method == 'highway':
                return _build_highway_rnn_cell()
            elif residual_method == 'residual':
                return _build_residual_rnn_cell()
            else:
                raise ValueError(
                    'Unknown residual_method:{0}'.format(residual_method))

        return _build_rnn_cell()

    with tf.variable_scope(scope or 'single_directional_lstm'):
        recurrent_net = tf.contrib.rnn.MultiRNNCell([
            _build_rnn_cell_model()
            for _ in xrange(num_hidden)
        ])

    return recurrent_net


def bilstm_sequence_representation_model(emb_inp, seq_len, rnn_config, mode,
                                         minus_feat=True, scope=None):
    """Creates Bi-directional LSTM representation model.

    Args:
        emb_inp: [batch_size, max_num_step, input_dim] sized input tensor.
        seq_len: [batch_size] sized tensor contains sequence length info.
        rnn_config: RNNConfig object contains parameters for RNN model.
        mode: String for the mode from {'TRAIN', 'TRAIN_EVAL', 'INFER', 'EVAL'}.
        scope: String for the variable scope.

    Returns:
        Concated output of forward and backward LSTM. And the final states.
    """
    with tf.variable_scope(scope or 'bilstm_sequence_representation_model'):
        with tf.variable_scope('fw'):
            fw_recurrent_net = single_directional_lstm(
                rnn_config.num_hidden,
                rnn_config.hidden_dim,
                rnn_config.residual_rnn,
                rnn_config.residual_method,
                rnn_config.keep_prob,
                mode,
                input_keep_prob=rnn_config.input_keep_prob,
                state_keep_prob=rnn_config.state_keep_prob,
                output_keep_prob=rnn_config.output_keep_prob,
                forget_bias=rnn_config.forget_bias,
                carry_bias_init=rnn_config.carry_bias_init,
                couple_carry_transform_gates=rnn_config.couple_carry_transform_gates,
                couple_gate_lstm=rnn_config.couple_gate_lstm,
                vr_recurrent=rnn_config.vr_recurrent,
                scope=tf.get_variable_scope()
            )
        with tf.variable_scope('bw'):
            bw_recurrent_net = single_directional_lstm(
                rnn_config.num_hidden,
                rnn_config.hidden_dim,
                rnn_config.residual_rnn,
                rnn_config.residual_method,
                rnn_config.keep_prob,
                mode,
                input_keep_prob=rnn_config.input_keep_prob,
                state_keep_prob=rnn_config.state_keep_prob,
                output_keep_prob=rnn_config.output_keep_prob,
                forget_bias=rnn_config.forget_bias,
                carry_bias_init=rnn_config.carry_bias_init,
                couple_carry_transform_gates=rnn_config.couple_carry_transform_gates,
                couple_gate_lstm=rnn_config.couple_gate_lstm,
                vr_recurrent=rnn_config.vr_recurrent,
                scope=tf.get_variable_scope()
            )

        rnn_outputs, rnn_stats = tf.nn.bidirectional_dynamic_rnn(
            cell_fw=fw_recurrent_net,
            cell_bw=bw_recurrent_net,
            inputs=emb_inp,
            sequence_length=seq_len,
            time_major=False,
            swap_memory=False,
            dtype=tf.float32,
            scope=tf.get_variable_scope()
        )

        output_feat = tf.concat(rnn_outputs, -1)
        if minus_feat:
            minus_output = rnn_outputs[0] - rnn_outputs[1]
            output_feat = tf.concat([output_feat, minus_output], -1)
        return output_feat, rnn_stats


def layer_normalize(x, output_dim, eps=1e-6, scope=None):
    """Layer normalization forward."""
    with tf.variable_scope(scope or 'layer_normalize'):
        a = tf.get_variable('a', dtype=tf.float32,
                            initializer=tf.ones([output_dim], dtype=tf.float32)
                            )
        b = tf.get_variable('b', dtype=tf.float32,
                            initializer=tf.zeros([output_dim], dtype=tf.float32)
                            )

        # This assume only the last dimension needs to be normalized.
        mean = tf.stop_gradient(tf.reduce_mean(x, axis=-1, keepdims=True))
        std = tf.stop_gradient(tf.sqrt(tf.reduce_mean(
            (x - mean) ** 2, axis=-1, keepdims=True)))

        return a * (x - mean) / (std + eps) + b


def layer_norm_fully_connected_layer(
        x, hidden_dim, eps=1e-6,
        biases_initializer=tf.constant_initializer(0.1),
        activation_fn=None, scope=None):
    """A fully_connected layer with layer normalization."""
    with tf.variable_scope(scope or 'layer_norm_fully_connected_layer'):

        y = tf.contrib.layers.fully_connected(
            x, hidden_dim, activation_fn=activation_fn,
            biases_initializer=biases_initializer
        )

        return layer_normalize(y, hidden_dim)


def highway_net(x, y, complement=True, carry_bias_init=1.0, element_wise=False,
                scope=None):
    """Givens two inputs with the same last dimension, computes the highway
        output. It's a gating mechanism for forward passing.

    Args:
        x: Tensor input for computing the transform and carry gates.
        y: Tensor with the same shape as x, usually y = f(x).
        complement: Bool whether to use complement carry gating.
        carry_bias_init: Float for init bias, used if complement=True.
        element_wise: Bool whether to perform element-wise gating or block-wise.
        scope: String for the variable scope.

    Returns:
        output: Tensor with the same shape as x and y.
    """
    with tf.variable_scope(scope or 'highway_net'):
        if element_wise:
            gate_size = x.get_shape()[-1].value
        else:
            gate_size = 1
        carry = tf.contrib.layers.fully_connected(
            x, gate_size, activation_fn=tf.nn.sigmoid,
            biases_initializer=tf.constant_initializer(
                carry_bias_init, dtype=tf.float32)
            if complement else tf.zeros_initializer(),
            scope='carry_forward'
        )
        if complement:
            transform = 1 - carry
        else:
            transform = tf.contrib.layers.fully_connected(
                x, gate_size, activation_fn=tf.nn.sigmoid,
                scope='transform_forward'
            )
    return x * carry + y * transform


def fusion_network(input_a, input_b, output_dim, comp_method='concat',
                   activation_fn=None, element_wise=False, reuse=False,
                   layer_norm=False, scope=None):
    """Fuses two inputs based on different composition methods.

    Args:
        input_a: Tensor for one input.
        input_b: Tensor for another input, input_a and input_b can be only
                    different for the last dimension.
        output_dim: Integer for the output last dimension.
        comp_method: String for the composition method.
        activation_fn: Activation function for projection if needed.
        gate_input: Bool whether to use input-wise gating.

    Returns:
        output: Composed tensor with the same dimensions but last dim as inputs.
    """
    if len(input_a.get_shape()) != len(input_b.get_shape()):
        raise ValueError('Both inputs must be of rank')

    project_a = project_b = False
    if input_a.get_shape()[-1] != output_dim:
        project_a = True
    if input_b.get_shape()[-1] != output_dim:
        project_b = True

    if layer_norm:
        transfer_func = layer_norm_fully_connected_layer
    else:
        transfer_func = tf.contrib.layers.fully_connected

    with tf.variable_scope(scope or 'fusion_network', reuse=reuse):
        if comp_method == 'concat':
            # output = tf.contrib.layers.fully_connected(
            #     tf.concat([input_a, input_b], -1), output_dim,
            #     activation_fn=activation_fn,
            #     scope='concat_projection'
            # )
            output = transfer_func(
                tf.concat([input_a, input_b], -1), output_dim,
                activation_fn=activation_fn,
                scope='concat_projection'
            )
        else:
            if project_a:
                # input_a = tf.contrib.layers.fully_connected(
                #     input_a, output_dim, activation_fn=activation_fn,
                #     biases_initializer=None,
                #     scope='a_projection'
                # )
                input_a = transfer_func(
                    input_a, output_dim, activation_fn=activation_fn,
                    biases_initializer=None,
                    scope='a_projection'
                )

            if project_b:
                # input_b = tf.contrib.layers.fully_connected(
                #     input_b, output_dim, activation_fn=activation_fn,
                #     biases_initializer=None,
                #     scope='b_projection'
                # )
                input_b = transfer_func(
                    input_b, output_dim, activation_fn=activation_fn,
                    biases_initializer=None,
                    scope='b_projection'
                )

            if comp_method == 'highway':
                output = highway_net(
                    input_a, input_b, element_wise=element_wise)
            elif comp_method == 'add':
                output = input_a + input_b
            elif comp_method == 'minus':
                output = input_a - input_b
            else:
                raise ValueError('Unknown comp_method: {0}'.format(
                    comp_method))

    return output



def bilinear_attention(query_tensor, mem_tensor, mem_dim,
                       valid_memory_length=None,
                       max_mem_seq_len=None, mode=None,
                       hard_attention=False, activation_fn=None,
                       project_query=True, project_mem=True, scope=None):
    """Computes bilinear attention between query and memory.

    Args:
        query_tensor: A [batch_size, query_seq_len, query_dim] sized tensor.
        mem_tensor: A [batch_size, mem_seq_len, mem_dim] sized tensor.
        valid_memory_length: A [batch_size, 1] sized tensor.
        mode: String for the mode.
        hard_attention: Bool whether to forward argmax or softmax.
        activation_fn: Function object to be used for projection.
        project_query: Bool whether to project the query_tensor to mem_dim.
        project_mem: Bool whether to project the mem_tensor to mem_dim.
        scope: String for the variable scope.

    Returns:
        mix_mem_tensor: A [batch_size, query_seq_len, mem_dim] sized tensor.
        sim_logits: A [batch_size, query_seq_len, mem_seq_len] sized scores.
        sim_softmax: A [batch_size, quer_seq_len, mem_seq_len] sized probs.
    """
    with tf.variable_scope(scope or 'bilinear_attention'):
        # Projects the query tensor if desired.
        if project_query:
            query_tensor = tf.contrib.layers.fully_connected(
                query_tensor, mem_dim, activation_fn=activation_fn,
                scope='query_projection'
            )

        if project_mem:
            mem_tensor = tf.contrib.layers.fully_connected(
                mem_tensor, mem_dim, activation_fn=activation_fn,
                scope='mem_projection'
            )
        # This mulitplication results in a [batch_size, query_seq_len,
        # mem_seq_len] sized tensor of unnormalized similarity.
        sim_logits = tf.matmul(query_tensor, mem_tensor, transpose_b=True)
        if valid_memory_length:
            # A [batch_size, mem_seq_length] sized boolean tensor.
            valid_mem_seq = tf.sequence_mask(valid_memory_length,
                                             maxlen=max_mem_seq_len)
            mem_seq_mask = tf.where(
                valid_mem_seq,
                tf.zeros_like(valid_mem_seq, dtype=tf.float32),
                VERY_NEG * tf.ones_like(valid_mem_seq, dtype=tf.float32))

            # Expands the query_seq_length dimension for broadcasting.
            mem_seq_mask = tf.expand_dims(mem_seq_mask, 1)
            sim_logits += mem_seq_mask

        sim_logits /= tf.sqrt(tf.to_float(mem_dim))
        sim_softmax = tf.nn.softmax(sim_logits)

        if hard_attention:
            batch_size = tf.shape(mem_tensor)[0]
            max_indices = tf.argmax(sim_softmax, axis=-1)
            offset = tf.expand_dims(tf.range(tf.to_int64(batch_size),
                                         dtype=tf.int64), -1)
            max_indices += offset
            max_indices = tf.stop_gradient(max_indices)

            flat_mem_tensor = tf.reshape(mem_tensor, [-1, mem_dim])
            mix_mem_tensor = tf.nn.embedding_lookup(flat_mem_tensor,
                                                    max_indices)
        else:
            mix_mem_tensor = tf.matmul(sim_softmax, mem_tensor)

    return mix_mem_tensor, sim_logits, sim_softmax
