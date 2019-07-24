#!/usr/bin/env python2
"""Sequence modeling helper functions."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from .attention_cell import AttentionCell
from .nn_helper import bilinear_attention
from .nn_helper import fusion_network
from .nn_helper import single_directional_lstm
from .nn_helper import bilstm_sequence_representation_model
from .nn_helper import layer_norm_fully_connected_layer


def compute_sequence_xentropy_loss(sequence_labels, sequence_logits,
                                   sequence_weights):
    """Computes sequence cross entropy loss.
        tf.nn.sparse_softmax_cross_entropy_with_logits would return a loss the
        same size as the labels.

    Args:
        sequence_labels: A [batch_size, max_sequence_length] sized tensor.
        sequence_logits: A [batch_size, max_sequence_length, word_vocab_size]
                         sized tensor.
        sequence_weights: A [batch_size, max_sequence_length] size matrix.

    Returns:
        A [batch_size, max_sequence_length] sized weighted loss.
    """
    sequence_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=sequence_labels,
        logits=sequence_logits
    )

    return sequence_loss * sequence_weights


def build_unidirectional_sequence_representation(
        input_features, sequence_lengths, rnn_config, mode, cond_feature=None,
        reuse=None, scope=None):
    """Builds single directional sequence representation."""
    with tf.variable_scope(scope or 'single_direct_sequence_rep'):
        if tf.shape(input_features)[-1] != rnn_config.hidden_dim:
            input_features = tf.contrib.layers.fully_connected(
                input_features,
                rnn_config.hidden_dim,
                activation_fn=None
            )

        couple_carry_transform_gates = rnn_config.couple_carry_transform_gates

        rnn_cell = single_directional_lstm(
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
            couple_carry_transform_gates=couple_carry_transform_gates,
            couple_gate_lstm=rnn_config.couple_gate_lstm,
            vr_recurrent=rnn_config.vr_recurrent
        )

        if cond_feature:
            init_states = [[None, None] for _ in xrange(rnn_config.num_hidden)]
            for layer_index in xrange(rnn_config.num_hidden):
                with tf.variable_scope('init_hidden_{0}'.format(layer_index),
                                       reuse=reuse):
                    init_states[
                        layer_index][0] = layer_norm_fully_connected_layer(
                        cond_feature, rnn_config.hidden_dim,
                        activation_fn=tf.nn.tanh
                    )
                    init_states[
                        layer_index][1] = layer_norm_fully_connected_layer(
                        cond_feature, rnn_config.hidden_dim,
                        activation_fn=tf.nn.tanh
                    )
            init_state = tuple([
                tf.nn.rnn_cell.LSTMStateTuple(init_s, init_h)
                for init_s, init_h in init_states
            ])

            hiddens, state = tf.nn.dynamic_rnn(
                rnn_cell, input_features, sequence_length=sequence_lengths,
                initial_state=init_state
            )
        else:
            hiddens, state = tf.nn.dynamic_rnn(
                rnn_cell, input_features, sequence_length=sequence_lengths,
                dtype=tf.float32
            )

    return hiddens, state


def build_rnn_sentence_representation(input_embedding, sentence_length,
                                      rnn_config, mode, reuse=None):
    """Builds contextual sentence representation using RNN."""
    with tf.variable_scope('rnn_sentence_representation', reuse=reuse):
        if tf.shape(input_embedding)[-1] != rnn_config.hidden_dim:
            input_embedding = tf.contrib.layers.fully_connected(
                input_embedding,
                rnn_config.hidden_dim,
                activation_fn=None
            )

        # A [batch_size, max_sequence_length, rnn_hidden_dim] sized tensor.
        context_word_embedding, states = bilstm_sequence_representation_model(
            input_embedding, sentence_length, rnn_config, mode, minus_feat=False
        )

        fw_states, bw_states = states

        # Multi-layer states are tupled sequentially.
        last_fw_hidden = fw_states[-1].h
        last_bw_hidden = bw_states[-1].h

        # A [batch_size, rnn_hidden_dim] sized matrix.
        sentence_rep = tf.reduce_sum(context_word_embedding, axis=1)
        sentence_rep /= tf.expand_dims(tf.to_float(sentence_length), axis=-1)

        last_hidden_sentence_rep = tf.concat([last_fw_hidden, last_bw_hidden],
                                             axis=-1)

        return sentence_rep, last_hidden_sentence_rep, context_word_embedding


def recurrent_attention_predictor(input_features, sequence_length,
                                  max_sequence_length, tag_embedding,
                                  num_tag_class, tag_dim, rnn_config, mode,
                                  hard_attention=False,
                                  recurrent_att_type='bilinear', scope=None,
                                  reuse=None):
    """Performs recurrent attention prediction."""

    with tf.variable_scope(
            scope or 'recurrent_attention_predictor', reuse=reuse):

        batch_size = tf.shape(input_features)[0]

        couple_carry_transform_gates = rnn_config.couple_carry_transform_gates
        rnn_cell = single_directional_lstm(
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
            couple_carry_transform_gates=couple_carry_transform_gates,
            couple_gate_lstm=rnn_config.couple_gate_lstm,
            vr_recurrent=rnn_config.vr_recurrent
        )

        attention_predictor = AttentionCell(
            rnn_cell, tag_embedding, num_tag_class, tag_dim,
            rnn_config.hidden_dim, hard_attention=hard_attention,
            attention_type=recurrent_att_type
        )

        tagger_hiddens, _ = tf.nn.dynamic_rnn(
            attention_predictor, input_features,
            sequence_length=sequence_length, dtype=tf.float32)

        flatten_tagger_hiddens = tf.reshape(
            tagger_hiddens, [-1, rnn_config.hidden_dim])
        flatten_tagger_logits, _ = attention_predictor.single_attention(
            flatten_tagger_hiddens
        )

        tagger_logits = tf.reshape(
            flatten_tagger_logits,
            [batch_size, max_sequence_length, num_tag_class]
        )
    return tagger_logits, tagger_hiddens


def build_sequence_encoder_network(input_embedding, sequence_length,
                                   mode_embedding, num_latent_class, latent_dim,
                                   rnn_config, mode, project_method='bilinear',
                                   last_hidden=False, name=None, reuse=None):
    """Builds a sequence encoder network."""
    scope_name = name or 'encoder_network'
    with tf.variable_scope(scope_name, reuse=reuse):
        (
            mean_word_hiddens, last_word_hiddens, _
        ) = build_rnn_sentence_representation(
            input_embedding, sequence_length, rnn_config, mode
        )

        if last_hidden:
            sentence_features = last_word_hiddens
        else:
            sentence_features = mean_word_hiddens

        with tf.variable_scope('latent_class_logits'):
            if project_method == 'bilinear':
                _, y_logits, prob_y = bilinear_attention(
                    sentence_features, mode_embedding, latent_dim,
                    hard_attention=False, activation_fn=None
                )
            else:
                y_logits = tf.contrib.layers.fully_connected(
                    sentence_features, num_latent_class, activation_fn=None
                )
                prob_y = tf.nn.softmax(y_logits)

        return y_logits, prob_y, sentence_features


def build_sequence_generation_network(input_features, latent_features,
                                      latent_feature_dim, sequence_lengths,
                                      max_sequence_length, word_vocab_size,
                                      rnn_config, mode, reuse=None):
    """Network for sequence generation."""
    if mode == 'TRAIN_EVAL':
        reuse = True

    with tf.variable_scope('generation_network', reuse=reuse):
        exp_latent_features = tf.tile(tf.expand_dims(latent_features, axis=1),
                                      [1, max_sequence_length, 1])

        fused_features = fusion_network(
            exp_latent_features, input_features, rnn_config.hidden_dim,
            comp_method='add', activation_fn=None, element_wise=False,
            reuse=reuse
        )

        couple_carry_transform_gates = rnn_config.couple_carry_transform_gates

        gen_rnn_cell = single_directional_lstm(
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
            couple_carry_transform_gates=couple_carry_transform_gates,
            couple_gate_lstm=rnn_config.couple_gate_lstm,
            vr_recurrent=rnn_config.vr_recurrent
        )

        init_states = [[None, None] for _ in xrange(rnn_config.num_hidden)]
        for layer_index in xrange(rnn_config.num_hidden):
            with tf.variable_scope('init_hidden_{0}'.format(layer_index),
                                   reuse=reuse):
                init_states[layer_index][0] = tf.contrib.layers.fully_connected(
                    latent_features, rnn_config.hidden_dim,
                    activation_fn=tf.nn.tanh
                )
                init_states[layer_index][1] = tf.contrib.layers.fully_connected(
                    latent_features, rnn_config.hidden_dim,
                    activation_fn=tf.nn.tanh
                )

        init_state = tuple([
            tf.nn.rnn_cell.LSTMStateTuple(init_s, init_h)
            for init_s, init_h in init_states
        ])

        hiddens, _ = tf.nn.dynamic_rnn(gen_rnn_cell, fused_features,
                                       sequence_length=sequence_lengths,
                                       initial_state=init_state)

        with tf.variable_scope('sequence_logits'):
            sentence_sequence_logits = fusion_network(
                exp_latent_features, hiddens, word_vocab_size,
                comp_method='add', activation_fn=None, element_wise=False)

    return sentence_sequence_logits
