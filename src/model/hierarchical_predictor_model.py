#!/usr/bin/env python2
"""Hierarchical user model."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys

import tensorflow as tf

from .configure import RNNConfig
from .nn_helper import build_embeddings
from .nn_helper import fusion_network
from .model_helper import build_optimizer
from .model_helper import compute_and_apply_gradient
from .sequence_model_helper import build_sequence_encoder_network
from .sequence_model_helper import build_unidirectional_sequence_representation
from .sequence_model_helper import recurrent_attention_predictor


class HierarchicalPredictorModel(object):
    """A hierarchical predictor model."""

    # Variable name list for fetching.
    _train_eval_fetch_var_names_ = (
        'y_logits', 'y_probs', 'loss_to_opt', 'dialog_act_logits',
    )

    def __init__(self, config, mode, embedding_to_load=None):
        """Initialization."""
        # Configs the model.
        self._config_model(config)

        # Constructs the model.
        self._inputs, self._outputs = self._construct_model(
            embedding_to_load, mode)

        # Null operation.
        self._no_op = tf.no_op()

    def __str__(self):
        """Function for printing the info of the model."""
        skip_vars = set([
            '_inputs', '_outputs', '_no_op', '_rnn_config',
            '_user_state_rnn_config', '_user_utterance_rnn_config',
            'conversation_context_rnn_config'
        ])

        model_config_str = '\n'.join([
            'model.{0}={1}'.format(attr, getattr(self, attr))
            for attr in self.__dict__ if attr not in skip_vars
        ])

        user_state_rnn_config_str = '\n'.join([
            'model.user_state_rnn_config.{0}={1}'.format(
                attr, getattr(self._user_state_rnn_config, attr))
            for attr in self._user_state_rnn_config.__slots__
        ])

        user_utterance_rnn_config_str = '\n'.join([
            'model.user_utterance_rnn_config.{0}={1}'.format(
                attr, getattr(self._user_utterance_rnn_config, attr))
            for attr in self._user_utterance_rnn_config.__slots__
        ])

        conversation_context_rnn_config_str = '\n'.join([
            'model.conversation_context_rnn_config.{0}={1}'.format(
                attr, getattr(self._conversation_context_rnn_config, attr))
            for attr in self._conversation_context_rnn_config.__slots__
        ])

        return '\n'.join([
            'Model configuraton:', model_config_str, user_state_rnn_config_str,
            user_utterance_rnn_config_str, conversation_context_rnn_config_str
        ])

    def _config_model(self, config):
        """Constructs the model parameters from the ModelConfig object."""
        self._word_vocab_size = config.word_vocab_size
        self._word_embed_dim = config.word_embed_dim
        self._train_word_embed = config.train_word_embed

        self._adapt_user_model = config.adapt_user_model
        self._num_dialog_act = config.num_dialog_act
        self._dialog_act_embed_dim = config.dialog_act_embed_dim

        self._keep_prob = config.keep_prob

        self._num_latent_class = config.num_latent_class
        self._latent_feature_dim = config.latent_feature_dim

        self._user_state_rnn_config = RNNConfig()
        self._user_state_rnn_config.parse_from_json_string(
            config.user_state_rnn_config.to_json_string()
        )

        self._user_utterance_rnn_config = RNNConfig()
        self._user_utterance_rnn_config.parse_from_json_string(
            config.user_utterance_rnn_config.to_json_string()
        )

        self._conversation_context_rnn_config = RNNConfig()
        self._conversation_context_rnn_config.parse_from_json_string(
            config.conversation_context_rnn_config.to_json_string()
        )

    def _construct_model(self, embedding_to_load, mode):
        """Constructs the model based on the config."""
        # Note: for now, the batch_size is equal to the number of turns.
        # A [batch_size, sentence_length] sized matrix containing word indices
        # for each sentence.
        sentence_a_indices = tf.placeholder(
            tf.int32, name='sentence_a_indices', shape=[None, None])

        # A [batch_size] sized vector containing the length of each sentence.
        sentence_a_lengths = tf.placeholder(
            tf.int32, name='sentence_a_lengths', shape=[None])

        # A [batch_size] sized vector containing the caller a turn indices.
        map_ab_indices_to_turn_indices = tf.placeholder(
            tf.int32, name='map_ab_indices_to_turn_indices', shape=[None])

        # A [batch_size, sentence_length] sized matrix containing word indices
        # for each sentence.
        sentence_b_indices = tf.placeholder(
            tf.int32, name='sentence_b_indices', shape=[None, None])

        # A [batch_size] sized vector containing the length of each sentence.
        sentence_b_lengths = tf.placeholder(
            tf.int32, name='sentence_b_lengths', shape=[None])

        # Scalar for the maximum sentence length in the current batch.
        max_sentence_length = tf.placeholder(
            tf.int32, name='max_sentence_length', shape=()
        )

        # Scalar for the current batch size.
        batch_size = tf.placeholder(tf.int32, name='batch_size', shape=())

        with tf.device('/cpu:0'), tf.variable_scope('input_embeddings'):
            word_embedding = build_embeddings(
                self._word_vocab_size, self._word_embed_dim, 'word',
                pretrain_embed=embedding_to_load.get(
                    'word_embedding', None) if embedding_to_load else None,
                trainable=self._train_word_embed
            )

            user_embedding = build_embeddings(
                self._num_latent_class, self._latent_feature_dim, 'user',
                trainable=True
            )

        sentence_a_embedding = tf.nn.embedding_lookup(
            word_embedding, sentence_a_indices
        )
        sentence_b_embedding = tf.nn.embedding_lookup(
            word_embedding, sentence_b_indices
        )

        def _reset_mode(mode_str):
            """Resets the mode string."""
            if not self._adapt_user_model or mode_str == 'TRAIN_EVAL':
                return mode_str
            return 'EVAL'

        # Encodes the user utterance information.
        (
            y_a_logits, y_a_probs, sentence_a_features
        ) = build_sequence_encoder_network(
            sentence_a_embedding, sentence_a_lengths + 1, user_embedding,
            self._num_latent_class, self._latent_feature_dim,
            self._user_utterance_rnn_config, _reset_mode(mode),
            last_hidden=True, reuse=False
        )

        (
            y_b_logits, y_b_probs, sentence_b_features
        ) = build_sequence_encoder_network(
            sentence_b_embedding, sentence_b_lengths + 1, user_embedding,
            self._num_latent_class, self._latent_feature_dim,
            self._user_utterance_rnn_config, _reset_mode(mode),
            last_hidden=True, reuse=True
        )

        # Merges user a and b features.
        y_probs = tf.concat([y_a_probs, y_b_probs], axis=0)
        y_logits = tf.concat([y_a_logits, y_b_logits], axis=0)
        sentence_features = tf.concat([sentence_a_features,
                                       sentence_b_features], axis=0)

        # Contextualizes all user utterance mode groups.
        # Expands the 1st dim.
        user_state_embedding = tf.expand_dims(
            tf.matmul(y_probs, user_embedding), axis=0)
        user_states, _ = build_unidirectional_sequence_representation(
            user_state_embedding, [batch_size], self._user_state_rnn_config,
            _reset_mode(mode)
        )
        user_state_dim = self._user_state_rnn_config.hidden_dim
        flatten_user_states = tf.reshape(
            user_states, shape=[batch_size, user_state_dim])

        if not self._adapt_user_model:
            print("The user model is fixed!", file=sys.stderr)
            sentence_features = tf.stop_gradient(sentence_features)
            flatten_user_states = tf.stop_gradient(flatten_user_states)
            y_probs = tf.stop_gradient(y_probs)
        else:
            print("The user model is fine-tuned!", file=sys.stderr)

        conv_hidden_dim = self._conversation_context_rnn_config.hidden_dim
        with tf.variable_scope('predictor_network'):
            if mode == 'TRAIN' and self._keep_prob < 1.0:
                sentence_features = tf.nn.dropout(
                    sentence_features, keep_prob=self._keep_prob
                )
                flatten_user_states = tf.nn.dropout(
                    flatten_user_states, keep_prob=self._keep_prob
                )

            with tf.variable_scope('step_feature'):
                step_features = fusion_network(
                    sentence_features, flatten_user_states, conv_hidden_dim,
                    comp_method='highway', activation_fn=tf.nn.relu,
                    layer_norm=True, element_wise=False,
                    scope='rnn_state_highway'
                )

                # Re-orders features based on the turn indices.
                reorder_step_features = tf.expand_dims(tf.nn.embedding_lookup(
                    step_features, map_ab_indices_to_turn_indices
                ), axis=0)

            with tf.variable_scope('dialog_act_logits'):
                dialog_act_tag_embedding = build_embeddings(
                    self._num_dialog_act, self._dialog_act_embed_dim,
                    'dialog_act_tag'
                )

                dialog_act_logits, _ = recurrent_attention_predictor(
                    reorder_step_features, None, batch_size,
                    dialog_act_tag_embedding, self._num_dialog_act,
                    self._dialog_act_embed_dim,
                    self._conversation_context_rnn_config, mode)

                dialog_act_logits = tf.reshape(
                    dialog_act_logits, [batch_size, self._num_dialog_act]
                )

        # Builds inputs.
        inputs = {
            'sentence_a_indices': sentence_a_indices,
            'sentence_a_lengths': sentence_a_lengths,
            'sentence_b_indices': sentence_b_indices,
            'sentence_b_lengths': sentence_b_lengths,
            'map_ab_indices_to_turn_indices': map_ab_indices_to_turn_indices,
            'max_sentence_length': max_sentence_length,
            'batch_size': batch_size
        }

        # Builds outputs.
        outputs = {
            'y_logits': y_logits,
            'y_probs': y_probs,
            'dialog_act_logits': dialog_act_logits
        }

        return inputs, outputs

    def _build_feed_dict(self, input_tuple_list):
        """Builds feed dict for inputs."""
        feed_dict_list = []
        inputs_dict = dict(input_tuple_list)

        for input_name, input_var in self._inputs.items():
            if input_name not in inputs_dict:
                raise ValueError('Unknown input_name: {0}'.format(input_name))
            feed_dict_list.append((input_var, inputs_dict[input_name]))
        return dict(feed_dict_list)

    def build_loss(self):
        """Builds loss variables."""
        # A [batch_size] sized vector.
        dialog_act_indices = tf.placeholder(
            tf.int32, name='dialog_act_indices', shape=[None]
        )

        dialog_act_logits = self._outputs['dialog_act_logits']

        loss_to_opt = tf.reduce_sum(
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=dialog_act_indices,
                logits=dialog_act_logits
            )
        )

        self._inputs['dialog_act_indices'] = dialog_act_indices
        self._outputs['loss_to_opt'] = loss_to_opt

    def build_opt_op(self, opt_config):
        """Builds optimization operator for the model."""
        # Input variable needed for optimization during training phase.
        learning_rate = tf.placeholder(
            tf.float32, name='learning_rate', shape=())

        self._inputs['learning_rate'] = learning_rate

        # Builds the specified optimizer for training.
        optimizer = build_optimizer(opt_config, learning_rate)

        loss_to_opt = self._outputs['loss_to_opt']

        return compute_and_apply_gradient(loss_to_opt, optimizer,
                                          clip_value=opt_config.clip_value)

    def _run_model(self, session, feed_dict, opt_op=None):
        """Performans a forward and backward pass of the model."""
        if opt_op is None:
            opt_op = self._no_op


        fetches = [self._outputs[var_name]
                   for var_name in self._train_eval_fetch_var_names_]
        fetches.append(opt_op)

        all_outputs = session.run(fetches, feed_dict)

        fetched_var_dict = dict([
            (var_name, all_outputs[idx])
            for idx, var_name in enumerate(self._train_eval_fetch_var_names_)
        ])

        return fetched_var_dict

    def train_or_eval_model(self, session, input_tuple_list, opt_op):
        """Trains or evaluates the model with one batch of data."""
        feed_dict = self._build_feed_dict(input_tuple_list)
        fetched_dict = self._run_model(session, feed_dict, opt_op)

        return fetched_dict

    def infer_model(self, session, input_tuple_list):
        """Inference with the model."""
        feed_dict = self._build_feed_dict(input_tuple_list)
        logits = session.run(self._outputs['y_logits'], feed_dict)

        return logits

    def variables_to_save_or_restore(self):
        """Returns variables to be saved or restored."""
        variables_to_restore = tf.trainable_variables()

        def _lookup_var(var_name):
            """Looks up variable by its name."""
            for variable in tf.global_variables():
                # This might need to be modified to consistent with the TF
                # variable naming convention.
                # Currently, the naming convention is:
                # {scope}/var_name:{numbering}, where scope is sep by '/'.
                basename = variable.name.split('/')[-1].split(':')[0]
                if var_name == basename:
                    return variable
            return None

        # Embedding variables might not be trainable.
        embed_name_list = ['word']
        for embed_name in embed_name_list:
            embed_var = _lookup_var('{0}_embedding'.format(embed_name))
            if embed_var and embed_var not in variables_to_restore:
                print(
                    'Adds {0} to variable list to save or restore'.format(
                        embed_var.name), file=sys.stderr)
                variables_to_restore.append(embed_var)

        return variables_to_restore

    def variables_to_restore_from_pretrain(self, global_scope_name='model'):
        """Returns variables to be restored from pretrained model."""
        variables_to_restore = self.variables_to_save_or_restore()

        not_pretrain_scope_list = ['predictor_network']
        pretrain_variable_to_store = []
        for variable in variables_to_restore:
            scope_name_list = variable.name.split('/')

            if len(scope_name_list) < 2:
                print('Skips variable {0}'.format(variable.name))
                continue

            scope_name = scope_name_list[1]
            if scope_name in not_pretrain_scope_list:
                print('Skips variable {0}'.format(variable.name))
                continue

            pretrain_variable_to_store.append(variable)

        return pretrain_variable_to_store
