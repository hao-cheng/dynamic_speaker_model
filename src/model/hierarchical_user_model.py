#!/usr/bin/env python2
"""Hierarchical user model."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys

import tensorflow as tf

from .nn_helper import build_embeddings
from .configure import RNNConfig
from .model_helper import build_optimizer
from .model_helper import compute_and_apply_gradient
from .sequence_model_helper import build_sequence_encoder_network
from .sequence_model_helper import build_sequence_generation_network
from .sequence_model_helper import build_unidirectional_sequence_representation
from .sequence_model_helper import compute_sequence_xentropy_loss


class HierachicalUserModel(object):
    """A hierachical user model."""

    # Variable name list for fetching.
    _train_eval_fetch_var_names_ = (
        'y_logits', 'y_probs', 'loss_to_opt', 'conditional_entropy_y',
    )

    def __init__(self, config, mode, embedding_to_load=None):
        """Initialization."""
        # Configs the model.
        self._config_model(config)

        # Constructs the model.
        if mode == 'INFER':
            self._inputs, self._outputs = self._construct_infer_model(mode)
        else:
            self._inputs, self._outputs = self._construct_model(
                embedding_to_load, mode)

        # Null operation.
        self._no_op = tf.no_op()

    def __str__(self):
        """Function for printing the info of the model."""
        skip_vars = set(['_inputs', '_outputs', '_no_op', '_rnn_config'])

        model_config_str = '\n'.join([
            'model.{0}={1}'.format(attr, getattr(self, attr))
            for attr in self.__dict__ if attr not in skip_vars
        ])

        user_state_rnn_config_str = '\n'.join([
            'model.user_state_rnn_config.{0}={1}'.format(
                attr, getattr(self._user_state_rnn_config, attr))
            for attr in self._user_state_rnn_config.__slots__
        ])

        user_utterance_encoder_rnn_config_str = '\n'.join([
            'model.user_utterance_encoder_rnn_config.{0}={1}'.format(
                attr, getattr(self._user_utterance_encoder_rnn_config, attr))
            for attr in self._user_utterance_encoder_rnn_config.__slots__
        ])

        user_utterance_decoder_rnn_config_str = '\n'.join([
            'model.user_utterance_decoder_rnn_config.{0}={1}'.format(
                attr, getattr(self._user_utterance_decoder_rnn_config, attr))
            for attr in self._user_utterance_decoder_rnn_config.__slots__
        ])

        return '\n'.join([
            'Model configuraton:', model_config_str, user_state_rnn_config_str,
            user_utterance_encoder_rnn_config_str,
            user_utterance_decoder_rnn_config_str,
        ])

    def _config_model(self, config):
        """Constructs the model parameters from the ModelConfig object."""
        self._word_vocab_size = config.word_vocab_size
        self._word_embed_dim = config.word_embed_dim
        self._train_word_embed = config.train_word_embed

        self._keep_prob = config.keep_prob

        self._num_latent_class = config.num_latent_class
        self._latent_feature_dim = config.latent_feature_dim

        self._user_state_rnn_config = RNNConfig()
        self._user_state_rnn_config.parse_from_json_string(
            config.user_state_rnn_config.to_json_string()
        )

        self._user_utterance_encoder_rnn_config = RNNConfig()
        self._user_utterance_encoder_rnn_config.parse_from_json_string(
            config.user_utterance_encoder_rnn_config.to_json_string()
        )
        self._user_utterance_decoder_rnn_config = RNNConfig()
        self._user_utterance_decoder_rnn_config.parse_from_json_string(
            config.user_utterance_decoder_rnn_config.to_json_string()
        )


    def _construct_infer_model(self, mode='INFER'):
        """Constructs the model based on the config."""
        # Note: for now, the batch_size is equal to the number of turns.
        # A [batch_size, sentence_length] sized matrix containing word indices
        # for each sentence.
        sentence_indices = tf.placeholder(
            tf.int32, name='sentence_indices', shape=[None, None])

        # A [batch_size] sized vector containing the length of each sentence.
        sentence_lengths = tf.placeholder(
            tf.int32, name='sentence_lengths', shape=[None])

        # Scalar for the maximum sentence length in the current batch.
        max_sentence_length = tf.placeholder(
            tf.int32, name='max_sentence_length', shape=()
        )

        # Scalar for the current batch size.
        batch_size = tf.placeholder(tf.int32, name='batch_size', shape=())

        with tf.device('/cpu:0'), tf.variable_scope('input_embeddings'):
            word_embedding = build_embeddings(
                self._word_vocab_size, self._word_embed_dim, 'word',
                pretrain_embed=None,
                trainable=self._train_word_embed
            )

            user_embedding = build_embeddings(
                self._num_latent_class, self._latent_feature_dim, 'user',
                trainable=True
            )

        sentence_embedding = tf.nn.embedding_lookup(
            word_embedding, sentence_indices
        )

        # Encodes the user utterance information.
        (y_logits, y_probs, _) = build_sequence_encoder_network(
            sentence_embedding, sentence_lengths + 1, user_embedding,
            self._num_latent_class, self._latent_feature_dim,
            self._user_utterance_encoder_rnn_config, mode, last_hidden=True
        )

        # Contextualizes all user utterance mode groups.
        # Expands the 1st dim.
        user_state_embedding = tf.expand_dims(
            tf.matmul(y_probs, user_embedding), axis=0)
        user_states, _ = build_unidirectional_sequence_representation(
            user_state_embedding, None, self._user_state_rnn_config,
            mode
        )
        user_state_dim = self._user_state_rnn_config.hidden_dim
        flatten_user_states = tf.squeeze(user_states, axis=0)

        # Builds inputs.
        inputs = {
            'sentence_indices': sentence_indices,
            'sentence_lengths': sentence_lengths,
            'max_sentence_length': max_sentence_length,
            'batch_size': batch_size
        }

        # Builds outputs.
        outputs = {
            'y_logits': y_logits,
            'y_probs': y_probs,
            'local_user_mode_vectors': tf.squeeze(user_state_embedding, axis=0),
            'user_state_vectors': flatten_user_states
        }

        return inputs, outputs


    def _construct_model(self, embedding_to_load, mode):
        """Constructs the model based on the config."""
        # Note: for now, the batch_size is equal to the number of turns.
        # A [batch_size, sentence_length] sized matrix containing word indices
        # for each sentence.
        sentence_indices = tf.placeholder(
            tf.int32, name='sentence_indices', shape=[None, None])

        # A [batch_size] sized vector containing the length of each sentence.
        sentence_lengths = tf.placeholder(
            tf.int32, name='sentence_lengths', shape=[None])

        # Scalar for the maximum sentence length in the current batch.
        max_sentence_length = tf.placeholder(
            tf.int32, name='max_sentence_length', shape=()
        )

        # Scalar for the current batch size.
        batch_size = tf.placeholder(tf.int32, name='batch_size', shape=())

        # TODO(chenghao): This should be put into a function.
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

        sentence_embedding = tf.nn.embedding_lookup(
            word_embedding, sentence_indices
        )

        # Encodes the user utterance information.
        (y_logits, y_probs, _) = build_sequence_encoder_network(
            sentence_embedding, sentence_lengths + 1, user_embedding,
            self._num_latent_class, self._latent_feature_dim,
            self._user_utterance_encoder_rnn_config, mode, last_hidden=True
        )

        # Contextualizes all user utterance mode groups.
        # Expands the 1st dim.
        user_state_embedding = tf.expand_dims(
            tf.matmul(y_probs, user_embedding), axis=0)
        user_states, _ = build_unidirectional_sequence_representation(
            user_state_embedding, None, self._user_state_rnn_config, mode
        )
        user_state_dim = self._user_state_rnn_config.hidden_dim
        flatten_user_states = tf.squeeze(user_states, axis=0)

        # Applies the user state feature for generation.
        sentence_sequence_logits = build_sequence_generation_network(
            sentence_embedding, flatten_user_states, user_state_dim,
            sentence_lengths + 1, max_sentence_length + 1,
            self._word_vocab_size, self._user_utterance_decoder_rnn_config, mode
        )

        # Builds inputs.
        inputs = {
            'sentence_indices': sentence_indices,
            'sentence_lengths': sentence_lengths,
            'max_sentence_length': max_sentence_length,
            'batch_size': batch_size
        }

        # Builds outputs.
        outputs = {
            'y_logits': y_logits,
            'y_probs': y_probs,
            'sentence_sequence_logits': sentence_sequence_logits
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
        target_sentence_indices = tf.placeholder(
            tf.int32, name='target_sentence_indices', shape=[None, None])

        target_weights = tf.sequence_mask(self._inputs['sentence_lengths'] + 1,
                                          dtype=tf.float32)

        sentence_logits = self._outputs['sentence_sequence_logits']

        # Entropy loss of latent class.
        cond_entropy_y = -tf.nn.softmax_cross_entropy_with_logits_v2(
            labels=self._outputs['y_probs'],
            logits=self._outputs['y_logits']
        ) - tf.log(1.0 / self._num_latent_class) * tf.reduce_sum(
            self._outputs['y_probs'], axis=1
        )

        # Computes data log probs.
        neg_seq_data_log_prob = tf.reduce_sum(
            compute_sequence_xentropy_loss(
                target_sentence_indices, sentence_logits, target_weights
            ), axis=1
        )

        loss_to_opt = neg_seq_data_log_prob

        self._inputs['target_sentence_indices'] = target_sentence_indices
        self._outputs['loss_to_opt'] = tf.reduce_sum(loss_to_opt)
        self._outputs['conditional_entropy_y'] = cond_entropy_y

    def build_opt_op(self, opt_config):
        """Builds optimization operator for the model."""
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
        fetch_var_name_list = ['y_probs', 'local_user_mode_vectors',
                               'user_state_vectors']
        fetch_list = [self._outputs[var_name]
                      for var_name in fetch_var_name_list]
        all_outputs = session.run(fetch_list, feed_dict)

        fetched_var_dict = dict([
            (var_name, all_outputs[idx])
            for idx, var_name in enumerate(fetch_var_name_list)
        ])

        return fetched_var_dict

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
