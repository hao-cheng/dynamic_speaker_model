#!/usr/bin/env python2
"""Helper function for train, eval and infer with the user model."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import time
from collections import Counter

import numpy as np
import tensorflow as tf

from .hierarchical_user_model import HierachicalUserModel
from .py_data_lib.data_interface import DataContainer


_PRINT_COUNT = 100
_PRINT_OUT = sys.stderr
_DECORATION = 16 * '='


def _build_initializer(initializer):
    """Builds initialization method for the TF model."""
    if initializer == 'Uniform':
        print('Using random_uniform_initializer', file=_PRINT_OUT)
        tf_initializer = tf.random_uniform_initializer(
            -0.1, 0.1, dtype=tf.float32
        )
    elif initializer == 'Gaussian':
        print('Using truncated_normal_initializer', file=_PRINT_OUT)
        tf_initializer = tf.truncated_normal_initializer(
            stddev=0.1, dtype=tf.float32
        )
    elif initializer == 'Xavier':
        print('Using xavier_initializer', file=_PRINT_OUT)
        tf_initializer = tf.contrib.layers.variance_scaling_initializer(
            factor=2.0, mode='FAN_IN', uniform=False, dtype=tf.float32
        )
    else:
        raise ValueError('Unknown initializer {0}!'.format(initializer))

    return tf_initializer


def run_epoch(model, session, data_container, learning_rate, opt_op, mode,
              verbose=True, auto_decay=False):
    """Runs one epoch over the data."""
    start_time = time.time()
    neg_log_like = 0.
    exp_kl_y = 0.
    iter_count = 0
    token_cnt = 0.0

    process_sample_cnt = 0
    prob_list = []
    prediction_list = []
    local_user_mode_vector_list = []
    user_state_vector_list = []

    for (input_tuple_list, _, num_sample) in data_container.extract_batches():
        if auto_decay and mode == 'TRAIN':
            model_size = 512.0
            factor = 2.5
            warmup = 4000.0
            step = session.run(tf.train.get_global_step()) + 1
            learning_rate = factor * model_size ** (-0.5) * min(
                step ** (-0.5), step * warmup ** (-1.5)
            )
            learning_rate = max(learning_rate, 1e-9)

        if mode == 'INFER':
            input_tuple_list.append(('batch_size', num_sample))
            input_dict = dict(input_tuple_list)
            fetched_dict = model.infer_model(session, input_tuple_list)
            local_user_mode_vector_list.append(
                fetched_dict['local_user_mode_vectors'])
            user_state_vector_list.append(
                fetched_dict['user_state_vectors'])
        else:
            if mode == 'TRAIN':
                input_tuple_list.append(('learning_rate', learning_rate))

            input_tuple_list.append(('batch_size', num_sample))

            input_dict = dict(input_tuple_list)
            fetched_dict = model.train_or_eval_model(
                session, input_tuple_list, opt_op
            )

            logits = fetched_dict['y_logits']
            assert logits.shape[0] == num_sample

            neg_log_like += fetched_dict['loss_to_opt']
            exp_kl_y += np.sum(fetched_dict['conditional_entropy_y'])

            token_cnt += np.sum(input_dict['sentence_lengths'])

            # Counts the sentence ends.
            token_cnt += num_sample

            prediction_list.append(np.argmax(logits, axis=1))

        prob_list.append(fetched_dict['y_probs'])

        iter_count += 1

        process_sample_cnt += num_sample

        if verbose and (iter_count) % _PRINT_COUNT == 0:
            print(
                'iter {:d}:, {:.3f} examples per second'.format(
                    iter_count,
                    process_sample_cnt / (time.time() - start_time)
                ),
                'loss {:.3f}'.format(fetched_dict['loss_to_opt']),
                file=_PRINT_OUT
            )

    if auto_decay and mode == 'TRAIN':
        print('Dynamic learning rate deacy', file=_PRINT_OUT)
        print('After one epoch, learning rate: {:.5f}\n'.format(learning_rate),
              file=_PRINT_OUT)

    print(
        'time for one epoch: {:.3f} secs'.format(time.time() - start_time),
        file=_PRINT_OUT
    )
    print('iters over {0} num of samples'.format(process_sample_cnt),
          file=_PRINT_OUT)
    print('iters over {0} num of tokens, including <END>'.format(
        token_cnt), file=_PRINT_OUT)

    eval_metric = {}
    output_dict = {
        'latent_class_probs': prob_list
    }

    if mode != 'INFER':
        preds = np.concatenate(prediction_list, axis=0)
        print('preds shape: ', preds.shape, file=sys.stderr)
        print('uniq pred_labels dis\n', np.unique(preds, return_counts=True),
              file=sys.stderr)

        eval_metric = {
            'nll': neg_log_like,
            'avg_nll': neg_log_like / process_sample_cnt,
            'log_ppl': neg_log_like / token_cnt,
            'cond_entropy_y': -exp_kl_y / process_sample_cnt,
        }
    else:
        output_dict['local_user_mode_vectors'] = local_user_mode_vector_list
        output_dict['user_state_vectors'] = user_state_vector_list

    return neg_log_like, eval_metric, output_dict


def train_model(train_filename, train_config, vocab_file_dict,
                valid_filename=None, shuffle_data=False, rand_seed=0,
                embedding_to_load=None, max_sentence_length=20):
    """ Training wrapper function."""
    if train_config.device == 'cpu':
        session_config = tf.ConfigProto(
            device_count={'GPU': 0},
            intra_op_parallelism_threads=train_config.num_cpus,
            inter_op_parallelism_threads=train_config.num_cpus,
            allow_soft_placement=True
        )
    else:
        session_config = tf.ConfigProto(
            intra_op_parallelism_threads=train_config.num_cpus,
            inter_op_parallelism_threads=train_config.num_cpus,
            allow_soft_placement=True
        )
        session_config.gpu_options.allow_growth = True

    # The data would sit in memory for the whole training process.
    print('Loads train: {0}'.format(train_filename), file=sys.stderr)
    word_vocab_filename = vocab_file_dict['word_vocab']
    train_data_container = DataContainer(
        train_filename, word_vocab_filename, 1,
        max_sentence_length, shuffle_data=shuffle_data
    )

    valid_data_container = None
    if valid_filename:
        print('Loads valid: {0}'.format(valid_filename), file=sys.stderr)
        valid_data_container = DataContainer(
            valid_filename, word_vocab_filename, 1,
            max_sentence_length, shuffle_data=False
        )

    if not os.path.exists(train_config.save_model_dir):
        raise ValueError(
            'save_model_dir ({0}) does not exist!'.format(
                train_config.save_model_dir)
        )

    with tf.Graph().as_default(), tf.Session(config=session_config) as session:
        tf.set_random_seed(rand_seed)
        np.random.seed(rand_seed)
        initializer = _build_initializer(train_config.initializer)

        with tf.variable_scope('model', reuse=None, initializer=initializer):
            model = HierachicalUserModel(
                train_config.model_config, 'TRAIN',
                embedding_to_load=embedding_to_load
            )

        # Prints out the model configuration.
        print(_DECORATION, file=_PRINT_OUT)
        print(model, file=_PRINT_OUT)
        print(_DECORATION, file=_PRINT_OUT)
        model_saver = tf.train.Saver(model.variables_to_save_or_restore())

        # This is needed for both TRAIN and EVAL.
        model.build_loss()

        with tf.variable_scope('model', reuse=True):
            valid_model = HierachicalUserModel(
                train_config.model_config, 'TRAIN_EVAL')
            valid_model.build_loss()

        # This operation is only needed for TRAIN phase.
        opt_op = model.build_opt_op(train_config.opt_config)

        session.run(tf.global_variables_initializer())
        tf.get_default_graph().finalize()

        start_decay_it = train_config.max_epoch_iter
        prev_valid_metric = np.finfo(np.float32).max

        for it in xrange(train_config.max_epoch_iter):
            print(_DECORATION, file=_PRINT_OUT)
            print('Train Iter {0}'.format(it), file=_PRINT_OUT)

            lr_decay = train_config.lr_decay_rate ** max(
                it - start_decay_it, 0.0
            )
            cur_lr = train_config.init_learning_rate * lr_decay

            print('Current learning rate: {:.5f}\n'.format(cur_lr),
                  file=_PRINT_OUT)

            _, train_metric, _ = run_epoch(
                model, session, train_data_container, cur_lr, opt_op, 'TRAIN',
                auto_decay=(not valid_data_container)
            )

            print('\n'.join([
                'train {}: {:.3f}'.format(metric_name, metric_val)
                for metric_name, metric_val in train_metric.items()
            ]), file=_PRINT_OUT)

            # Saves the current model.
            if valid_data_container is None or (
                    it < train_config.skip_valid_iter):
                model_saver.save(
                    session,
                    os.path.join(train_config.save_model_dir, 'model.ckpt'),
                    global_step=it
                )
                continue

            # Validates using the valid_data.
            print(_DECORATION, file=_PRINT_OUT)
            _, valid_metric, _ = run_epoch(
                valid_model, session, valid_data_container, 0.0, None, 'EVAL',
                verbose=False
            )

            print('\n'.join([
                'valid {}: {:.5f}'.format(metric_name, metric_val)
                for metric_name, metric_val in valid_metric.items()
            ]), file=_PRINT_OUT)

            if train_config.validate_metric == 'acc':
                cur_valid_metric = -valid_metric['acc']
            else:
                cur_valid_metric = valid_metric[train_config.validate_metric]

            if prev_valid_metric < cur_valid_metric:
                print(_DECORATION, file=_PRINT_OUT)
                print('Restores the previous model.', file=_PRINT_OUT)
                print(_DECORATION, file=_PRINT_OUT)
                ckpt = tf.train.get_checkpoint_state(
                    train_config.save_model_dir)
                model_saver.restore(session, ckpt.model_checkpoint_path)

                # Stops the training cuz an increase for the second time.
                if start_decay_it < train_config.max_epoch_iter:
                    break

                start_decay_it = it
            else:
                print(_DECORATION, file=_PRINT_OUT)
                print('Saves the current model.', file=_PRINT_OUT)
                print(_DECORATION, file=_PRINT_OUT)
                model_saver.save(
                    session,
                    os.path.join(train_config.save_model_dir, 'model.ckpt'),
                    global_step=it
                )
                prev_valid_metric = cur_valid_metric

        print('Training model done!', file=_PRINT_OUT)

        # Validates using the valid_data.
        if valid_data_container:
            _, valid_metric, _ = run_epoch(
                valid_model, session, valid_data_container, 0.0, None, 'EVAL',
                verbose=False
            )

            print('\n'.join([
                'valid {}: {:.5f}'.format(metric_name, metric_val)
                for metric_name, metric_val in valid_metric.items()
            ]), file=_PRINT_OUT)
            print(_DECORATION, file=_PRINT_OUT)

    return True


def eval_model(eval_filename, vocab_file_dict, model_config, save_model_dir,
               num_cpus=1, max_sentence_length=20, is_infer=False,
               eval_output_basename=None):
    """Evaluation wrapper function."""
    session_config = tf.ConfigProto(
        device_count={'GPU': 0},
        intra_op_parallelism_threads=num_cpus,
        inter_op_parallelism_threads=num_cpus,
        allow_soft_placement=True
    )

    eval_data_container = DataContainer(
        eval_filename, vocab_file_dict['word_vocab'], 1, max_sentence_length,
        shuffle_data=False
    )

    if not os.path.exists(save_model_dir):
        raise ValueError(
            'save_model_dir ({0}) does not exist!'.format(save_model_dir)
        )

    with tf.Graph().as_default(), tf.Session(config=session_config) as session:
        # Creates the model.
        with tf.variable_scope('model', reuse=None):
            model = HierachicalUserModel(
                model_config, 'INFER' if is_infer else 'EVAL'
            )

        # Prints out the model configuration.
        print(_DECORATION, file=_PRINT_OUT)
        print(model, file=_PRINT_OUT)
        print(_DECORATION, file=_PRINT_OUT)

        model_saver = tf.train.Saver(model.variables_to_save_or_restore())

        # This is needed for both TRAIN and EVAL.
        if not is_infer:
            model.build_loss()

        session.run(tf.global_variables_initializer())

        # Loads in the trained model.
        ckpt = tf.train.get_checkpoint_state(save_model_dir)
        if ckpt:
            print('Reading model from {0}'.format(ckpt.model_checkpoint_path),
                  file=_PRINT_OUT)
            model_saver.restore(session, ckpt.model_checkpoint_path)
        else:
            print('Can not read in the paremters from {0}'.format(
                ckpt.model_checkpoint_path), file=_PRINT_OUT)
            sys.exit(1)

        tf.get_default_graph().finalize()

        print(_DECORATION, file=_PRINT_OUT)
        print('Evaluation phase starts', file=_PRINT_OUT)

        _, eval_metric, eval_output = run_epoch(
            model, session, eval_data_container, 0, None,
            'INFER' if is_infer else 'EVAL', verbose=False
        )

        print('\n'.join([
            'eval {}: {:.5f}'.format(metric_name, metric_val)
            for metric_name, metric_val in eval_metric.items()
        ]), file=_PRINT_OUT)
        print('Evaluation phase done!', file=_PRINT_OUT)
        print(_DECORATION, file=_PRINT_OUT)

        # Outputs embeddings if is_infer is True.
        if is_infer and eval_output_basename:
            to_save_embed_name = ['latent_class_probs',
                                  'local_user_mode_vectors',
                                  'user_state_vectors']
            embed_dict = dict([
                (embed_name, eval_output[embed_name])
                for embed_name in to_save_embed_name
            ])
            eval_data_container.save_embed_to_file(
                embed_dict, eval_output_basename
            )

        return True
