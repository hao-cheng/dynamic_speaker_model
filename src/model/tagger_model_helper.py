#!/usr/bin/env python2
"""Helper function for train, eval and infer with the tagger model."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import time

import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score

from .hierarchical_predictor_model import HierarchicalPredictorModel
from .py_data_lib.data_interface import DialogActDataContainer


_PRINT_COUNT = 100
_PRINT_OUT = sys.stderr
_DECORATION = 16 * '='


def softmax(logits, axis=1):
    """Computes softmax along the given axis."""
    exps = np.exp(logits - np.max(logits, axis=axis, keepdims=True))
    return exps / np.sum(exps, axis=axis, keepdims=True)


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


def restore_model(model_dir, model_saver, session):
    """Restores the model from the given directory and model saver."""
    ckpt = tf.train.get_checkpoint_state(model_dir)
    if ckpt:
        model_saver.restore(session, ckpt.model_checkpoint_path)
        return

    raise ValueError('Can not load model from {0}'.format(model_dir))


def run_epoch(model, session, data_container, learning_rate, opt_op, mode,
              verbose=True, eval_func=None):
    """Runs one epoch over the data."""
    start_time = time.time()
    neg_log_like = 0.
    iter_count = 0
    token_cnt = 0.0

    process_sample_cnt = 0
    prob_list = []
    prediction_list = []
    gold_label_list = []

    num_predictions = 0

    for (input_tuple_list, dialog_act_indices, num_sample
        ) in data_container.extract_batches():
        if mode == 'INFER':
            logits = model.infer_model(session, input_tuple_list)
        else:
            if mode == 'TRAIN':
                # Adds learning_rate for training.
                input_tuple_list.append(('learning_rate', learning_rate))
            input_tuple_list.append(('batch_size', num_sample))
            input_tuple_list.append(('dialog_act_indices', dialog_act_indices))

            input_dict = dict(input_tuple_list)
            fetched_dict = model.train_or_eval_model(
                session, input_tuple_list, opt_op
            )
            total_loss = fetched_dict['loss_to_opt']
            logits = fetched_dict['dialog_act_logits']

            assert logits.shape[0] == len(dialog_act_indices)
            assert logits.shape[1] == model._num_dialog_act
            assert logits.shape[0] > 0

            num_predictions += logits.shape[0]
            neg_log_like += fetched_dict['loss_to_opt']

            token_cnt += np.sum(input_dict['sentence_a_lengths'])
            token_cnt += np.sum(input_dict['sentence_b_lengths'])

            # Count the sentence ends.
            token_cnt += num_sample

            gold_label_list.append(dialog_act_indices)

        prediction_list.append(np.argmax(logits, axis=1))
        prob_list.append(softmax(logits))

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

    print(
        'time for one epoch: {:.3f} secs'.format(time.time() - start_time),
        file=_PRINT_OUT
    )
    print('iters over {0} num of samples'.format(process_sample_cnt),
          file=_PRINT_OUT)
    print('iters over {0} num of tokens, including <END>'.format(
        token_cnt), file=_PRINT_OUT)

    eval_metric = {}
    preds = np.concatenate(prediction_list, axis=0)
    probs = np.concatenate(prob_list, axis=0)
    output_dict = {
        'preds': preds,
        'probs': probs
    }

    if mode != 'INFER':
        gold_labels = np.concatenate(gold_label_list, axis=0)
        print('gold_labels shape: ', gold_labels.shape, file=sys.stderr)
        print('preds shape: ', preds.shape, file=sys.stderr)
        print('uniq pred_labels dis\n',
               np.unique(preds, return_counts=True), file=sys.stderr)

        eval_metric = {
            'nll': neg_log_like,
            'accuracy': accuracy_score(gold_labels, preds)
        }
        if eval_func:
            eval_metric.update(
                eval_func(probs, gold_labels)
            )
        output_dict['gold_labels'] = gold_labels

    return neg_log_like, eval_metric, output_dict


def train_model(train_filename, train_config, vocab_file_dict,
                valid_filename=None, batch_size=1,
                shuffle_data=False, rand_seed=0, embedding_to_load=None,
                max_sentence_length=20):
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
    train_data_container = DialogActDataContainer(
        train_filename, vocab_file_dict['word_vocab'],
        vocab_file_dict['dialog_act_vocab'], 1, max_sentence_length,
        shuffle_data=shuffle_data
    )

    valid_data_container = None
    if valid_filename:
        print('Loads valid: {0}'.format(valid_filename), file=sys.stderr)
        valid_data_container = DialogActDataContainer(
            valid_filename, vocab_file_dict['word_vocab'],
            vocab_file_dict['dialog_act_vocab'], 1, max_sentence_length,
            shuffle_data=False
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
            model = HierarchicalPredictorModel(
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
            valid_model = HierarchicalPredictorModel(
                train_config.model_config, 'TRAIN_EVAL')
            valid_model.build_loss()

        # This operation is only needed for TRAIN phase.
        opt_op = model.build_opt_op(train_config.opt_config)

        session.run(tf.global_variables_initializer())

        # Loads pretrain model parameters if specified.
        if train_config.pretrain_model_dir:
            print('Loads pretrained model from {0}'.format(
                train_config.pretrain_model_dir), file=_PRINT_OUT)
            temp_saver = tf.train.Saver(
                model.variables_to_restore_from_pretrain())
            restore_model(train_config.pretrain_model_dir, temp_saver, session)
            del temp_saver

        tf.get_default_graph().finalize()

        start_decay_it = train_config.max_epoch_iter
        prev_valid_metric = np.finfo(np.float32).max

        # Logs iteration information.
        log_file_fp = open(os.path.join(
            train_config.save_model_dir, 'summary.tsv'), 'w')
        log_file_fp.write('iter\tmetric\tvalue\n')

        for it in xrange(train_config.max_epoch_iter):
            print(_DECORATION, file=_PRINT_OUT)
            print('Train Iter {0}'.format(it), file=_PRINT_OUT)

            lr_decay = train_config.lr_decay_rate ** max(it - start_decay_it, 0.0)
            cur_lr = train_config.init_learning_rate * lr_decay

            print('Current learning rate: {:.5f}\n'.format(cur_lr),
                  file=_PRINT_OUT)

            _, train_metric, _ = run_epoch(
                model, session, train_data_container, cur_lr, opt_op, 'TRAIN',
                eval_func=None
            )

            print('\n'.join([
                'train {}: {:.3f}'.format(metric_name, metric_val)
                for metric_name, metric_val in train_metric.items()
            ]), file=_PRINT_OUT)

            # Logs metrics.
            for metric_name, metric_val in train_metric.items():
                if metric_name == 'nll':
                    metric_val = -metric_val
                log_file_fp.write('{0}\ttrain__{1}\t{2}\n'.format(
                    it, metric_name, metric_val
                ))

            if valid_data_container is None or it < train_config.skip_valid_iter:
                # Saves the current model.
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
                verbose=False, eval_func=None
            )

            print('\n'.join([
                'valid {}: {:.5f}'.format(metric_name, metric_val)
                for metric_name, metric_val in valid_metric.items()
            ]), file=_PRINT_OUT)

            # Logs metrics.
            for metric_name, metric_val in valid_metric.items():
                if metric_name == 'nll':
                    metric_val = -metric_val
                log_file_fp.write('{0}\tvalidation__{1}\t{2}\n'.format(
                    it, metric_name, metric_val
                ))

            if train_config.validate_metric == 'acc':
                cur_valid_metric = -valid_metric['acc']
            else:
                cur_valid_metric = valid_metric[train_config.validate_metric]

            if prev_valid_metric < cur_valid_metric:
                print(_DECORATION, file=_PRINT_OUT)
                print('Restores the previous model.', file=_PRINT_OUT)
                print(_DECORATION, file=_PRINT_OUT)
                restore_model(train_config.save_model_dir, model_saver, session)

                if start_decay_it < train_config.max_epoch_iter:
                    # Stops the training cuz an increase for the second time.
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

        log_file_fp.close()

        print('Training model done!', file=_PRINT_OUT)

        # Validates using the valid_data.
        if valid_data_container:
            _, valid_metric, _ = run_epoch(
                valid_model, session, valid_data_container, 0.0, None, 'EVAL',
                verbose=False, eval_func=None
            )

            print('\n'.join([
                'valid {}: {:.5f}'.format(metric_name, metric_val)
                for metric_name, metric_val in valid_metric.items()
            ]), file=_PRINT_OUT)
            print(_DECORATION, file=_PRINT_OUT)

        with open(os.path.join(train_config.save_model_dir, 'objectives.tsv'),
                  'w') as fout:
            # Writes valid metrics.
            fout.write('metric\tvalue\n')
            for metric_name, metric_val in valid_metric.items():
                if metric_name == 'nll':
                    metric_val = -metric_val
                fout.write('{0}\t{1}\n'.format(metric_name, metric_val))

    return True


def eval_model(eval_filename, vocab_file_dict, model_config, save_model_dir,
               num_cpus=1, batch_size=1, max_sentence_length=20,
               eval_output_basename=None):
    """Evaluation wrapper function."""
    session_config = tf.ConfigProto(
        device_count={'GPU': 0},
        intra_op_parallelism_threads=num_cpus,
        inter_op_parallelism_threads=num_cpus,
        allow_soft_placement=True
    )

    eval_data_container = DialogActDataContainer(
        eval_filename, vocab_file_dict['word_vocab'],
        vocab_file_dict['dialog_act_vocab'], 1, max_sentence_length,
        shuffle_data=False
    )

    if not os.path.exists(save_model_dir):
        raise ValueError(
            'save_model_dir ({0}) does not exist!'.format(save_model_dir)
        )

    with tf.Graph().as_default(), tf.Session(config=session_config) as session:
        # Creates the model.
        with tf.variable_scope('model', reuse=None):
            model = HierarchicalPredictorModel(model_config, 'EVAL')

        # Prints out the model configuration.
        print(_DECORATION, file=_PRINT_OUT)
        print(model, file=_PRINT_OUT)
        print(_DECORATION, file=_PRINT_OUT)

        model_saver = tf.train.Saver(model.variables_to_save_or_restore())

        # This is needed for both TRAIN and EVAL.
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
                save_model_dir), file=_PRINT_OUT)
            sys.exit(1)

        tf.get_default_graph().finalize()

        print(_DECORATION, file=_PRINT_OUT)
        print('Evaluation phase starts', file=_PRINT_OUT)

        _, eval_metric, eval_output = run_epoch(
            model, session, eval_data_container, 0, None, 'EVAL', verbose=False,
            eval_func=None
        )

        with open(os.path.join(save_model_dir, 'eval_summary.tsv'), 'w'
                 ) as log_fp:
            log_fp.write('iter\tmetric\tvalue\n')
            for metric_name, metric_val in eval_metric.items():
                if metric_name == 'nll':
                    metric_val = -metric_val
                log_fp.write('{0}\t{1}\t{2}\n'.format(
                    0, metric_name, metric_val
                ))

        print('\n'.join([
            'eval {}: {:.5f}'.format(metric_name, metric_val)
            for metric_name, metric_val in eval_metric.items()
        ]), file=_PRINT_OUT)
        print('Evaluation phase done!', file=_PRINT_OUT)
        print(_DECORATION, file=_PRINT_OUT)

        if eval_output_basename:
            eval_output_filename = '{0}.prediction'.format(eval_output_basename)

            eval_data_container.write_prediction(
                eval_output['probs'], eval_output_filename
            )

        return True
