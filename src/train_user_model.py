#!/usr/bin/env python
"""Train wrapper for user model."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import argparse
import cPickle as pickle
import StringIO
import ConfigParser

from .model.user_model_helper import train_model
from .model.configure import UserModelConfiguration


def main():
    """main function"""

    defaults = {}

    # ===============
    # Parse config file.
    # ===============
    conf_parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        add_help=False
    )
    conf_parser.add_argument('-c', '--config',
                             help="config file")
    args, remaining_argv = conf_parser.parse_known_args()
    if args.config:
        # NOTE: SafeConfigParser deals with INI format config files.
        # It is more flexible than BOOST parse_config_file.
        # However, we need to add a default dummy section in order
        # to conver config to INI format.
        config_fp = StringIO.StringIO()
        config_fp.write('[dummy]\n')
        config_fp.write(open(args.config).read())
        config_fp.seek(0, os.SEEK_SET)

        config_parser = ConfigParser.SafeConfigParser()
        config_parser.readfp(config_fp)
        defaults.update(dict(config_parser.items('dummy')))

    # ===============
    # Parses commandline options and overwrites the config file options.
    # NOTE: We cannot set default values in ArgumentParser.
    # Otherwise, they will be overwritten if set by config file.
    # ===============
    cmdline_parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        parents=[conf_parser])
    cmdline_parser.add_argument('--model_outdir',
                                help='output directory for saved model(s)')
    cmdline_parser.add_argument('--batch_size',
                                type=int,
                                default=1,
                                help='maximum acceptable number of instances in'
                                ' a batch'
                                ' (used to allocate enough memory'
                                ' for the model)')

    # ==============================
    # Text representation parameters.
    cmdline_parser.add_argument('--train_word_embed',
                                type=int,
                                default=1,
                                help='train word embedding.')
    cmdline_parser.add_argument('--max_sentence_length',
                                type=int,
                                default=0,
                                help='max sentence length')

    cmdline_parser.add_argument('--word_vocab',
                                help='word vocab filename.')

    cmdline_parser.add_argument('--word_vocab_size',
                                type=int,
                                default=0,
                                help='word vocab size.')
    cmdline_parser.add_argument('--word_embed_dim',
                                type=int,
                                default=0,
                                help='word embedding size.')

    cmdline_parser.add_argument('--num_latent_class',
                                type=int,
                                default=0,
                                help='number of latent classes.')
    cmdline_parser.add_argument('--latent_feature_dim',
                                type=int,
                                default=0,
                                help='Latent feature dimension.')

    # ==============================
    # RNN related parameters.

    cmdline_parser.add_argument('--hidden_dim',
                                type=int,
                                default=0,
                                help='LSTM hidden layer size')
    cmdline_parser.add_argument('--num_hidden',
                                type=int,
                                default=0,
                                help='LSTM num hidden layers')
    cmdline_parser.add_argument('--residual_rnn',
                                type=int,
                                default=1,
                                help='use residual RNN')
    cmdline_parser.add_argument('--residual_method',
                                default='highway',
                                help='use highway or residual RNN'
                                '{highway, add}')

    cmdline_parser.add_argument('--keep_prob',
                                type=float,
                                default=1.0,
                                help='dropout keep probability')

    cmdline_parser.add_argument('--state_keep_prob',
                                default=None,
                                help='dropout keep probability for RNN state')
    cmdline_parser.add_argument('--input_keep_prob',
                                default=None,
                                help='dropout keep probability for RNN input')
    cmdline_parser.add_argument('--output_keep_prob',
                                default=None,
                                help='dropout keep probability for RNN output')
    cmdline_parser.add_argument('--forget_bias',
                                type=float,
                                default=1.0,
                                help='forget bias initial for RNN')
    cmdline_parser.add_argument('--carry_bias_init',
                                type=float,
                                default=1.0,
                                help='carry gate bias initial for RNN highway')
    cmdline_parser.add_argument('--couple_carry_transform_gates',
                                type=int,
                                default=1,
                                help='couple carry gate for RNN highway')
    cmdline_parser.add_argument('--couple_gate_lstm',
                                type=int,
                                default=0,
                                help='couple gates for RNN')
    cmdline_parser.add_argument('--vr_recurrent',
                                type=int,
                                default=1,
                                help='use varational RNN')

    # ===========================================================
    def rnn_param_config(cmd_parser, prefix):
        """Sets up RNN related parameter parser."""
        cmd_parser.add_argument('--{0}_hidden_dim'.format(prefix),
                                default=None,
                                help='LSTM hidden layer size')
        cmd_parser.add_argument('--{0}_num_hidden'.format(prefix),
                                default=None,
                                help='LSTM num hidden layers')
        cmd_parser.add_argument('--{0}_residual_rnn'.format(prefix),
                                default=None,
                                help='use residual RNN')
        cmd_parser.add_argument('--{0}_residual_method'.format(prefix),
                                default=None,
                                help='use highway or residual RNN'
                                '{highway, add}')

        cmd_parser.add_argument('--{0}_keep_prob'.format(prefix),
                                default=None,
                                help='dropout keep probability')

        cmd_parser.add_argument('--{0}_state_keep_prob'.format(prefix),
                                default=None,
                                help='dropout keep probability for RNN state')
        cmd_parser.add_argument('--{0}_input_keep_prob'.format(prefix),
                                default=None,
                                help='dropout keep probability for RNN input')
        cmd_parser.add_argument('--{0}_output_keep_prob'.format(prefix),
                                default=None,
                                help='dropout keep probability for RNN output')

        cmd_parser.add_argument('--{0}_forget_bias'.format(prefix),
                                default=None,
                                help='forget bias initial for RNN')

        cmd_parser.add_argument('--{0}_carry_bias_init'.format(prefix),
                                default=None,
                                help='carry gate bias initial for RNN highway')

        cmd_parser.add_argument(
            '--{0}_couple_carry_transform_gates'.format(prefix),
            default=None, help='couple carry gate for RNN highway'
        )

        cmd_parser.add_argument('--{0}_couple_gate_lstm'.format(prefix),
                                default=None,
                                help='couple gates for RNN')

        cmd_parser.add_argument('--{0}_vr_recurrent'.format(prefix),
                                default=None,
                                help='use varational RNN')

    rnn_param_prefix_set = ['user_state', 'user_utterance_encoder',
                            'user_utterance_decoder']

    for rnn_param_prefix in rnn_param_prefix_set:
        rnn_param_config(cmdline_parser, rnn_param_prefix)
    # ============================================================

    cmdline_parser.add_argument('--label_smoothing',
                                type=float,
                                default=0,
                                help='label smoothing value')

    cmdline_parser.add_argument('--word_token_keep_prob',
                                type=float,
                                default=1.0,
                                help='dropout keep probability')

    cmdline_parser.add_argument('--num_cpus',
                                type=int,
                                default=1,
                                help='number of CPU devices (threads)')
    # This argument is not used.
    cmdline_parser.add_argument('--num_gpus',
                                type=int,
                                default=0,
                                help='number of GPU devices (threads)')
    cmdline_parser.add_argument('--model_indir',
                                help='name of the model directory'
                                ' for training restart mode')

    cmdline_parser.add_argument('--train_data_filename',
                                help='training data filename')
    cmdline_parser.add_argument('--validation_data_filename',
                                help='validation data file')

    cmdline_parser.add_argument('--shuffle_instances',
                                default=False,
                                action='store_true',
                                help='shuffle instances within each training data file')
    cmdline_parser.add_argument('--init_learning_rate',
                                type=float,
                                default=0.0,
                                help='initial learning rate')
    cmdline_parser.add_argument('--lr_decay_rate',
                                type=float,
                                default=0.75,
                                help='learning rate decay rate')

    cmdline_parser.add_argument('--num_noneval_iterations',
                                type=int,
                                default=0,
                                help='number of iterations without'
                                ' evaluation on validation data'
                                ' (+x: skip evaluation of first x iterations.'
                                ' -x: skip evaluation of both first x'
                                ' iterations and iterations after learning'
                                ' rate halves; in this case, the training'
                                ' terminates based on max_iterations.)')
    cmdline_parser.add_argument('--max_iterations',
                                type=int,
                                default=0,
                                help='maximum number of iterations'
                                ' (0: do not check max_iterations.'
                                ' +x: run at most x iterations.'
                                ' -x: only valid when num_noneval_iterations < 0;'
                                ' run x iterations after learning rate halves.)')

    # More control over optimization parameters.
    cmdline_parser.add_argument('--opt_method',
                                default='Adam',
                                help='SGD, Adagrad, Adadelta, Adam.')
    cmdline_parser.add_argument('--clip_value',
                                type=float,
                                default=0.0,
                                help='Clip value for gradient norm.')
    cmdline_parser.add_argument('--adam_beta1',
                                type=float,
                                default=0.9,
                                help='Parameter beta1 for Adam.')
    cmdline_parser.add_argument('--adam_beta2',
                                type=float,
                                default=0.97,
                                help='Parameter beta2 for Adam.')
    cmdline_parser.add_argument('--adam_epsilon',
                                type=float,
                                default=1e-6,
                                help='Parameter epsilon for Adam.')

    cmdline_parser.add_argument('--initializer',
                                default='Xavier',
                                help='{Uniform, Gaussian, Xavier}')
    cmdline_parser.add_argument('--stddev',
                                type=float,
                                default=0.1,
                                help='standard deviation of the initializer')
    cmdline_parser.add_argument('--scale',
                                type=float,
                                default=0.1,
                                help='scale range for the initializer')
    cmdline_parser.add_argument('--validate_metric',
                                default='nll',
                                help='validate metric')

    cmdline_parser.add_argument('--verbose',
                                type=int,
                                default=1,
                                help='print verbose level')

    cmdline_parser.add_argument('--device',
                                default='cpu',
                                help='device for training {cpu, gpu}')

    cmdline_parser.add_argument('--word_embed',
                                default=None,
                                help='pretrain word embedding')

    cmdline_parser.add_argument('--rand_seed',
                                type=int,
                                default=0,
                                help='random seed')

    cmdline_parser.set_defaults(**defaults)
    args = cmdline_parser.parse_args(remaining_argv)
    print(16 * '=')
    print('training config:')
    for arg in vars(args):
        print('{0}: {1}'.format(arg, getattr(args, arg)))
    print(16 * '=')

    word_embed = None
    if args.word_embed is not None:
        with open(args.word_embed) as fin:
            print('Loading word embedding from {0}'.format(args.word_embed))
            word_embed = pickle.load(fin)
    else:
        print('No pretrained word embedding!')

    embedding_to_load = [('word_embedding', word_embed)]

    # Builds the training configuration.
    args_dict = vars(args)
    train_config = UserModelConfiguration()
    train_config.parse_from_dict(args_dict)

    train_config.save_model_dir = args.model_outdir

    vocab_list = ['word_vocab']

    vocab_name_to_file = [
        (vocab_name, args_dict.get(vocab_name))
        for vocab_name in vocab_list
    ]

    # Saves the model config for evaluation.
    model_config_filename = os.path.join(
        train_config.save_model_dir, 'model.config')
    assert train_config.model_config.dump_to_file(model_config_filename)

    train_model(
        args.train_data_filename, train_config, dict(vocab_name_to_file),
        valid_filename=args.validation_data_filename, rand_seed=args.rand_seed,
        shuffle_data=bool(args.shuffle_instances),
        max_sentence_length=args.max_sentence_length,
        embedding_to_load=dict(embedding_to_load)
    )

    exit(0)


if __name__ == '__main__':
    main()
