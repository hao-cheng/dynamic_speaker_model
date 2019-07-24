#!/usr/bin/env python2
"""Evaluate wrapper for tagger model."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import argparse
import StringIO
import ConfigParser

from .model.configure import TaggerModelConfig
from .model.tagger_model_helper import eval_model


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
    # Parse commandline options and overwrites the config file options.
    # NOTE: We cannot set default values in ArgumentParser.
    # Otherwise, they will be overwritten if set by config file.
    # ===============
    cmdline_parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        parents=[conf_parser])
    cmdline_parser.add_argument('--model_indir',
                                help='name of the model directory'
                                ' for testing mode or'
                                ' training restart mode')
    cmdline_parser.add_argument('--eval_data_filename',
                                help='data filename for eval')
    cmdline_parser.add_argument('--outbase',
                                help='output basename')
    cmdline_parser.add_argument('--word_vocab',
                                help='word vocab filename.')

    cmdline_parser.add_argument('--dialog_act_vocab',
                                help='dialog act vocab filename.')

    cmdline_parser.add_argument('--max_sentence_length',
                                type=int,
                                default=0,
                                help='max sentence length')

    cmdline_parser.add_argument('--num_cpus',
                                type=int,
                                default=0,
                                help='number of CPU devices (threads)')

    cmdline_parser.set_defaults(**defaults)
    args = cmdline_parser.parse_args(remaining_argv)

    if args.model_indir is None:
        cmdline_parser.print_usage()
        sys.stderr.write('error: argument --model_indir is required\n')
        exit(1)
    if args.eval_data_filename is None:
        cmdline_parser.print_usage()
        sys.stderr.write('error: argument --eval_data_filename is required\n')
        exit(1)

    # Reads in the training config.
    model_config = TaggerModelConfig()
    model_config_filename = os.path.join(args.model_indir, 'model.config')
    model_config.parse_from_txt_file(model_config_filename)

    vocab_list = ['word_vocab', 'dialog_act_vocab']

    args_dict = vars(args)
    vocab_name_to_file = [
        (vocab_name, args_dict.get(vocab_name))
        for vocab_name in vocab_list
    ]

    eval_model(
        args.eval_data_filename, dict(vocab_name_to_file), model_config,
        args.model_indir, num_cpus=args.num_cpus,
        max_sentence_length=args.max_sentence_length,
        eval_output_basename=args.outbase
    )

    exit(0)


if __name__ == '__main__':
    main()
