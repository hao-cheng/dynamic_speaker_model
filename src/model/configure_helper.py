#!/usr/bin/env python2
"""Configuration helper functions."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import codecs
import json


def int_str2bool(int_str, false_value=0):
    """Converts an integer string to boolean."""
    return int(int_str) > false_value


def convert_str_by_type(str_val, type_func):
    """Converts the given string using the type_func."""
    try:
        converted_val = type_func(str_val)
    except ValueError:
        print('Can not convert {0} using {1}'.format(str_val, type_func),
              file=sys.stderr)
        sys.exit(1)
    return converted_val


def convert_optional_str_by_type(str_val, type_func, default_val):
    """Converts an optional string using the type_func, else uses the default.
    """
    if (str_val == '<UNK>') or (str_val is None):
        return type_func(default_val)

    return type_func(str_val)


class RNNConfig(object):
    """Configuration object for Recurrent Neural Network."""

    # Those slots values are determined by function single_directional_lstm.
    __slots__ = ('num_hidden', 'hidden_dim', 'residual_rnn', 'residual_method',
                 'keep_prob', 'input_keep_prob', 'state_keep_prob',
                 'output_keep_prob',
                 'forget_bias', 'carry_bias_init',
                 'couple_carry_transform_gates', 'couple_gate_lstm',
                 'vr_recurrent')

    __update_slots__ = ('num_hidden', 'hidden_dim', 'residual_rnn',
                        'residual_method', 'couple_carry_transform_gates',
                        'couple_gate_lstm')

    __slot_dtype_func__ = {
        'num_hidden': int,
        'hidden_dim': int,
        'residual_rnn': int_str2bool,
        'residual_method': str,
        'keep_prob': float,
        'input_keep_prob': float,
        'state_keep_prob': float,
        'output_keep_prob': float,
        'forget_bias': float,
        'carry_bias_init': float,
        'couple_carry_transform_gates': int_str2bool,
        'couple_gate_lstm': int_str2bool,
        'vr_recurrent': int_str2bool
    }

    def __init__(self):
        """Initialization."""
        self.num_hidden = 0
        self.hidden_dim = 0

        self.residual_rnn = True
        self.residual_method = None
        self.keep_prob = 1.0
        self.input_keep_prob = 1.0
        self.state_keep_prob = 1.0
        self.output_keep_prob = 1.0
        self.forget_bias = 1.0
        self.carry_bias_init = 1.0
        self.couple_carry_transform_gates = True
        self.couple_gate_lstm = False
        self.vr_recurrent = True

        for attr in self.__slots__:
            if self.__slot_dtype_func__.get(attr, None) is None:
                raise ValueError(
                    'attribute: {0} not defined type!'.format(attr))

    def parse_from_model_config(self, model_config):
        """Parses parameters from a ModelConfig object."""
        for attr in self.__slots__:
            if attr in model_config.__slots__:
                setattr(self, attr, getattr(model_config, attr))
            else:
                raise ValueError('RNNConfig.{0} not found!'.format(attr))
        return True

    def parse_from_dict(self, config_dict, prefix=None):
        """Parses parameters from a config dictionary."""
        for attr in self.__slots__:
            dtype_func = self.__slot_dtype_func__[attr]

            # This handles keep_prob specifications.
            _, base_att = attr.split('_', 1)

            default_val = config_dict.get(base_att)

            if default_val:
                attr_val = convert_optional_str_by_type(
                    config_dict.get(attr), dtype_func,
                    convert_str_by_type(config_dict.get(base_att), dtype_func)
                )
            else:
                attr_val = convert_str_by_type(
                    config_dict.get(attr), dtype_func)

            # If prefix specified, the non-prefix value would be over-written.
            prefix_attr_name = '{0}_{1}'.format(prefix, attr)
            prefix_attr_val = convert_optional_str_by_type(
                config_dict.get(prefix_attr_name), dtype_func, attr_val
            )

            setattr(self, attr, prefix_attr_val)

        return True

    def update_with_rnn_config(self, rnn_config, update_all=False):
        """Updates the current config with another RNNConfig."""
        update_set = self.__slots__ if update_all else self.__update_slots__
        for attr in update_set:
            new_val = getattr(rnn_config, attr)
            old_val = getattr(self, attr)
            if new_val != old_val:
                print('Updates RNNConfig.{0}={1}, to {0}={2}'.format(
                    attr, old_val, new_val), file=sys.stderr)
                setattr(self, attr, getattr(rnn_config, attr))
        return True

    def to_json_string(self):
        """Converts the object into json string."""
        return json.dumps(dict([
            (attr, getattr(self, attr)) for attr in self.__slots__
        ]))

    def parse_from_json_string(self, json_string):
        """Parses the object from json string."""
        attr_dict = json.loads(json_string)
        for attr in self.__slots__:
            setattr(self, attr, attr_dict[attr])

        return True


class OptConfig(object):
    """Configuration for optimizatin."""

    __slots__ = (
        'opt_method', 'clip_value', 'adam_beta1', 'adam_beta2', 'adam_epsilon'
    )

    def __init__(self):
        """Initialization."""
        self.opt_method = None
        self.clip_value = 0

        # Adam optimizer related parameters.
        self.adam_beta1 = 0.999
        self.adam_beta2 = 0.999
        self.adam_epsilon = 1e-6

    def __str__(self):
        """Converts object to string."""
        return '\n'.join([
            'opt_config.{0}={1}'.format(attr, getattr(self, attr))
            for attr in self.__slots__
        ])

    def parse_from_dict(self, config_dict):
        """Builds the object from a dictionary."""
        self.opt_method = config_dict.get('opt_method')
        self.clip_value = convert_str_by_type(
            config_dict.get('clip_value'), float)

        self.adam_beta1 = convert_str_by_type(
            config_dict.get('adam_beta1'), float)
        self.adam_beta2 = convert_str_by_type(
            config_dict.get('adam_beta2'), float)
        self.adam_epsilon = convert_str_by_type(
            config_dict.get('adam_epsilon'), float)

        return True

    def to_json_string(self):
        """Dumps the object attribute values to json string."""
        return json.dumps(dict(
            [(attr, getattr(self, attr)) for attr in self.__slots__]
        ))

    def parse_from_json_string(self, json_string):
        """Parses object attributes from json string."""
        attributes_dict = json.loads(json_string)
        for attr in self.__slots__:
            setattr(self, attr, attributes_dict[attr])
        return True

    def dump_to_file(self, filename):
        """Dumps the config to file in the format of JSON string."""
        with codecs.open(filename, mode='wt', encoding='utf8') as fout:
            fout.write(self.to_json_string())
        return True

    def parse_from_txt_file(self, filename):
        """Parses the config from the text file."""
        with codecs.open(filename, encoding='utf8') as fin:
            if not self.parse_from_json_string(fin.readline()):
                raise ValueError('Can not parse model config!')
        return True
