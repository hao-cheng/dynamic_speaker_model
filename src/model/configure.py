#!/usr/bin/env python2
"""Defines configuration object."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import codecs
import json

from .configure_helper import RNNConfig
from .configure_helper import OptConfig
from .configure_helper import int_str2bool
from .configure_helper import convert_str_by_type


class UserModelConfig(object):
    """Configuration for the user model."""

    __slots__ = (
        'word_vocab_size', 'word_embed_dim', 'keep_prob', 'train_word_embed',
        'num_latent_class', 'latent_feature_dim', 'user_state_rnn_config',
        'user_utterance_encoder_rnn_config',
        'user_utterance_decoder_rnn_config',
    )

    __object_slots__ = (
        'user_state_rnn_config',
        'user_utterance_encoder_rnn_config',
        'user_utterance_decoder_rnn_config',
    )

    __slots_dtype_func__ = {
        'word_vocab_size': int,
        'word_embed_dim': int,
        'train_word_embed': int_str2bool,
        'num_latent_class': int,
        'latent_feature_dim': int,
        'keep_prob': float,
    }

    def __init__(self):
        """Initialization."""
        # Inputs related parameters.
        self.word_vocab_size = 0
        self.word_embed_dim = 0
        self.train_word_embed = True

        self.num_latent_class = 0
        self.latent_feature_dim = 0

        self.keep_prob = 1.0

        self.user_state_rnn_config = RNNConfig()
        self.user_utterance_encoder_rnn_config = RNNConfig()
        self.user_utterance_decoder_rnn_config = RNNConfig()

        for attr in self.__slots__:
            if attr in self.__object_slots__:
                continue

            # For primitive attributes, the dtype function should be specified.
            if attr not in self.__slots_dtype_func__:
                raise ValueError('Unknown attribute {0} dtype!'.format(attr))

    def parse_from_dict(self, config_dict):
        """Builds the object from a dictionary."""
        for attr in self.__slots__:
            if attr in self.__object_slots__:
                continue
            setattr(
                self, attr, convert_str_by_type(config_dict.get(attr),
                                                self.__slots_dtype_func__[attr])
            )

        self.user_state_rnn_config.parse_from_dict(config_dict, 'user_state')
        self.user_utterance_encoder_rnn_config.parse_from_dict(
            config_dict, 'user_utterance_encoder'
        )
        self.user_utterance_decoder_rnn_config.parse_from_dict(
            config_dict, 'user_utterance_decoder'
        )
        return True

    def to_json_string(self):
        """Dumps the object attribute values to json string."""
        return json.dumps(dict(
            [
                (attr, getattr(self, attr).to_json_string())
                if attr in self.__object_slots__ else
                (attr, getattr(self, attr))
                for attr in self.__slots__
            ]
        ))

    def parse_from_json_string(self, json_string):
        """Parses object attributes from json string."""
        attributes_dict = json.loads(json_string)
        for attr in self.__slots__:
            if attr in self.__object_slots__:
                getattr(self, attr).parse_from_json_string(
                    attributes_dict[attr])
                continue

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


class UserModelConfiguration(object):
    """Configuration object for user model."""
    def __init__(self):
        """Initialization."""
        # Model related parameters.
        self.model_config = UserModelConfig()

        # Optimization related parameters.
        self.opt_config = OptConfig()

        # General enviroment parameters.
        self.initializer = None
        self.device = None
        self.num_cpus = 0
        self.save_model_dir = None

        # Learning protocol paramters.
        self.max_epoch_iter = 0
        self.lr_decay_rate = 1.0
        self.init_learning_rate = 0.0
        self.skip_valid_iter = 0
        self.validate_metric = None
        self.rand_seed = 0

    def __str__(self):
        """Print helper."""
        skip_vars = set(['model_config', 'opt_config'])

        config_str = '\n'.join([
            'configuration.{0}={1}'.format(attr, getattr(self, attr))
            for attr in self.__dict__ if attr not in skip_vars
        ])

        opt_config_str = str(self.opt_config)

        return '\n'.join([
            'Training configuration:', config_str, opt_config_str
        ])

    def parse_from_dict(self, config_dict):
        """Builds the object from a dictionary."""
        self.model_config.parse_from_dict(config_dict)

        self.opt_config.parse_from_dict(config_dict)

        # General enviroment parameters.
        self.initializer = config_dict.get('initializer')
        self.device = config_dict.get('device')
        self.num_cpus = convert_str_by_type(
            config_dict.get('num_cpus'), int)
        self.save_model_dir = config_dict.get('model_outdir')

        self.max_epoch_iter = convert_str_by_type(
            config_dict.get('max_iterations'), int)
        self.init_learning_rate = convert_str_by_type(
            config_dict.get('init_learning_rate'), float)
        self.lr_decay_rate = convert_str_by_type(
            config_dict.get('lr_decay_rate'), float)
        self.skip_valid_iter = convert_str_by_type(
            config_dict.get('num_noneval_iterations'), int)
        self.validate_metric = config_dict.get('validate_metric')

        return True

    def to_json_string(self):
        """Dumps the object attribute values to json string."""
        attributes_dict = dict(self.__dict__)
        attributes_dict['model_config'] = self.model_config.to_json_string()
        attributes_dict['opt_config'] = self.opt_config.to_json_string()
        return json.dumps(attributes_dict)

    def parse_from_json_string(self, json_string):
        """Parses the object attribute values from json string."""
        attributes_dict = json.loads(json_string)

        self.model_config.parse_from_json_string(attributes_dict['model_config'])
        self.opt_config.parse_from_json_string(attributes_dict['opt_config'])

        skip_attrs = set(['model_config', 'opt_config'])
        for attr in self.__dict__:
            if attr in skip_attrs:
                getattr(self, attr).parse_from_json_string(
                    attributes_dict[attr]
                )
                continue

            # Sets values for other attributes.
            setattr(self, attr, attributes_dict[attr])

        return True


class TaggerModelConfig(object):
    """Configuration for tagging model."""

    __slots__ = (
        'word_vocab_size', 'word_embed_dim', 'keep_prob', 'num_dialog_act',
        'train_word_embed', 'num_latent_class', 'latent_feature_dim',
        'dialog_act_embed_dim', 'user_state_rnn_config',
        'user_utterance_rnn_config', 'conversation_context_rnn_config',
        'adapt_user_model',
    )

    __object_slots__ = (
        'user_state_rnn_config', 'user_utterance_rnn_config',
        'conversation_context_rnn_config',
    )

    # The below parameters must be kept the same to ensure loading user model.
    __user_model_slots__ = (
        'word_vocab_size', 'word_embed_dim', 'num_latent_class',
        'latent_feature_dim', 'user_state_rnn_config',
        'user_utterance_rnn_config',
    )

    __slots_dtype_func__ = {
        'word_vocab_size': int,
        'word_embed_dim': int,
        'train_word_embed': int_str2bool,
        'num_dialog_act': int,
        'dialog_act_embed_dim': int,
        'num_latent_class': int,
        'latent_feature_dim': int,
        'keep_prob': float,
        'adapt_user_model': int_str2bool,
    }

    def __init__(self):
        """Initialization."""
        self.word_vocab_size = 0
        self.word_embed_dim = 0
        self.train_word_embed = True

        self.num_latent_class = 0
        self.latent_feature_dim = 0

        self.adapt_user_model = True

        self.keep_prob = 1.0

        self.num_dialog_act = 0
        self.dialog_act_embed_dim = 0

        # RNN ralated parameters.
        self.user_state_rnn_config = RNNConfig()
        self.user_utterance_rnn_config = RNNConfig()
        self.conversation_context_rnn_config = RNNConfig()

        for attr in self.__slots__:
            if attr in self.__object_slots__:
                continue

            # For primitive attributes, the dtype function should be specified.
            if attr not in self.__slots_dtype_func__:
                raise ValueError('Unknown attribute {0} dtype!'.format(attr))

    def parse_from_dict(self, config_dict):
        """Builds the object from a dictionary."""
        for attr in self.__slots__:
            if attr in self.__object_slots__:
                continue
            setattr(self, attr,
                    convert_str_by_type(config_dict.get(attr),
                                        self.__slots_dtype_func__[attr]))

        self.user_state_rnn_config.parse_from_dict(config_dict, 'user_state')
        self.user_utterance_rnn_config.parse_from_dict(config_dict,
                                                       'user_utterance')
        self.conversation_context_rnn_config.parse_from_dict(
            config_dict, 'conversation_context'
        )

        return True

    def update_user_model_related_parameter(self, user_model_config):
        """Updates parameters related with user model."""
        for attr in self.__user_model_slots__:
            if attr in self.__object_slots__:
                continue
            predefined_val = getattr(user_model_config, attr)
            cur_val = getattr(self, attr)
            if predefined_val != cur_val:
                print('Updates {0}={1} to {0}={2}'.format(
                    attr, cur_val, predefined_val), file=sys.stderr)
                setattr(self, attr, predefined_val)

        self.user_state_rnn_config.update_with_rnn_config(
            user_model_config.user_state_rnn_config)
        self.user_utterance_rnn_config.update_with_rnn_config(
            user_model_config.user_utterance_encoder_rnn_config)
        return True

    def to_json_string(self):
        """Dumps the object attribute values to json string."""
        return json.dumps(dict(
            [
                (attr, getattr(self, attr).to_json_string())
                if attr in self.__object_slots__ else
                (attr, getattr(self, attr))
                for attr in self.__slots__
            ]
        ))

    def parse_from_json_string(self, json_string):
        """Parses object attributes from json string."""
        attributes_dict = json.loads(json_string)
        for attr in self.__slots__:
            if attr in self.__object_slots__:
                getattr(self, attr).parse_from_json_string(
                    attributes_dict[attr])
                continue

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


class TaggerModelConfiguration(object):
    """Configuration object for all configurations of the tagger."""
    def __init__(self):
        """Initialization."""
        # Model related parameters.
        self.model_config = TaggerModelConfig()

        # Optimization related parameters.
        self.opt_config = OptConfig()

        # General enviroment parameters.
        self.initializer = None
        self.device = None
        self.num_cpus = 0
        self.save_model_dir = None

        self.pretrain_model_dir = None
        self.pretrain_model_config = None

        # Learning protocol paramters.
        self.max_epoch_iter = 0
        self.lr_decay_rate = 1.0
        self.init_learning_rate = 0.0
        self.skip_valid_iter = 0
        self.validate_metric = None
        self.rand_seed = 0

    def __str__(self):
        """Print helper."""
        skip_vars = set(['model_config', 'opt_config'])

        config_str = '\n'.join([
            'configuration.{0}={1}'.format(attr, getattr(self, attr))
            for attr in self.__dict__ if attr not in skip_vars
        ])

        opt_config_str = str(self.opt_config)

        return '\n'.join([
            'Training configuration:', config_str, opt_config_str
        ])

    def parse_from_dict(self, config_dict):
        """Builds the object from a dictionary."""
        self.model_config.parse_from_dict(config_dict)

        self.opt_config.parse_from_dict(config_dict)

        # General enviroment parameters.
        self.initializer = config_dict.get('initializer')
        self.device = config_dict.get('device')
        self.num_cpus = convert_str_by_type(
            config_dict.get('num_cpus'), int)
        self.save_model_dir = config_dict.get('model_outdir')
        self.max_epoch_iter = convert_str_by_type(
            config_dict.get('max_iterations'), int)
        self.init_learning_rate = convert_str_by_type(
            config_dict.get('init_learning_rate'), float)
        self.lr_decay_rate = convert_str_by_type(
            config_dict.get('lr_decay_rate'), float)
        self.skip_valid_iter = convert_str_by_type(
            config_dict.get('num_noneval_iterations'), int)
        self.validate_metric = config_dict.get('validate_metric')
        self.pretrain_model_dir = config_dict.get('pretrain_model_dir')
        self.pretrain_model_config = config_dict.get('pretrain_model_config')

        return True

    def to_json_string(self):
        """Dumps the object attribute values to json string."""
        attributes_dict = dict(self.__dict__)
        attributes_dict['model_config'] = self.model_config.to_json_string()
        attributes_dict['opt_config'] = self.opt_config.to_json_string()
        return json.dumps(attributes_dict)

    def parse_from_json_string(self, json_string):
        """Parses the object attribute values from json string."""
        attributes_dict = json.loads(json_string)

        self.model_config.parse_from_json_string(
            attributes_dict['model_config'])
        self.opt_config.parse_from_json_string(attributes_dict['opt_config'])

        skip_attrs = set(['model_config', 'opt_config'])
        for attr in self.__dict__:
            if attr in skip_attrs:
                getattr(self, attr).parse_from_json_string(
                    attributes_dict[attr]
                )
                continue

            # Sets values for other attributes.
            setattr(self, attr, attributes_dict[attr])

        return True
