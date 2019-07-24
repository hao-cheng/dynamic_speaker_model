#!/usr/bin/env python2
"""Implements data interface."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import json
import codecs
import cPickle as pickle

from tqdm import tqdm
import numpy as np
import pandas as pd


# Defines the reserved tokens.
UNK_TOKEN = u'<UNK>'
PAD_TOKEN = u'<PAD>'
ROOT_TOKEN = u'<ROOT>'
RESERVED_TOKEN_LIST = [PAD_TOKEN, UNK_TOKEN, ROOT_TOKEN]

# Defines print decoration.
PRINT_DECORATION = 16 * '='


class Token(object):
    """Token object."""

    __slots__ = ('word_form', 'word_idx')

    def __init__(self, token_str=None):
        self.word_form = token_str
        self.word_idx = -1

    def to_json_string(self):
        """Dumps to json string."""
        attr_val_list = [(attr, getattr(self, attr)) for attr in self.__slots__]
        return json.dumps(dict(attr_val_list))

    def parse_from_json_string(self, json_str):
        """Parse object from json string."""
        attr_val_dict = json.loads(json_str)
        for attr, val in attr_val_dict.items():
            setattr(self, attr, val)

        return True


class Sentence(object):
    """Sentence object."""

    __slots__ = ('sentence_id', 'sentence_length', 'sentence_label', 'tokens',
                 'turn_index', 'sentence_label_index',)

    def __init__(self):
        self.sentence_id = None
        self.sentence_length = 0
        self.sentence_label = None
        self.sentence_label_index = -1
        self.tokens = []
        self.turn_index = -1

    def __str__(self):
        """Print utility."""
        sent_attr_vals = [
            'Sentence.{0}={1}'.format(attr, getattr(self, attr))
            for attr in self.__slots__ if attr != 'tokens'
        ]
        token_attr_vals = [
            'Sentence.token_{0}={1},{2}'.format(tk_idx, token.word_form,
                                                token.word_idx)
            for tk_idx, token in enumerate(self.tokens)
        ]
        return '\n'.join(sent_attr_vals + token_attr_vals)

    def parse_from_text_sentence(self, txt_line, turn_index):
        """Parses the sentence from the text, split by spaces."""
        self.tokens = [
            Token(token) for token in txt_line.split()
        ]
        self.sentence_length = len(self.tokens)
        self.turn_index = turn_index
        return True

    def to_json_string(self):
        """Dumps the object into json string."""
        attr_val_tuples = [(attr, getattr(self, attr))
                           for attr in self.__slots__ if attr != 'tokens']
        attr_val_tuples.append(
            ('tokens', json.dumps(
                [token.to_json_string() for token in self.tokens])
            )
        )
        return json.dumps(dict(attr_val_tuples))

    def parse_from_json_string(self, json_str):
        """Parses sentence object from json string form."""
        attr_val_dict = json.loads(json_str)
        for attr, val in attr_val_dict.items():
            if attr == 'tokens':
                token_json_list = json.loads(val)
                self.tokens = [Token() for _ in xrange(len(token_json_list))]
                for token, token_json in zip(self.tokens, token_json_list):
                    token.parse_from_json_string(token_json)
                continue

            setattr(self, attr, val)

        return True

    def build_from_json_string(self, json_str):
        """Builds the object from json string."""
        if not self.parse_from_json_string(json_str):
            raise ValueError(
                'Can not parse topic decision from {0}'.format(json_str))
        return self


class UserDialog(object):
    """Dialog object."""

    __slots__ = ('session_id', 'caller', 'num_turn', 'sentences', 'gender',
                 'dialect', 'education', 'topic',)

    __object_slots__ = ('sentences',)

    def __init__(self):
        """Default initializer."""
        # Required parameters defined below.
        # String for the session id.
        self.session_id = None

        # Integer for the total number of turns.
        self.num_turn = 0

        # List of Sentence objects.
        self.sentences = []

        # String for the caller gender.
        self.gender = None

        # String for the caller dialect.
        self.dialect = None

        # Integer for the caller education level.
        self.education = -1

        # String for the discussion topic description.
        self.topic = None

    def add_sentence_from_text(self, txt_line, turn_index):
        """Parses the sentence from the text, split by spaces."""
        sentence = Sentence()
        if not sentence.parse_from_text_sentence(txt_line, turn_index):
            raise ValueError('Can not parse sentence from text!')

        self.sentences.append(sentence)
        self.num_turn += 1
        return True

    def to_json_string(self):
        """Dumps the object into json string."""
        attr_val_tuples = [(attr, getattr(self, attr))
                           for attr in self.__slots__
                           if attr not in self.__object_slots__]
        attr_val_tuples.append(
            ('sentences', json.dumps(
                [sent.to_json_string() for sent in self.sentences])
            )
        )
        return json.dumps(dict(attr_val_tuples))

    def parse_from_json_string(self, json_str):
        """Parses sentence object from json string form."""
        attr_val_dict = json.loads(json_str)
        for attr, val in attr_val_dict.items():
            if attr == 'sentences':
                json_list = json.loads(val)
                self.sentences[:] = [
                    Sentence().build_from_json_string(json_str)
                    for json_str in json_list
                ]
                continue

            setattr(self, attr, val)

        return True

    def build_from_json_string(self, json_str):
        """Builds the object from json string."""
        if not self.parse_from_json_string(json_str):
            raise ValueError(
                'Can not parse topic decision from {0}'.format(json_str))
        return self


class TwoPartyDialog(object):
    """Dialog object."""

    __slots__ = ('session_id', 'caller_a_turn_indices',
                 'caller_b_turn_indices',
                 'num_turn', 'sentences',
                 'a_gender', 'a_dialect', 'a_education',
                 'b_gender', 'b_dialect', 'b_education',
                 'topic',)

    __object_slots__ = ('sentences',)

    def __init__(self):
        """Default initializer."""
        # Required parameters defined below.
        # String for the session id.
        self.session_id = None

        # Integer for the total number of turns.
        self.num_turn = 0

        # List of Sentence objects.
        self.sentences = []

        # List of caller a turn indices.
        self.caller_a_turn_indices = []
        self.caller_b_turn_indices = []

        # String for the caller gender.
        self.a_gender = None

        # String for the caller dialect.
        self.a_dialect = None

        # Integer for the caller education level.
        self.a_education = -1

        # String for the caller gender.
        self.b_gender = None

        # String for the caller dialect.
        self.b_dialect = None

        # Integer for the caller education level.
        self.b_education = -1

        # String for the discussion topic description.
        self.topic = None

    def add_sentence_from_text(self, txt_line, turn_index):
        """Parses the sentence from the text, split by spaces."""
        sentence = Sentence()
        if not sentence.parse_from_text_sentence(txt_line, turn_index):
            raise ValueError('Can not parse sentence from text!')

        self.sentences.append(sentence)
        self.num_turn += 1
        return True

    def to_json_string(self):
        """Dumps the object into json string."""
        attr_val_tuples = [(attr, getattr(self, attr))
                           for attr in self.__slots__
                           if attr not in self.__object_slots__]
        attr_val_tuples.append(
            ('sentences', json.dumps(
                [sent.to_json_string() for sent in self.sentences])
            )
        )
        return json.dumps(dict(attr_val_tuples))

    def parse_from_json_string(self, json_str):
        """Parses sentence object from json string form."""
        attr_val_dict = json.loads(json_str)
        for attr in self.__slots__:
            val = attr_val_dict.get(attr, None)

            if val is None:
                raise ValueError('Can not parse {0}!'.format(attr))

            if attr == 'sentences':
                json_list = json.loads(val)
                self.sentences[:] = [
                    Sentence().build_from_json_string(json_str)
                    for json_str in json_list
                ]
                continue

            setattr(self, attr, val)

        return True

    def build_from_json_string(self, json_str):
        """Builds the object from json string."""
        if not self.parse_from_json_string(json_str):
            raise ValueError(
                'Can not parse topic decision from {0}'.format(json_str))
        return self


def read_vocab(filename, check_reserved_symbols=False):
    """Reads in vocabulary into a dicionary."""
    value2id_list = []
    with codecs.open(filename, encoding='utf8') as fin:
        value2id_list = [
            (line.strip(), index) for index, line in enumerate(fin)
        ]
    value2id = dict(value2id_list)

    if check_reserved_symbols:
        if value2id.get(UNK_TOKEN, None) is None:
            raise ValueError('UNK symbol {0} is not found in vocab {1}!'.format(
                UNK_TOKEN, filename))

        if value2id.get(PAD_TOKEN, None) is None:
            raise ValueError('PAD symbol {0} is not found in vocab {1}!'.format(
                PAD_TOKEN, filename))

        if value2id.get(ROOT_TOKEN, None) is None:
            raise ValueError('ROOT symbol {0} is not found in vocab {1}!'.format(
                ROOT_TOKEN, filename))

    return value2id


def index_sentence(sentence, word_vocab, unk_idx, cnt_info):
    """Index sentence using the given vocab."""
    for token in sentence.tokens:
        token.word_idx = word_vocab.get(token.word_form, unk_idx)
        if token.word_idx == unk_idx:
            cnt_info['word_oov_cnt'] += 1
        cnt_info['word_token_cnt'] += 1


def index_sentence_w_dialog_act(sentence, word_vocab, unk_idx, dialog_act_vocab,
                                cnt_info):
    """Index sentence using the given vocab."""
    for token in sentence.tokens:
        token.word_idx = word_vocab.get(token.word_form, unk_idx)
        if token.word_idx == unk_idx:
            cnt_info['word_oov_cnt'] += 1
        cnt_info['word_token_cnt'] += 1

    dialog_act_index = dialog_act_vocab.get(sentence.sentence_label, -1)
    if dialog_act_index < 0:
        raise ValueError('Unknown dialog act: {0}'.format(
            sentence.sentence_label
        ))
    sentence.sentence_label_index = dialog_act_index
    cnt_info['da_cnt'] += 1


def index_user_dialog(user_dialog, word_vocab, unk_idx, cnt_info):
    """Index sentence using the given vocab."""
    for sentence in user_dialog.sentences:
        index_sentence(sentence, word_vocab, unk_idx, cnt_info)

    return user_dialog


def index_user_dialog_with_dialog_act(user_dialog, word_vocab, dialog_act_vocab,
                                      unk_idx, cnt_info):
    """Converts tokens into indices using the given vocab."""
    for sentence in user_dialog.sentences:
        index_sentence_w_dialog_act(sentence, word_vocab, unk_idx,
                                    dialog_act_vocab, cnt_info)

    return user_dialog


def read_user_dialog(filename):
    """Reads in UserDialog object from file."""
    with codecs.open(filename, encoding='utf8') as fin:
        user_dialog_list = [
            UserDialog().build_from_json_string(line.strip())
            for line in fin
        ]
    return user_dialog_list


def read_two_party_dialog(filename):
    """Reads in TwoPartyDialog object from file."""
    with codecs.open(filename, encoding='utf8') as fin:
        dialog_list = [
            TwoPartyDialog().build_from_json_string(line.strip())
            for line in fin
        ]
    return dialog_list


def write_jsonable_object_to_file(object_list, filename):
    """Writes a list of JSONable objects to file.

    Args:
        object_list: A list of objects that have to_json_string() method.
        filenameL String for the output filename.
    """
    with codecs.open(filename, mode='wt', encoding='utf8') as fout:
        for obj in tqdm(object_list):
            fout.write(obj.to_json_string())
            fout.write('\n')


class DataContainer(object):
    """Data container object."""

    def __init__(self, filename, word_vocab_filename, batch_size,
                 max_sentence_length, shuffle_data=False, rand_seed=0,
                 start_offset=0):
        """Initialization."""
        if start_offset > 0:
            print('Carries out start_offset {0}'.format(start_offset),
                  file=sys.stderr)

        # Reads in the sentence objects.
        dialog_list = read_user_dialog(filename)

        # Reads vocabulary for word.
        word_vocab = read_vocab(word_vocab_filename,
                                check_reserved_symbols=True)

        self.unk_idx = word_vocab[UNK_TOKEN]
        self.pad_idx = word_vocab[PAD_TOKEN]
        self.root_idx = word_vocab[ROOT_TOKEN]

        self.max_sentence_length = max_sentence_length

        cnt_info = {
            'word_oov_cnt': 0, 'word_token_cnt': 0,
        }
        self.dialog_list = [
            index_user_dialog(dialog, word_vocab, self.unk_idx, cnt_info)
            for dialog in dialog_list
        ]

        self.rand_seed = rand_seed

        self.print_dest = sys.stdout

        print(
            'word_oov ratio: {:.3f}'.format(
                float(cnt_info['word_oov_cnt']) / cnt_info['word_token_cnt']),
            file=self.print_dest
        )

        np.random.seed(rand_seed)
        self.num_sample = len(self.dialog_list)

        print('Read in {0} dialogs from {1}'.format(self.num_sample, filename),
              file=self.print_dest)

        self.shuffle_data = shuffle_data
        self.processed_sample_cnt = self.num_sample
        self.batch_size = batch_size

    def start_epoch(self, batch_size=None):
        """Starts an new epoch."""
        if self.processed_sample_cnt < self.num_sample:
            print(
                'Warning: there are still remaining {0} dialogs!'.format(
                    self.num_sample - self.processed_sample_cnt),
                file=self.print_dest
            )

        print('Starting a new epoch!', file=self.print_dest)
        self.processed_sample_cnt = 0

        if self.shuffle_data:
            print('Shuffle data', file=self.print_dest)
            np.random.shuffle(self.dialog_list)

        if batch_size:
            print('Resize the batch_size for data_container!',
                  file=self.print_dest)
            self.batch_size = batch_size

        print('The new epoch will have batch_size: {0}'.format(self.batch_size),
              file=self.print_dest)

    def save_embed_to_file(self, embed_dict, output_basename):
        """Saves the extracted embeddings to the output file."""
        if self.shuffle_data:
            raise ValueError('This function can not be correct if shuffle_data')

        session_id_list = ['{0}={1}'.format(dialog.session_id, dialog.caller)
                           for dialog in self.dialog_list]

        num_rows = None

        # Checks the output integrity.
        for embed_name, embed_mat_group in embed_dict.items():
            if isinstance(embed_mat_group, list):
                if len(embed_mat_group) != len(self.dialog_list):
                    raise ValueError(
                        '{0} embed list length != num_of_dialogs'.format(
                            embed_name))
            elif isinstance(embed_mat_group, np.ndarray):
                if num_rows and embed_mat_group.shape[0] != num_rows:
                    raise ValueError(
                        '{0} embed num_row={1} != {2}'.format(
                            embed_name, embed_mat_group.shape[0], num_rows)
                    )
            else:
                raise ValueError(
                    'Unknown empty_mat_group type {0} for {1}'.format(
                        type(embed_mat_group), embed_name))

        embed_dict['session_id'] = session_id_list
        with open('{0}.all_embeds'.format(output_basename), 'wb') as fout:
            pickle.dump(embed_dict, fout)

        print('Embedding dummped!', file=sys.stderr)

    def extract_batches(self):
        """Extracts one batch of data."""
        self.start_epoch()

        # For now, the batch_size is fixed to be 1.
        print('There are {0} dialogs in one epoch'.format(self.num_sample),
              file=self.print_dest)

        for dialog in self.dialog_list:
            sentence_lengths = []
            sentence_indices = []
            target_sentence_indices = []
            max_sentence_length = 0

            for sentence in dialog.sentences:
                cur_sentence_length = sentence.sentence_length

                if cur_sentence_length > self.max_sentence_length:
                    cur_sentence_length = self.max_sentence_length

                if cur_sentence_length > max_sentence_length:
                    max_sentence_length = cur_sentence_length

                sentence_lengths.append(cur_sentence_length)

            for (sentence, sent_length) in zip(
                    dialog.sentences, sentence_lengths):
                padding_length = (max_sentence_length - sent_length)

                sentence_list = [self.root_idx] + [
                    token.word_idx
                    for token in sentence.tokens[:max_sentence_length]
                ] + [self.pad_idx] + [
                    self.pad_idx for _ in xrange(padding_length)
                ]

                sentence_indices.append(sentence_list[:-1])
                target_sentence_indices.append(sentence_list[1:])

            num_sample = len(sentence_lengths)
            self.processed_sample_cnt += 1

            input_tuple_list = [
                ('sentence_indices', sentence_indices),
                ('sentence_lengths', sentence_lengths),
                ('target_sentence_indices', target_sentence_indices),
                ('max_sentence_length', max_sentence_length),
            ]

            yield (input_tuple_list, None, num_sample)


class DialogActDataContainer(object):
    """Data container object for dialog act prediction."""

    def __init__(self, filename, word_vocab_filename, dialog_act_vocab_filename,
                 batch_size, max_sentence_length, shuffle_data=False,
                 rand_seed=0):
        """Initialization."""
        dialog_list = read_two_party_dialog(filename)

        # Reads vocabulary for word.
        word_vocab = read_vocab(word_vocab_filename,
                                check_reserved_symbols=True)

        dialog_act_vocab = read_vocab(dialog_act_vocab_filename)
        self.dialog_act_vocab = dialog_act_vocab

        self.unk_idx = word_vocab[UNK_TOKEN]
        self.pad_idx = word_vocab[PAD_TOKEN]
        self.root_idx = word_vocab[ROOT_TOKEN]

        self.max_sentence_length = max_sentence_length

        cnt_info = {'word_oov_cnt': 0, 'word_token_cnt': 0,
                    'da_cnt': 0,}

        self.dialog_list = [
            index_user_dialog_with_dialog_act(
                dialog, word_vocab, dialog_act_vocab, self.unk_idx, cnt_info
            )
            for dialog in dialog_list
        ]

        self.num_predictions = cnt_info['da_cnt']
        self.rand_seed = rand_seed
        self.print_dest = sys.stdout

        print(
            'word_oov ratio: {:.3f}'.format(
                float(cnt_info['word_oov_cnt']) / cnt_info['word_token_cnt']),
            file=self.print_dest
        )

        np.random.seed(rand_seed)
        self.num_sample = len(self.dialog_list)

        print('Read in {0} dialogs from {1}'.format(self.num_sample, filename),
              file=self.print_dest)

        print('Total number of dialog act predictions: {0}'.format(
            self.num_predictions), file=self.print_dest)

        self.shuffle_data = shuffle_data
        self.processed_sample_cnt = self.num_sample
        self.batch_size = batch_size

    def start_epoch(self, batch_size=None):
        """Starts an new epoch."""
        if self.processed_sample_cnt < self.num_sample:
            print(
                'Warning: there are still remaining {0} dialogs!'.format(
                    self.num_sample - self.processed_sample_cnt),
                file=self.print_dest
            )

        print('Starting a new epoch!', file=self.print_dest)
        self.processed_sample_cnt = 0

        if self.shuffle_data:
            print('Shuffle data', file=self.print_dest)
            np.random.shuffle(self.dialog_list)

        if batch_size:
            print('Resize the batch_size for data_container!',
                  file=self.print_dest)
            self.batch_size = batch_size

        print('The new epoch will have batch_size: {0}'.format(self.batch_size),
              file=self.print_dest)

    def extract_batches(self):
        """Extracts one batch of data."""
        self.start_epoch()

        # For now, the batch_size is fixed to be 1.
        print('There are {0} dialogs in one epoch'.format(self.num_sample),
              file=self.print_dest)

        for dialog in self.dialog_list:

            sentence_a_lengths = []
            sentence_a_indices = []

            sentence_b_lengths = []
            sentence_b_indices = []

            sentence_lengths = []

            num_a_turn = 0
            num_b_turn = 0

            caller_a_turn_indices = []
            caller_b_turn_indices = []

            dialog_act_indices = []
            max_sentence_length = 0

            for turn_index, sentence in enumerate(dialog.sentences):
                cur_sentence_length = sentence.sentence_length

                if cur_sentence_length > self.max_sentence_length:
                    cur_sentence_length = self.max_sentence_length

                if cur_sentence_length > max_sentence_length:
                    max_sentence_length = cur_sentence_length

                sentence_lengths.append(cur_sentence_length)

                if sentence.sentence_id == 'A':
                    caller_a_turn_indices.append(turn_index)
                elif sentence.sentence_id == 'B':
                    caller_b_turn_indices.append(turn_index)
                else:
                    raise ValueError(
                        'Unknown sentence_id: {0}'.format(sentence.sentence_id))

            for (sentence, sent_length) in zip(
                    dialog.sentences, sentence_lengths):
                padding_length = (max_sentence_length - sent_length)

                sentence_list = [self.root_idx] + [
                    token.word_idx
                    for token in sentence.tokens[:max_sentence_length]
                ] + [self.pad_idx] + [
                    self.pad_idx for _ in xrange(padding_length)
                ]

                if sentence.sentence_id == 'A':
                    sentence_a_indices.append(sentence_list[:-1])
                    sentence_a_lengths.append(sent_length)
                    num_a_turn += 1
                elif sentence.sentence_id == 'B':
                    sentence_b_indices.append(sentence_list[:-1])
                    sentence_b_lengths.append(sent_length)
                    num_b_turn += 1
                else:
                    raise ValueError(
                        'Unknown sentence_id: {0}'.format(sentence.sentence_id))

                dialog_act_index = sentence.sentence_label_index
                assert dialog_act_index > -1
                dialog_act_indices.append(sentence.sentence_label_index)

            map_ab_indices_to_turn_indices, _ = zip(*sorted(
                [
                    (concat_index, turn_index)
                    for concat_index, turn_index in enumerate(
                        caller_a_turn_indices + caller_b_turn_indices
                    )
                ],
                key=lambda x: x[1]
            ))

            num_sample = len(sentence_lengths)
            self.processed_sample_cnt += 1

            input_tuple_list = [
                ('sentence_a_indices', sentence_a_indices),
                ('sentence_a_lengths', sentence_a_lengths),
                ('sentence_b_indices', sentence_b_indices),
                ('sentence_b_lengths', sentence_b_lengths),
                ('map_ab_indices_to_turn_indices',
                 map_ab_indices_to_turn_indices),
                ('max_sentence_length', max_sentence_length),
            ]

            yield (input_tuple_list, dialog_act_indices, num_sample)

    def write_prediction(self, label_probs, output_filename):
        """Writes out the prediction."""
        if self.shuffle_data:
            raise ValueError(
                'Can not map the prediction back, because data is shuffled')

        if len(label_probs) != self.num_predictions:
            raise ValueError(
                'shape[0] of label_probs must be the same as num_predictions')

        turn_index_offset = 0

        da_index_to_name = dict([
            (index, da_name)
            for da_name, index in self.dialog_act_vocab.items()
        ])

        output_rows = []

        for dialog in self.dialog_list:
            start = turn_index_offset
            end = start + len(dialog.sentences)

            for turn_index, (hypo_prob, sentence) in enumerate(zip(
                    label_probs[start:end], dialog.sentences)):
                max_index = np.argmax(hypo_prob)
                output_rows.append({
                    'dialog_act_label': sentence.sentence_label,
                    'predicted_dialog_act_label': da_index_to_name[max_index],
                    'turn_id': turn_index,
                    'sentence_id': sentence.sentence_id,
                    'session_id': dialog.session_id
                })

            turn_index_offset = end

        dataframe = pd.DataFrame(output_rows)

        dataframe.to_csv(output_filename, sep='\t', encoding='utf8',
                         index=False)

        return True
