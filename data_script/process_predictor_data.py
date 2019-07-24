#!/usr/bin/env python
"""Extracts data for tagger model."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import argparse
import re
import string
import StringIO
import ConfigParser

import spacy

from .swda_utils.swda import CorpusReader
from .py_data_lib.data_interface import TwoPartyDialog, UserDialog, Sentence
from .py_data_lib.data_interface import write_jsonable_object_to_file


nlp = spacy.load('en', disable=['tagger', 'parser', 'ner'])

# Nonverbals can be in the form of <throat_clearing> or throat_clearing.
# nonverbal_reg_exp = r"\<+[^\<\>]+\>+|[a-zA-Z]+_[a-zA-Z]+"
nonverbal_reg_exp_list = [re.compile(r"\<+[^\<\>]+\>+"), re.compile(r"\w+_\w+")]
nonverbal_token = "NONVERBAL"

skip_punt_reg_exp = re.compile(r"[#\-\(\)]+")


def assert_caller_attribute(user_dialog_obj, utterance):
    """Asserts whether the caller attributes matches with the utterance."""
    if user_dialog_obj.caller != utterance.caller:
        raise ValueError(
            'user_dialog.caller[{0}] != utterance.caller[{1}]'.format(
                user_dialog_obj.caller, utterance.caller
            ))
    if user_dialog_obj.gender != utterance.caller_sex:
        raise ValueError(
            'user_dialog.gender[{0}] != utterance.caller_sex[{1}]'.format(
                user_dialog_obj.gender, utterance.caller_sex
            ))
    if user_dialog_obj.dialect != utterance.caller_dialect_area:
        raise ValueError(
            'user_dialog.dialect[{0}] != utterance.dialect[{1}]'.format(
                user_dialog_obj.dialect, utterance.caller_dialect_area
            ))
    if user_dialog_obj.education != utterance.caller_education:
        raise ValueError(
            'user_dialog.education[{0}] != utterance.education[{1}]'.format(
                user_dialog_obj.education, utterance.caller_education
            ))


def update_user_dialog(user_dialog_obj, utterance, turn_index, lowercase=True,
                       skip_punct=False, filter_disfluency=True,
                       skip_mark_nonverbal=False, verbose=False):
    """Updates the given UserDialog object."""
    assert_caller_attribute(user_dialog_obj, utterance)
    sentence = Sentence()

    utterance_text = u' '.join(
        utterance.text_words(filter_disfluency=filter_disfluency)
    )

    for nonverbal_reg_exp in nonverbal_reg_exp_list:
        find_match = re.search(nonverbal_reg_exp, utterance_text)
        if verbose and find_match:
            print('Found marked utterance from\n ', utterance_text,
                  file=sys.stdout)
        if skip_mark_nonverbal:
            # Removes the non-verbal expression.
            utterance_text = re.sub(nonverbal_reg_exp, ' ', utterance_text)
        else:
            # Normalizes the non-verbal expression.
            utterance_text = re.sub(nonverbal_reg_exp, nonverbal_token,
                                    utterance_text)

        if verbose and find_match:
            print('Normlized marked utterance:\n ', utterance_text,
                  file=sys.stdout)

    utterance_text = re.sub(skip_punt_reg_exp, ' ', utterance_text)

    doc = nlp(utterance_text)

    word_list = [
        token.text.lower()
        if lowercase and token.text != nonverbal_token else token.text
        for token in doc
    ]

    if skip_punct:
        # Skips punctuation if required.
        word_list = [
            word
            for word in word_list if word not in string.punctuation
        ]

    if not word_list:
        print('Empty utterance: ', utterance.text, file=sys.stderr)
        print('swda_filename: ', utterance.swda_filename, file=sys.stderr)
        print('sentence_label: ', utterance.damsl_act_tag(), file=sys.stderr)
        token_list = utterance.text_words(filter_disfluency=True)
        for token in token_list:
            print('Token: ', token, file=sys.stderr)

        word_list.append(nonverbal_token)

    # Extracts DAMSL dialog acts.
    sentence.sentence_label = utterance.damsl_act_tag()

    sentence.parse_from_text_sentence(u' '.join(word_list), turn_index)

    sentence.sentence_id = '{0}-{1}'.format(utterance.swda_filename, turn_index)

    user_dialog_obj.sentences.append(sentence)
    return True


def merge_utterance_based_on_act_tag(user_dialog):
    """Merges utterance with its previous one based on '+' dialog act tag."""
    sentence_list = []

    for cur_sentence in user_dialog.sentences:
        if cur_sentence.sentence_label == '+':
            if not sentence_list:
                print('The first sentence in the dialog with + tag',
                      file=sys.stderr)
                continue

            # Merges the current sentence with the previous one.
            prev_sentence = sentence_list.pop()
            full_token_list = prev_sentence.tokens + cur_sentence.tokens

            # Only updates the sentence_length and tokens.
            prev_sentence.tokens[:] = full_token_list
            prev_sentence.sentence_length = len(full_token_list)
            sentence_list.append(prev_sentence)
        else:
            sentence_list.append(cur_sentence)

    user_dialog.sentences[:] = sentence_list


def merge_two_party_user_dialog(user_a_dialog, user_b_dialog):
    """Merges two party user dialogs."""
    two_party_dialog = TwoPartyDialog()

    # Gathers meta information from separate user dialogs.
    two_party_dialog.topic = user_a_dialog.topic
    two_party_dialog.session_id = user_a_dialog.session_id

    two_party_dialog.a_gender = user_a_dialog.gender
    two_party_dialog.a_dialect = user_a_dialog.dialect
    two_party_dialog.a_education = user_a_dialog.education

    two_party_dialog.b_gender = user_b_dialog.gender
    two_party_dialog.b_dialect = user_b_dialog.dialect
    two_party_dialog.b_education = user_b_dialog.education

    user_a_sentences = list(user_a_dialog.sentences)
    user_b_sentences = list(user_b_dialog.sentences)

    turn_index = 0
    utterance_list = []
    caller_a_turn_indices = []
    caller_b_turn_indices = []
    a_index = 0
    b_index = 0

    while a_index < len(user_a_sentences) and b_index < len(user_b_sentences):
        a_utt = user_a_sentences[a_index]
        b_utt = user_b_sentences[b_index]

        utterance = Sentence()
        if a_utt.turn_index < b_utt.turn_index:
            utterance.parse_from_json_string(a_utt.to_json_string())
            utterance.turn_index = turn_index
            utterance.sentence_id = 'A'
            caller_a_turn_indices.append(turn_index)
            a_index += 1
        else:
            utterance.parse_from_json_string(b_utt.to_json_string())
            utterance.turn_index = turn_index
            utterance.sentence_id = 'B'
            caller_b_turn_indices.append(turn_index)
            b_index += 1

        turn_index += 1

        utterance_list.append(utterance)

    assert turn_index == a_index + b_index

    while a_index < len(user_a_sentences):
        a_utt = user_a_sentences[a_index]
        utterance = Sentence()
        utterance.parse_from_json_string(a_utt.to_json_string())
        utterance.turn_index = turn_index
        utterance.sentence_id = 'A'
        caller_a_turn_indices.append(turn_index)
        a_index += 1
        turn_index += 1
        utterance_list.append(utterance)

    while b_index < len(user_b_sentences):
        b_utt = user_b_sentences[b_index]
        utterance = Sentence()
        utterance.parse_from_json_string(b_utt.to_json_string())
        utterance.turn_index = turn_index
        utterance.sentence_id = 'B'
        caller_b_turn_indices.append(turn_index)
        b_index += 1
        turn_index += 1
        utterance_list.append(utterance)

    assert len(utterance_list) == len(user_a_sentences) + len(user_b_sentences)

    two_party_dialog.sentences[:] = utterance_list
    two_party_dialog.caller_a_turn_indices[:] = caller_a_turn_indices
    two_party_dialog.caller_b_turn_indices[:] = caller_b_turn_indices
    two_party_dialog.num_turn = len(utterance_list)

    return two_party_dialog


def extract_caller_dialog(swda_basedir, lowercase=True, skip_punct=True,
                          skip_mark_nonverbal=False):
    """Extracts utterances and group by caller."""
    corpus = CorpusReader(swda_basedir)
    user_dialog_list = []
    two_party_dialog_list = []
    empty_utterance_cnt = 0
    total_utterance_cnt = 0
    after_merge_total_utterance_cnt = 0
    for trans in corpus.iter_transcripts(display_progress=False):
        if not trans.utterances:
            print('Skip trans wo utterances:', trans.conversation_no)
            continue

        # User A is the from caller user.
        user_a_dialog = UserDialog()
        user_a_dialog.session_id = 'session_{0}'.format(trans.conversation_no)
        user_a_dialog.caller = 'A'
        user_a_dialog.gender = trans.from_caller_sex
        user_a_dialog.education = trans.from_caller_education
        user_a_dialog.dialect = trans.from_caller_dialect_area
        user_a_dialog.topic = trans.topic_description

        # User B is the to caller user.
        user_b_dialog = UserDialog()
        user_b_dialog.session_id = 'session_{0}'.format(trans.conversation_no)
        user_b_dialog.caller = 'B'
        user_b_dialog.gender = trans.to_caller_sex
        user_b_dialog.education = trans.to_caller_education
        user_b_dialog.dialect = trans.to_caller_dialect_area
        user_b_dialog.topic = trans.topic_description

        turn_index_count = 0
        num_turn_a = 0
        num_turn_b = 0

        for utt in trans.utterances:
            total_utterance_cnt += 1
            if utt.caller == 'A':
                if update_user_dialog(
                        user_a_dialog, utt, utt.transcript_index,
                        lowercase=lowercase, skip_punct=skip_punct,
                        skip_mark_nonverbal=skip_mark_nonverbal):
                    turn_index_count += 1
                    num_turn_a += 1
                else:
                    print('transcript_index: ', utt.transcript_index, file=sys.stderr)
                    empty_utterance_cnt += 1
            elif utt.caller == 'B':
                if update_user_dialog(
                        user_b_dialog, utt, utt.transcript_index,
                        lowercase=lowercase, skip_punct=skip_punct,
                        skip_mark_nonverbal=skip_mark_nonverbal):
                    turn_index_count += 1
                    num_turn_b += 1
                else:
                    print('transcript_index: ', utt.transcript_index, file=sys.stderr)
                    empty_utterance_cnt += 1
            else:
                raise ValueError('Unknown caller {0}'.format(utt.caller))

        assert len(user_a_dialog.sentences) == num_turn_a
        assert len(user_b_dialog.sentences) == num_turn_b

        user_a_dialog.num_turn = num_turn_a
        user_b_dialog.num_turn = num_turn_b

        merge_utterance_based_on_act_tag(user_a_dialog)
        merge_utterance_based_on_act_tag(user_b_dialog)

        after_merge_total_utterance_cnt += len(user_a_dialog.sentences)
        after_merge_total_utterance_cnt += len(user_b_dialog.sentences)

        if num_turn_a > 0:
            user_dialog_list.append(user_a_dialog)
        else:
            print('no utterance for caller A in trans:', trans.conversation_no)
        if num_turn_b > 0:
            user_dialog_list.append(user_b_dialog)
        else:
            print('no utterance for caller B in trans:', trans.conversation_no)

        merge_dialog = merge_two_party_user_dialog(user_a_dialog, user_b_dialog)

        if merge_dialog.num_turn < 1:
            print('num_turn=0 for, ', merge_dialog.session_id, file=sys.stderr)
        else:
            two_party_dialog_list.append(merge_dialog)

    print('num of empty utterance: ', empty_utterance_cnt, file=sys.stderr)
    print('num of total utterance: ', total_utterance_cnt, file=sys.stderr)
    print('num of after merge total utterance: ',
          after_merge_total_utterance_cnt, file=sys.stderr)

    return user_dialog_list, two_party_dialog_list


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
    cmdline_parser.add_argument('--swda_basedir',
                                help='SWDA data directory')
    cmdline_parser.add_argument('--train_dialog_list',
                                help='SWDA train dialog list')
    cmdline_parser.add_argument('--test_dialog_list',
                                help='SWDA test dialog list')
    cmdline_parser.add_argument('--out_basedir',
                                help='output base directory')
    cmdline_parser.add_argument('--lowercase',
                                type=int,
                                default=1,
                                help='lowercase tokens')
    cmdline_parser.add_argument('--skip_punct',
                                type=int,
                                default=1,
                                help='skips punctuation')
    cmdline_parser.add_argument('--skip_mark_nonverbal',
                                type=int,
                                default=0,
                                help='removes nonverbal markups')

    cmdline_parser.set_defaults(**defaults)
    args = cmdline_parser.parse_args(remaining_argv)
    print(16 * '=')
    print('Processing config:')
    for arg in vars(args):
        print('{0}: {1}'.format(arg, getattr(args, arg)))
    print(16 * '=')

    if not args.swda_basedir:
        raise ValueError('Please specify swda_basedir!')

    if not os.path.exists(args.out_basedir):
        os.mkdir(args.out_basedir)

    # Extracts user dialogs from raw file.
    user_dialog_list, two_party_dialog_list = extract_caller_dialog(
        args.swda_basedir, lowercase=bool(args.lowercase),
        skip_punct=bool(args.skip_punct),
        skip_mark_nonverbal=bool(args.skip_mark_nonverbal)
    )

    # Reads in train/test split from file.
    with open(args.test_dialog_list) as fin:
        test_dialog_id_list = set([
            'session_{0}'.format(line.strip()[2:])
            for line in fin
        ])
    num_test_dialog = len(test_dialog_id_list)

    with open(args.train_dialog_list) as fin:
        train_dialog_id_list = set([
            'session_{0}'.format(line.strip()[2:])
            for line in fin
        ])
    num_train_dialog = len(train_dialog_id_list)

    def split_dialog(full_dialog_list, train_dialog_ids, test_dialog_ids,
                     out_dir):
        """Splits dialog into train/dev/test splits."""
        split2dialogs = {
            'train': [], 'dev': [], 'test': []
        }

        split2dialog_cnts = {
            'train': 0, 'dev': 0, 'test': 0
        }

        split2dialog_utt_cnts = {
            'train': 0, 'dev': 0, 'test': 0
        }

        for dialog in full_dialog_list:
            if not dialog.sentences:
                raise ValueError('No sentence for dialog: {0}'.format(
                    dialog.session_id
                ))

            if dialog.session_id in test_dialog_ids:
                split2dialog_utt_cnts['test'] += len(dialog.sentences)
                split2dialogs['test'].append(dialog)
            elif dialog.session_id in train_dialog_ids:
                split2dialog_utt_cnts['train'] += len(dialog.sentences)
                split2dialogs['train'].append(dialog)
            else:
                split2dialog_utt_cnts['dev'] += len(dialog.sentences)
                split2dialogs['dev'].append(dialog)

        for split in ['train', 'test', 'dev']:
            split2dialog_cnts[split] = len(split2dialogs[split])
            filename = os.path.join(out_dir,
                                    'swda_{0}.jsonline'.format(split))
            write_jsonable_object_to_file(split2dialogs[split],
                                          filename)
            print('There are {0} utterances and {1} dialogs in {2}'.format(
                split2dialog_utt_cnts[split], split2dialog_cnts[split], split
            ))

        return split2dialog_cnts

    user_dialog_dir = os.path.join(args.out_basedir, 'swda_user_dialog_dir')
    if not os.path.exists(user_dialog_dir):
        os.mkdir(user_dialog_dir)

    user_dialog_cnts = split_dialog(user_dialog_list, train_dialog_id_list,
                                    test_dialog_id_list, user_dialog_dir)

    assert (user_dialog_cnts['train'] == 2 * num_train_dialog)
    assert (user_dialog_cnts['test'] == 2 * num_test_dialog)

    predictor_dialog_dir = os.path.join(
        args.out_basedir, 'swda_predictor_dialog_dir'
    )
    if not os.path.exists(predictor_dialog_dir):
        os.mkdir(predictor_dialog_dir)

    predictor_dialog_cnts = split_dialog(
        two_party_dialog_list, train_dialog_id_list, test_dialog_id_list,
        predictor_dialog_dir
    )
    assert predictor_dialog_cnts['train'] == num_train_dialog
    assert predictor_dialog_cnts['test'] == num_test_dialog

    exit(0)


if __name__ == '__main__':
    main()
