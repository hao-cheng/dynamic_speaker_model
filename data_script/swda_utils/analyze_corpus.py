#!/usr/bin/env python2
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import re
import codecs
from collections import defaultdict
from collections import Counter

import pandas as pd

from .swda import CorpusReader


def extract_utterances_by_damsl_act_tag(swda_basedir):
    """Extracts SWDA utterances by DAMSL act tag."""
    all_utterances = defaultdict(list)
    corpus = CorpusReader(swda_basedir)
    for trans in corpus.iter_transcripts(display_progress=False):
        for utt in trans.utterances:
            tokens = utt.pos_lemmas(wn_lemmatize=False)
            words = []
            for token in tokens:
                # skip punctuation by checking the POS tag.
                if not re.match(r'^[a-zA-Z]', token[1]):
                    continue
                words.append(token[0].lower())
            if not words:
                # ignore empty utterance
                continue
            utterance = ' '.join(words)
            all_utterances[utt.damsl_act_tag()].append(utterance)

    utterance_counters = {}
    for damsl_act_tag, utterances in all_utterances.iteritems():
        utterance_counters[damsl_act_tag] = Counter(utterances)

    return utterance_counters


def save_utterances_by_damsl_act_tag(utterances_counters,
                                     outdir):
    """Saves utterances by DAMSL act tag."""

    stat_rows = []
    for damsl_act_tag, counter in utterances_counters.iteritems():
        stat_rows.append({
            'DamslActTag': damsl_act_tag,
            'NumUtterances': len(counter),
            'NumOccurrences': sum(counter.values())
        })

        fp = codecs.open(
            os.path.join(outdir,
                         '{}.utterances.txt'.format(damsl_act_tag)),
            'w',
            encoding='utf-8'
        )
        for utterance, cnt in counter.most_common():
            fp.write(str(cnt))
            fp.write('\t')
            fp.write(utterance)
            fp.write('\n')
        fp.close()

    stat_df = pd.DataFrame(stat_rows)
    stat_df.sort_values(
        'NumOccurrences',
        ascending=False,
        inplace=True
    )

    return stat_df


def extract_utterances_by_raw_act_tag(swda_basedir):
    """Extracts SWDA utterances by raw act tag."""
    all_utterances = defaultdict(list)
    corpus = CorpusReader(swda_basedir)
    for trans in corpus.iter_transcripts(display_progress=False):
        for utt in trans.utterances:
            tokens = utt.pos_lemmas(wn_lemmatize=False)
            words = []
            for token in tokens:
                # skip punctuation by checking the POS tag.
                if not re.match(r'^[a-zA-Z]', token[1]):
                    continue
                words.append(token[0].lower())
            if not words:
                # ignore empty utterance
                continue
            utterance = ' '.join(words)
            all_utterances[utt.act_tag].append(utterance)

    utterance_counters = {}
    for act_tag, utterances in all_utterances.iteritems():
        utterance_counters[act_tag] = Counter(utterances)

    return utterance_counters


def save_utterances_by_raw_act_tag(utterances_counters,
                                   outdir):
    """Saves utterances by raw act tag."""

    stat_rows = []
    for act_tag, counter in utterances_counters.iteritems():
        stat_rows.append({
            'ActTag': act_tag,
            'NumUtterances': len(counter),
            'NumOccurrences': sum(counter.values())
        })

        fp = codecs.open(
            os.path.join(outdir,
                         '{}.utterances.txt'.format(act_tag)),
            'w',
            encoding='utf-8'
        )
        for utterance, cnt in counter.most_common():
            fp.write(str(cnt))
            fp.write('\t')
            fp.write(utterance)
            fp.write('\n')
        fp.close()

    stat_df = pd.DataFrame(stat_rows)
    stat_df.sort_values(
        'NumOccurrences',
        ascending=False,
        inplace=True
    )

    return stat_df


def extract_conversations(swda_basedir,
                          conversation_txt_dir):
    corpus = CorpusReader(swda_basedir)
    for trans in corpus.iter_transcripts(display_progress=False):
        if not trans.utterances:
            continue

        utterances = []
        curr_utterance = []
        curr_caller = trans.utterances[0].caller
        for utt in trans.utterances:
            tokens = utt.pos_lemmas(wn_lemmatize=False)
            if not tokens:
                continue

            if utt.caller != curr_caller:
                utterances.append(curr_utterance)
                curr_caller = utt.caller
                curr_utterance = []

            for token in tokens:
                curr_utterance.append(token[0].lower())
        if curr_utterance:
            utterances.append(curr_utterance)

        if not utterances:
            continue

        conversation_txt = os.path.join(
            conversation_txt_dir,
            '{}.txt'.format(trans.conversation_no)
        )
        assert not os.path.exists(conversation_txt)
        fp = codecs.open(
            conversation_txt,
            'w',
            encoding='utf-8'
        )
        for utterance in utterances[1:]:
            assert utterance
            fp.write('> ')
            fp.write(' '.join(utterance))
            fp.write('\n')
        fp.close()
