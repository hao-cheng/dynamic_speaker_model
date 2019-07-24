"""Treebank Word Detokenizer
Copied from NLTK-3.3 nltk/tokenize/treebank.py
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import re

from nltk.tokenize.api import TokenizerI


class TreebankWordDetokenizer(TokenizerI):
    """
    The Treebank detokenizer uses the reverse regex operations corresponding to
    the Treebank tokenizer's regexes.
    Note:
    - There're additional assumption mades when undoing the padding of [;@#$%&]
      punctuation symbols that isn't presupposed in the TreebankTokenizer.
    - There're additional regexes added in reversing the parentheses tokenization,
       - the r'([\]\)\}\>])\s([:;,.])' removes the additional right padding added
         to the closing parentheses precedding [:;,.].
    - It's not possible to return the original whitespaces as they were because
      there wasn't explicit records of where '\n', '\t' or '\s' were removed at
      the text.split() operation.
        >>> from nltk.tokenize.treebank import TreebankWordTokenizer, TreebankWordDetokenizer
        >>> s = '''Good muffins cost $3.88\\nin New York.  Please buy me\\ntwo of them.\\nThanks.'''
        >>> d = TreebankWordDetokenizer()
        >>> t = TreebankWordTokenizer()
        >>> toks = t.tokenize(s)
        >>> d.detokenize(toks)
        'Good muffins cost $3.88 in New York. Please buy me two of them. Thanks.'
    The MXPOST parentheses substitution can be undone using the `convert_parentheses`
    parameter:
    >>> s = '''Good muffins cost $3.88\\nin New (York).  Please (buy) me\\ntwo of them.\\n(Thanks).'''
    >>> expected_tokens = ['Good', 'muffins', 'cost', '$', '3.88', 'in',
    ... 'New', '-LRB-', 'York', '-RRB-', '.', 'Please', '-LRB-', 'buy',
    ... '-RRB-', 'me', 'two', 'of', 'them.', '-LRB-', 'Thanks', '-RRB-', '.']
    >>> expected_tokens == t.tokenize(s, convert_parentheses=True)
    True
    >>> expected_detoken = 'Good muffins cost $3.88 in New (York). Please (buy) me two of them. (Thanks).'
    >>> expected_detoken == d.detokenize(t.tokenize(s, convert_parentheses=True), convert_parentheses=True)
    True
    During tokenization it's safe to add more spaces but during detokenization,
    simply undoing the padding doesn't really help.
    - During tokenization, left and right pad is added to [!?], when
      detokenizing, only left shift the [!?] is needed.
      Thus (re.compile(r'\s([?!])'), r'\g<1>')
    - During tokenization [:,] are left and right padded but when detokenizing,
      only left shift is necessary and we keep right pad after comma/colon
      if the string after is a non-digit.
      Thus (re.compile(r'\s([:,])\s([^\d])'), r'\1 \2')
    >>> from nltk.tokenize.treebank import TreebankWordDetokenizer
    >>> toks = ['hello', ',', 'i', 'ca', "n't", 'feel', 'my', 'feet', '!', 'Help', '!', '!']
    >>> twd = TreebankWordDetokenizer()
    >>> twd.detokenize(toks)
    "hello, i can't feel my feet! Help!!"
    >>> toks = ['hello', ',', 'i', "can't", 'feel', ';', 'my', 'feet', '!',
    ... 'Help', '!', '!', 'He', 'said', ':', 'Help', ',', 'help', '?', '!']
    >>> twd.detokenize(toks)
    "hello, i can't feel; my feet! Help!! He said: Help, help?!"
    """
    _contractions = MacIntyreContractions()
    CONTRACTIONS2 = [re.compile(pattern.replace('(?#X)', '\s'))
                     for pattern in _contractions.CONTRACTIONS2]
    CONTRACTIONS3 = [re.compile(pattern.replace('(?#X)', '\s'))
                     for pattern in _contractions.CONTRACTIONS3]

    # ending quotes
    ENDING_QUOTES = [
        (re.compile(r"([^' ])\s('ll|'LL|'re|'RE|'ve|'VE|n't|N'T) "), r"\1\2 "),
        (re.compile(r"([^' ])\s('[sS]|'[mM]|'[dD]|') "), r"\1\2 "),
        (re.compile(r'(\S)(\'\')'), r'\1\2 '),
        (re.compile(r" '' "), '"')
    ]

    # Handles double dashes
    DOUBLE_DASHES = (re.compile(r' -- '), r'--')

    # Optionally: Convert parentheses, brackets and converts them from PTB symbols.
    CONVERT_PARENTHESES = [
        (re.compile('-LRB-'), '('), (re.compile('-RRB-'), ')'),
        (re.compile('-LSB-'), '['), (re.compile('-RSB-'), ']'),
        (re.compile('-LCB-'), '{'), (re.compile('-RCB-'), '}')
    ]

    # Undo padding on parentheses.
    PARENS_BRACKETS = [(re.compile(r'\s([\[\(\{\<])\s'), r' \g<1>'),
                       (re.compile(r'\s([\]\)\}\>])\s'), r'\g<1> '),
                       (re.compile(r'([\]\)\}\>])\s([:;,.])'), r'\1\2')]

    # punctuation
    PUNCTUATION = [
        (re.compile(r"([^'])\s'\s"), r"\1' "),
        (re.compile(r'\s([?!])'), r'\g<1>'),  # Strip left pad for [?!]
        # (re.compile(r'\s([?!])\s'), r'\g<1>'),
        (re.compile(r'([^\.])\s(\.)([\]\)}>"\']*)\s*$'), r'\1\2\3'),
        # When tokenizing, [;@#$%&] are padded with whitespace regardless of
        # whether there are spaces before or after them.
        # But during detokenization, we need to distinguish between left/right
        # pad, so we split this up.
        (re.compile(r'\s([#$])\s'), r' \g<1>'),  # Left pad.
        (re.compile(r'\s([;%])\s'), r'\g<1> '),  # Right pad.
        (re.compile(r'\s([&])\s'), r' \g<1> '),  # Unknown pad.
        (re.compile(r'\s\.\.\.\s'), r'...'),
        (re.compile(r'\s([:,])\s$'), r'\1'),
        (re.compile(r'\s([:,])\s([^\d])'), r'\1 \2')  # Keep right pad after comma/colon before non-digits.
        # (re.compile(r'\s([:,])\s([^\d])'), r'\1\2')
    ]

    # starting quotes
    STARTING_QUOTES = [
        (re.compile(r'([ (\[{<])\s``'), r'\1"'),
        (re.compile(r'\s(``)\s'), r'\1'),
        (re.compile(r'^``'), r'\"'),
    ]

    def tokenize(self, tokens, convert_parentheses=False):
        """
        Python port of the Moses detokenizer.
        :param tokens: A list of strings, i.e. tokenized text.
        :type tokens: list(str)
        :return: str
        """
        text = ' '.join(tokens)
        # Reverse the contractions regexes.
        # Note: CONTRACTIONS4 are not used in tokenization.
        for regexp in self.CONTRACTIONS3:
            text = regexp.sub(r'\1\2', text)
        for regexp in self.CONTRACTIONS2:
            text = regexp.sub(r'\1\2', text)

        # Reverse the regexes applied for ending quotes.
        for regexp, substitution in self.ENDING_QUOTES:
            text = regexp.sub(substitution, text)

        # Undo the space padding.
        text = text.strip()

        # Reverse the padding on double dashes.
        regexp, substitution = self.DOUBLE_DASHES
        text = regexp.sub(substitution, text)

        if convert_parentheses:
            for regexp, substitution in self.CONVERT_PARENTHESES:
                text = regexp.sub(substitution, text)

        # Reverse the padding regexes applied for parenthesis/brackets.
        for regexp, substitution in self.PARENS_BRACKETS:
            text = regexp.sub(substitution, text)

        # Reverse the regexes applied for punctuations.
        for regexp, substitution in self.PUNCTUATION:
            text = regexp.sub(substitution, text)

        # Reverse the regexes applied for starting quotes.
        for regexp, substitution in self.STARTING_QUOTES:
            text = regexp.sub(substitution, text)

        return text.strip()

    def detokenize(self, tokens, convert_parentheses=False):
        """ Duck-typing the abstract *tokenize()*."""
        return self.tokenize(tokens, convert_parentheses)
