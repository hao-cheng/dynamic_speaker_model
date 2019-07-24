#!/usr/bin/env python

"""
Functions for using swda.py to explore the Switchboard Dialog Act Corpus.
"""

__author__ = "Christopher Potts"
__copyright__ = "Copyright 2011, Christopher Potts"
__credits__ = []
__license__ = "Creative Commons Attribution-NonCommercial-ShareAlike 3.0 Unported License: http://creativecommons.org/licenses/by-nc-sa/3.0/"
__version__ = "1.0"
__maintainer__ = "Christopher Potts"
__email__ = "See the author's website"
		
######################################################################

from collections import defaultdict
from operator import itemgetter
from swda import CorpusReader

######################################################################
 
def swda_education_region():
    """Create a count dictionary relating education and region."""    
    d = defaultdict(int)
    corpus = CorpusReader('swda')
    # Iterate throught the transcripts; display_progress=True tracks progress:
    for trans in corpus.iter_transcripts(display_progress=True):
        d[(trans.from_caller_education, trans.from_caller_dialect_area)] += 1
        d[(trans.to_caller_education, trans.to_caller_dialect_area)] += 1
    # Turn d into a list of tuples as d.items(), sort it based on the
    # second (index 1 member) of those tuples, largest first, and
    # print out the results:
    for key, val in sorted(d.items(), key=itemgetter(1), reverse=True):
        print key, val

# swda_education_region()

######################################################################

def tag_counts():
    """Gather and print counts of the tags."""
    d = defaultdict(int)
    corpus = CorpusReader('swda')
    # Loop, counting tags:
    for utt in corpus.iter_utterances(display_progress=True):
        d[utt.act_tag] += 1
    # Print the results sorted by count, largest to smallest:
    for key, val in sorted(d.items(), key=itemgetter(1), reverse=True):
        print key, val

# tag_counts()        

######################################################################

def count_matches():
    """Determine how many utterances have a single precisely matching tree."""
    d = defaultdict(int)
    corpus = CorpusReader('swda')
    for utt in corpus.iter_utterances():
        if len(utt.trees) == 1:
            if utt.tree_is_perfect_match():
                d['match'] += 1
            else: 
                d['mismatch'] += 1
    print "match: %s (%s percent)" % (d['match'], d['match']/float(sum(d.values())))

# count_matches()

######################################################################

def act_tags_and_rootlabels():
    """
    Create a CSV file named swda-actags-and-rootlabels.csv in
    which each utterance utt has its own row consisting of just

      utt.act_tag, utt.damsl_act_tag(), and utt.trees[0].node

    restricting attention to cases in which utt has a single,
    perfectly matching tree associated with it.
    """
    csvwriter = csv.writer(open('swda-actags-and-rootlabels.csv', 'w'))
    csvwriter.writerow(['ActTag', 'DamslActTag', 'RootNode'])
    corpus = CorpusReader('swda')    
    for utt in corpus.iter_utterances(display_progress=True):
        if utt.tree_is_perfect_match():
            csvwriter.writerow([utt.act_tag, utt.damsl_act_tag(), utt.trees[0].node])
