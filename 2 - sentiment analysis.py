# -*- coding: utf-8 -*-
"""
Calculate per-sentence sentiment scores using 3 lexicons.

"""

import pandas as pd

# load raw data from pickled df
raw_data = pd.read_pickle('raw_data_5_year.pk1')
# version without sentences columns for easier loading/viewing
raw_data_small = raw_data.drop(['sentences', 'clean_sentences'], axis=1)



# Afinn - compute sentiment scores -------------------------------------------
# get total score and average score per sentence

# initialize afinn sentiment analyzer object
from afinn import Afinn
# create analyzer object
af = Afinn()

# sample score
af.score('this is a fun test sentence')


# store scores in a list
raw_afinn_scores = []

# loop through every set of extracted sentences
for sentences_extract in raw_data.loc[:,'clean_sentences']:
    
    # get the total score of each sentence
    sentence_scores = [af.score(sentence) for sentence in sentences_extract]
    
    # sum scores from each sentence
    total_score = sum(sentence_scores)
    
    # save total score
    raw_afinn_scores.append(total_score)


# assign to dataframe
raw_data['total_afinn_score'] = raw_afinn_scores

# divide by measure of total text to get sentiment per sentence
raw_data['per_sentence_afinn_score'] = raw_data['total_afinn_score'] / raw_data['sentence_count_est']
    



# TextBlob --------------------------------------------------------------------
# get total score and average score per sentence

from textblob import TextBlob

# sample score
TextBlob('this is a fun test sentence').sentiment
# Sentiment(polarity=0.3, subjectivity=0.2)
# returns both polarity and subjectivity. Use polarity.
# obj = TextBlob('this is a fun test sentence')
# object has lots of data and methods, including built in pre-processing: obj.stripped

# store scores in a list
raw_TB_scores = []

# loop through every set of extracted sentences
for sentences_extract in raw_data.loc[:,'clean_sentences']:
    
    # get the total score of each sentence
    sentence_scores = [TextBlob(sentence).sentiment.polarity for sentence in sentences_extract]
    
    # sum scores from each sentence
    total_score = sum(sentence_scores)
    
    # save total score
    raw_TB_scores.append(total_score)


# assign to dataframe
raw_data['total_TextBlob_score'] = raw_TB_scores

# divide by measure of total text to get sentiment per sentence
raw_data['per_sentence_TextBlob_score'] = raw_data['total_TextBlob_score'] / raw_data['sentence_count_est']



# VADER ----------------------------------------------------------------------
# focused on social media language so may not work well.

from nltk.sentiment.vader import SentimentIntensityAnalyzer
#import nltk
#nltk.download('vader_lexicon') # download dictionary when using for the first time

# create analyzer object
vader = SentimentIntensityAnalyzer()

# sample score
vader.polarity_scores('this is a fun test sentence')
# {'neg': 0.0, 'neu': 0.395, 'pos': 0.605, 'compound': 0.5574}
# returns a negative, neurtal and positive score. Compound normalizes and combines
# the three scores. Use compound for our analysis.

# store scores in a list
raw_vader_scores = []

# loop through every set of extracted sentences
for sentences_extract in raw_data.loc[:,'clean_sentences']:
    
    # get the total score of each sentence
    sentence_scores = [vader.polarity_scores(sentence)['compound'] for sentence in sentences_extract]
    
    # sum scores from each sentence
    total_score = sum(sentence_scores)
    
    # save total score
    raw_vader_scores.append(total_score)


# assign to dataframe
raw_data['total_VADER_score'] = raw_vader_scores

# divide by measure of total text to get sentiment per sentence
raw_data['per_sentence_VADER_score'] = raw_data['total_VADER_score'] / raw_data['sentence_count_est']



'''
# SentiWordNet ----------------------------------------------------------------

from nltk.corpus import sentiwordnet as swn
#import nltk
#nltk.download('sentiwordnet') # download dictionary when using for the first time
#nltk.download('wordnet') # download dictionary when using for the first time

# scores words according to part of speech (noun, verb, adjective, etc) and 
# usage (01, 02, 03... higher = more common). 

# must first perform POS tagging of sentences. 

# sample score
swn.senti_synset('breakdown.n.03')
'''






















'''

# testing scores on different sentences --------------------------------------
test_sentences_extract = raw_data.loc[74,'clean_sentences']

for i in range(30):
    print(i)
    print(test_sentences_extract[i])
    print(af.score(test_sentences_extract[i]))


# score the sentences in the table:
for sentence in [(26,25), (1,9), (1,19), (2,1), (2,12), (2,28), (31,6), (74,9)]:    
    
    txt = raw_data.loc[sentence[0],'clean_sentences'][sentence[1]]
    
    print(txt)
    print(af.score(txt))
    print(TextBlob(txt).sentiment.polarity)
    print(TextBlob(txt).sentiment.subjectivity)
    print(vader.polarity_scores(txt)['compound'])




# check correlation among lexions
cor = raw_data_small.loc[:,['per_sentence_TextBlob_score', 
                            'per_sentence_afinn_score',
                            'per_sentence_VADER_score']].corr()



'''




# Save clean file for further analysis ----------------------------------------

# get clean dataframe
data_with_sentiment = raw_data.loc[:, ['drug_index', 'drug_name', 'key_date',
                                       'clean_sentences', 'character_count',
                                       'sentence_count_est',
                                       'total_afinn_score', 'per_sentence_afinn_score',
                                       'total_TextBlob_score', 'per_sentence_TextBlob_score',
                                       'total_VADER_score', 'per_sentence_VADER_score',
                                       'label']]

# save dataframe
data_with_sentiment.to_pickle('raw_data_with_sentiment_5_year_protocol_4.pk1', protocol=4) 

# to load the pickled df in later analysis
#import pandas as pd
#unpickled_data = pd.read_pickle('raw_data_with_sentiment.pk1')




