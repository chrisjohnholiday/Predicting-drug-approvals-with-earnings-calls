# -*- coding: utf-8 -*-
"""

Correct some of the erroneous sentence extracts.

A minority of pdfs were not tokenized correctly, causing very large sections
to be extracted as single sentences. 

This script drops those obsevations and fixes the sentence counts.


"""

import pandas as pd

# load raw data from pickled df
raw_data = pd.read_pickle('raw_data_5_year.pk1')

# -----------------------------------------------------------------------------
# Loop through all sentence extracts and remove erroneous (very large) extracts

# list of character counts from all sentences for a drug
character_counts = []
# list of cleaned sentences for all drugs
clean_sentences_all_drugs = []

# go through the extracted sentences for each drug
for sentences_extract in raw_data.loc[:,'sentences']:   
    
    # get count of characters
    char_count = 0
    # get new list of sentences
    clean_sentences = []
    
    for sentence in sentences_extract:
        
        n_chars = len(sentence)
        
        # remove the very large sentences that are really big document chunks
        if n_chars < 500:
            # add to list of good sentences
            clean_sentences.append(sentence)
            char_count += n_chars
            
    # add to list for all drugs
    character_counts.append(char_count)
    clean_sentences_all_drugs.append(clean_sentences)
            
    
# append to main dataset
raw_data['clean_sentences'] = clean_sentences_all_drugs
raw_data['character_count'] = character_counts
raw_data['sentence_count_est'] = raw_data['character_count'] / 150  # approximation to make more intuitive


# save updated dataframe to main raw dataset
raw_data.to_pickle('raw_data_5_year.pk1') 


