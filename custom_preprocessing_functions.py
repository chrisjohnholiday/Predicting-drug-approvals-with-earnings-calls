# -*- coding: utf-8 -*-
"""
Functions to process raw pdfs for further analysis 

------------------------------------------------------------------------------
extract_relevant_pdf_text:

    inputs:
        drug names (list)
        path to file location (string)
    
    output:
        list of sentences i) containing one of the drug names ii) following 2 sentences

Reads a pdf from a file location. Splits into senences. Finds the occruances of
drug names and extracts those + following 2 sentences. Returns a list of the raw
sentences.
------------------------------------------------------------------------------
(not yet written)
process_text_BOW:
    
    inputs:
        raw text as a single variable (string)
        
    output:
        vector of words?
        vector of word frequencies?

Processes sentences from extract_relevant_pdf_text() for bag-of-word (BOW) analyses
Performs standard pre-processing steps to get words and word frequences. 

Spelling correction, removing stopwords, case-correction, 

"""

import pdfminer.high_level
from nltk.tokenize import word_tokenize, sent_tokenize


def extract_relevant_pdf_text(pdf_location, drug_names=[]):
    
    # read file into large string
    text = pdfminer.high_level.extract_text(pdf_location)
    # parse string into sentences
    sentence_tokens = sent_tokenize(text)
    
    
    # get the indeces of sentences containing one of the drug names
    sentence_indices = []  
    # loop though all drug names ...
    for name in drug_names:
        # ... get the 'index' of each 'sentence' where the drug 'name' is in the 'sentence' (use lower case)
        sentence_index = [index for (index, sentence) in enumerate(sentence_tokens) if name.lower() in sentence.lower()]
        # ... add to list of indices
        sentence_indices += sentence_index
        
    # Not yet implemented: get the indeces of sentences containing the relevant indication
    # same concept as the loop above. 
    
        
    # 1. build list of the actual sentences
    extracted_sentences = []
    # extract sentences + the following 2 sentences by index values
    for index in sentence_indices:
        
        # create indeces to extract 
        start = index   # index of the sentence containing drug name
        end = index + 3 # index for following 2 senences
        
        # make sure the sentences are not past the end of the document 
        if end > len(sentence_tokens): end = len(sentence_tokens)
        
        # add sentences to list
        extracted_sentences += sentence_tokens[start:end]
    
    # 2. put all the sentences into a single variable
    extracted_text = ''
    for sentence in extracted_sentences: 
        extracted_text += " " + sentence
    
    # return the list of sentences and the combined text
    return extracted_sentences, extracted_text
 
  
'''
# test code
pdf_loc = "Transcripts/Q1 2019 ALDR.pdf"
drug_names = ['eptinezumab']
sentences, text = extract_relevant_pdf_text(pdf_loc, drug_names)
''' 
        