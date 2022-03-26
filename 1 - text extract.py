# -*- coding: utf-8 -*-
"""
Read through pdfs of earnings call transcripts and extract sentences surrounding the mention of a drug.

"""
import pandas as pd
import os                                    # for file/directory reading
import custom_preprocessing_functions as cpf # custom function to read pd fs and extract text

# location of transcripts
# files are large and numerous so stored locally
transcripts_folder = 'W:\BB transcripts'

# location of drug list (table with all relevant drug info)
drug_list_loc = 'compounds_in_pipeline - Copy.csv'


# ----------------------------------------------------------------------------
# Load and process drug list

# get master list of drugs with company and dates
drug_list = pd.read_csv(drug_list_loc)

# remove drugs whose companies were not public / had no trancripts
drug_list = drug_list.loc[drug_list['Public_and_transcripts']==1,:]

# convert relevant date fields to date-time format and make nan=0
drug_list['Approval Date'] = pd.to_datetime(drug_list['Approval Date'])
drug_list['Terminated Date'] = pd.to_datetime(drug_list['Terminated Date'])

# combine approval and terminated dates in one column 
drug_list['key_event_date'] = 0
drug_list.loc[pd.notna(drug_list['Approval Date']), 'key_event_date'] = drug_list['Approval Date']
drug_list.loc[pd.notna(drug_list['Terminated Date']), 'key_event_date'] = drug_list['Terminated Date']

# list of drugs for which we want text
drug_indices = list(drug_list.index)
 # drug_indices = [26,26,26] # short list for testing purposes

# label = 1 if approval date exists, otherwise label = 0 
drug_labels = [int(not pd.isna(approval_date)) for approval_date in drug_list.loc[drug_indices, 'Approval Date']]


# ----------------------------------------------------------------------------
# Create function to extract text in parallel

# enclose text extraction in a function for parallelization
# use a single drug index as the input instead of drug_indicies 
def extract_text_parallel(drug_index, drug_list, transcripts_folder):
   
    # get drug names
    drug_names = drug_list.loc[drug_index, ['Drug', 'Drug Name 2', 'Drug Name 3', 'Drug Name 4']]
    drug_names = [name for name in drug_names if not pd.isna(name)] # convert to basic list and drop blank names
    
    # get company names
    company_names = drug_list.loc[drug_index, ['Ticker', 'Ticker 2']] 
    company_names = [name for name in company_names if not pd.isna(name)] # convert to basic list and drop blank names
    
    # build corpus of text and sentences for a given drug
    ##all_text = '' | currently just getting sentences, not combined text.
    all_sentences = []
    
    n_transcripts = 0 # record the number of transcripts read through for each drug
    
    # loop through all company names
    for company in company_names:
    
        # folder with transcripts for given company
        transcript_folder_company = transcripts_folder + "\\" + company
        # get file name of all transcripts in the transcript folder
        transcript_file_names = os.listdir(transcript_folder_company)

        # relevant dates. get in YYYYMMDD format to match transcript file names
        # key_date = 20210101 # this date gets all transcripts up to Jan 2021 instead
        key_date = drug_list.loc[drug_index, 'key_event_date'].strftime('%Y%m%d')
        
        ######################
        x=3
        #key_date = key_date[0:2] + str(int(key_date[2:4]) - x) + key_date[4:]  # transcripts up to x number of years before key date
        key_date = str(int(key_date[0:4]) - x) + key_date[4:]  # transcripts up to x number of years before key date

        ######################
        
        # get all transcripts up to the phase_2_start
        transcript_file_names = [file for file in transcript_file_names if int(file[0:8]) <= int(key_date)]
                                    # for file in the transcript folder, if transcript date is before phase 2, list file name
        
        n_transcripts += len(transcript_file_names) # record the number of transcripts 
        
        
        # loop though all the transcripts to collect relevant sentences
        for transcript in transcript_file_names:
            
            # get location of transcript file
            transcript_loc = transcript_folder_company + "\\" + transcript
            
            # extract relevant sentences using custom function
            sentences, text = cpf.extract_relevant_pdf_text(transcript_loc, drug_names)
            
            # add to master list of sentences and combined text
            all_sentences += sentences
            ##all_text += text | ignore combined text for now
            
                  
    # count the number of sentences extracted for each drug
    # note - this is 3x the number of sentences mentioning the drug if extracing
    # the following 2
    n_sentences = len(all_sentences)    
     
    # for parallelization, each "row" (drug - label - text) is computed separately
    # want the function to return data in a convenient row format
    result_row = [drug_index, all_sentences, n_sentences, n_transcripts]
    return result_row

# test for drug with small number of transcripts:
# row = extract_text_parallel(26, drug_list)


# -----------------------------------------------------------------------------
# Set up multiprocessing
# https://medium.com/@mjschillawski/quick-and-easy-parallelization-in-python-32cb9027e490

import multiprocessing
from joblib import Parallel, delayed
from tqdm import tqdm # progress tracker 

num_cores = multiprocessing.cpu_count()
# inputs = drug_indices # will iterate over one input. One element per call.
inputs = tqdm(drug_indices) # wrap with object to keep track of progress

# key loop for the function call
results = Parallel(n_jobs=num_cores) (delayed(extract_text_parallel)(i, drug_list, transcripts_folder) for i in inputs)
# processed_list should contain one element [drug_index, sentences] for each drug index

# note: no longer need to protect the main loop for joblib >0.12.
# search "protect" on: https://joblib.readthedocs.io/en/latest/parallel.html

# -----------------------------------------------------------------------------
# Save results
result_table = pd.DataFrame(results, columns=['drug_index', 'sentences', 
                                              'number_of_sentences', 'number_of_transcripts'])

# labels and other relevant fields
result_table['label'] = drug_labels
result_table['drug_name'] = drug_list.loc[drug_indices,'Drug'].tolist()
result_table['key_date']  = drug_list.loc[drug_indices,'key_event_date'].tolist()

# .csv and .xlsx can't handle the large amount of text in a single cell
#result_table.to_csv('raw_data_v1.csv', index=False)
#result_table.to_excel('raw_data_v1.xlsx', index=False)

# pickle the dataframe directly
#result_table.to_pickle('raw_data_3_year.pk1') 

# to load the pickled df in later analysis
#unpickled_data = pd.read_pickle('raw_data.pk1')
