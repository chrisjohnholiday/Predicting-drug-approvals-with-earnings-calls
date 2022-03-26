# +
"""
3 - create master dataset.py

Merging new drug features on drug data with sentiment score and text metrics


"""

## Managing dependencies ##
import pandas as pd
import string
import re
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
# nltk.download('stopwords')
# nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import preprocessing

## NLTK Settings ##
stop_words = set(stopwords.words('english'))
stop_words.add("\'s'")
punctuation = set(string.punctuation)
lemmatizer = WordNetLemmatizer()
port_stemmer = PorterStemmer()

## Count vectorize ##
vectorizer = TfidfVectorizer(min_df=0.0, max_df=1.0, ngram_range=(1, 1))

## Normalization ##
normalizer = preprocessing.MinMaxScaler()


# +

## Read in raw data with scores ##
raw_data = pd.read_pickle('raw_data_with_sentiment_protocol_4.pk1')
raw_data_1_year = pd.read_pickle('raw_data_with_sentiment_1_year_protocol_4.pk1')
raw_data_3_year = pd.read_pickle('raw_data_with_sentiment_3_year_protocol_4.pk1')
raw_data_5_year = pd.read_pickle('raw_data_with_sentiment_5_year_protocol_4.pk1')

# +
## Read in extra features data and rename ##
extra_features = pd.read_csv('compounds_in_pipeline_extra_features.csv')
extra_features = extra_features.rename(columns = {"Drug": "drug_name", "Company Market Cap": "market_cap",
                                 "Molecule Type": "molecule_type", "Orphan Designation": "orphan_fda",
                                "Fastrack Designation": "fastrack_fda", 
                                "Breakthrough Therapy": "breakthrough_fda",
                                "Priority Review": "priority_fda",
                                "Rare Pediatric Disease Designation": "rare_fda"})

## Fix ROA ##
ROA_df = extra_features.pop('ROA').str.get_dummies(';')
extra_features = pd.concat([extra_features, ROA_df], axis = 1)
## Fix market cap ##
extra_features["market_cap"] = extra_features["market_cap"].replace(',','', regex = True).astype('float64')
extra_features['market_cap'] = extra_features['market_cap'].fillna(np.nanmean(extra_features['market_cap'].to_numpy()))
## Fix molecule type ##
molecule_type_df = extra_features.pop('molecule_type').str.get_dummies()
extra_features = pd.concat([extra_features, molecule_type_df], axis = 1)


# -

## Flat sentence cleaner ##
## note that I used stemming in this case, we could use lemming but feature space will be larger ##
def cleanFlatSentence(sentence):
    split_words = sentence.lower().split() ## lower case and split ##
    split_words2 = [word.strip(string.punctuation) for word in split_words]
    ## Cleaning conditions ##
    ## 1. not in stopwords list ##
    ## 2. not in puncutation ##
    ## 3. Does not contain digits ##
    ## 4. larger than one character length ##
    clean_sentence = [word for word in split_words2 if word not in stop_words if word not in punctuation if not re.search(r'\d', word) if len(word) > 1]
    sentence_stemmed = [lemmatizer.lemmatize(word) for word in clean_sentence]
    return " ".join([word for word in sentence_stemmed])


def createMasterDF(data):
    
    ## create bag of words using tFid from sklearn ##
    sentence_column = data['clean_sentences']
    flattened_sentences = [' '.join(sentence) for sentence in sentence_column]
    clean_sentence_column = list(map(cleanFlatSentence, flattened_sentences))
    feature_matrix = vectorizer.fit_transform(clean_sentence_column)
    normalized_frequency_df = pd.DataFrame(feature_matrix.toarray(), columns = vectorizer.get_feature_names())
    data = pd.concat([data, normalized_frequency_df], axis = 1)
    
    ## Set Nan sentiment scores to average ##
    avg_afinn_sentence = np.nanmean(data['per_sentence_afinn_score'].to_numpy())
    data['per_sentence_afinn_score'] = data['per_sentence_afinn_score'].fillna(avg_afinn_sentence)
    avg_textblob_sentence = np.nanmean(data['per_sentence_TextBlob_score'].to_numpy())
    data['per_sentence_TextBlob_score'] = data['per_sentence_TextBlob_score'].fillna(avg_textblob_sentence)
    avg_vader_sentence = np.nanmean(data['per_sentence_VADER_score'].to_numpy())
    data['per_sentence_VADER_score'] = data['per_sentence_VADER_score'].fillna(avg_vader_sentence)
    
    ## Merge extra features
    data = data.merge(extra_features, how = 'left', on = 'drug_name')
    data = data.drop(['clean_sentences'], axis = 1)
    
    ## Normalize data ##
    label_df = data[['drug_index', 'drug_name', 'key_date']]
    feature_df = data.drop(['drug_index', 'drug_name', 'key_date'], axis = 1)
    feature_colnames = feature_df.columns
    feature_values = feature_df.values
    feature_values_normalized = min_max_scaler.fit_transform(feature_values)
    feature_df = pd.DataFrame(feature_values_normalized, columns = feature_colnames)
    master_data = pd.concat([label_df, feature_df], axis = 1)
    print(len(feature_colnames))
    return master_data


## Create master datasets ##
master_data_main = createMasterDF(raw_data)
master_data_1_year = createMasterDF(raw_data_1_year)
master_data_3_year = createMasterDF(raw_data_3_year)
master_data_5_year = createMasterDF(raw_data_5_year)


# +
## Export data ##
master_data_main.to_pickle("master_data_with_sentiment.pk1", protocol = 4)
master_data_1_year.to_pickle("master_data_with_sentiment_1_year.pk1", protocol = 4)
master_data_3_year.to_pickle("master_data_with_sentiment_3_year.pk1", protocol = 4)
master_data_5_year.to_pickle("master_data_with_sentiment_5_year.pk1", protocol = 4)

## ------------------------------------------------------------------------------------------##
