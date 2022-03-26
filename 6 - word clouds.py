"""

6 word clouds.py

This script generates word clouds which are based off of the frequency of words that are correlated
with approval or eventual rejection

https://towardsdatascience.com/simple-wordcloud-in-python-2ae54a9f58e5

"""

# +
import string
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
from wordcloud import WordCloud
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
plt.rcParams['figure.dpi'] = 300 ## Increase DPI of plot

stop_words = set(stopwords.words('english'))
stop_words.add("\'s'")
punctuation = set(string.punctuation)
lemmatizer = WordNetLemmatizer()
vectorizer = CountVectorizer()


## Read in raw data with scores ##
raw_data = pd.read_pickle('raw_data_with_sentiment_protocol_4.pk1')


# -

## Clean and vectorize Sentences ##
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

# +


sentence_column = raw_data['clean_sentences']
flattened_sentences = [' '.join(sentence) for sentence in sentence_column]
clean_sentence_column = list(map(cleanFlatSentence, flattened_sentences))
feature_matrix = vectorizer.fit_transform(clean_sentence_column)
count_df = pd.DataFrame(feature_matrix.toarray(), columns = vectorizer.get_feature_names())

count_df = count_df.rename(columns=lambda x: re.sub(u"\uFB00",'ff',x))
# -

## Split data
data = pd.concat([raw_data['label'].rename("LABEL"), count_df], axis = 1)
#approved_array = data['label'] == 1
approved = data[data['LABEL'] == 1].drop('LABEL', axis = 1)
failed = data[data['LABEL'] == 0].drop('LABEL', axis = 1)

freq_approved = approved.sum().nlargest(n = 100, keep='first')
freq_failed = failed.sum().nlargest(n = 100, keep='first')


# +

def random_blue(word=None, font_size=None, position=None,  orientation=None, font_path=None, random_state=None):
    h = 230#int(360.0 * 21.0 / 255.0)
    s = int(100.0 * 255.0 / 255.0)
    l = int(100.0 * float(random_state.randint(60, 120)) / 255.0)
    return "hsl({}, {}%, {}%)".format(h, s, l)

def random_orange(word=None, font_size=None, position=None,  orientation=None, font_path=None, random_state=None):
    h = 0#int(360.0 * 21.0 / 255.0)
    s = int(100.0 * 255.0 / 255.0)
    l = int(100.0 * float(random_state.randint(60, 120)) / 255.0)
    return "hsl({}, {}%, {}%)".format(h, s, l)

wordcloud_approved = WordCloud(width = 1200, height = 800,                
                       background_color ='white', max_words=1000,  
                        color_func=random_blue,
                       min_font_size = 10).generate_from_frequencies(freq_approved) 
wordcloud_failed = WordCloud(width = 1200, height = 800,                
                       background_color ='white', max_words=1000,   
                       color_func = random_orange,
                       min_font_size = 10).generate_from_frequencies(freq_failed) 
# -

plt.figure(figsize = (12, 8))
plt.imshow(wordcloud_approved) 
plt.axis("off") 
plt.show() 

plt.figure(figsize = (12, 8))
plt.imshow(wordcloud_failed) 
plt.axis("off") 
plt.show() 

# +
## --------------------------------------------------------------------------------------- ##
