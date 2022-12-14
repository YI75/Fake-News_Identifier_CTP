import streamlit as st
import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.corpus import stopwords
from nltk import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
import re

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
stopwords = stopwords.words('english')

model = pickle.load(open('models/model.pkl', 'rb'))
vectorizer = pickle.load(open('models/vectorizer.pkl', 'rb'))

def remove_unwanted_char(a_string):    
    a_string = re.sub(r"(@\[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|^rt|http.+?", "", str(a_string))
    return a_string

def remove_punctuation(a_string):    
    a_string = re.sub(r'[^\w\s]','', str(a_string))
    return a_string

def make_lower(a_string):
    return a_string.lower()

def remove_stopwords(a_string):
    # Break the sentence down into a list of words
    words = word_tokenize(a_string)
    
    # Make a list to append valid words into
    valid_words = []
    
    # Loop through all the words
    for word in words:
        
        # Check if word is not in stopwords
        if word not in stopwords:
            
            # If word not in stopwords, append to our valid_words
            valid_words.append(word)

    # Join the list of words together into a string
    a_string = ' '.join(valid_words)

    return a_string

def text_pipeline(input_string):
    input_string = make_lower(input_string)
    input_string = remove_punctuation(input_string)
    input_string = remove_stopwords(input_string)    
    return input_string

st.image('1.png')

st.title('Fake News Identifier')

sentence = st.text_input('Enter your news headline here:') 
st.button('Identify')

if sentence:

    sentence = text_pipeline(sentence)
    sentence = vectorizer.transform([sentence])
    prediction = model.predict(sentence)
    prediction_proba = model.predict_proba(sentence)
    prediction = prediction.replace(r'['\[\]\]','')
    st.header(prediction)