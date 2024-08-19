# -*- coding: utf-8 -*-
"""
Created on Thu Aug 15 09:59:04 2024

@author: Prateek.Mishra
"""

#################################### Module Import ###################################
import pandas as pd
import numpy as np
import re
import nltk 
from nltk.stem import wordnet                                  
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer    
from nltk import pos_tag                                     
from sklearn.metrics import pairwise_distances              
from nltk import word_tokenize                                
from nltk.corpus import stopwords   
nltk.download('stopwords')           
stop = stopwords.words('english')
nltk.download('averaged_perceptron_tagger')    
from spacy.lang.en.stop_words import STOP_WORDS as spacy_stopwords
import spacy
spacy_model = spacy.load('en_core_web_sm')
cv = CountVectorizer()                           

################################# Import FAQ ########################################
df = pd.read_csv("C:/Users/Prateek.Mishra/Downloads/Mental_Health_FAQ.csv")
df=df.drop(['Question_ID'], axis=1)

################################ Cleaning #####################################
contra_Expan_Dict = {"ain`t": "am not","aren`t": "are not","can`t": "cannot","can`t`ve": "cannot have","`cause": "because",
"could`ve": "could have","couldn`t": "could not","couldn`t`ve": "could not have","didn`t": "did not",
"doesn`t": "does not","don`t": "do not","hadn`t": "had not","hadn`t`ve": "had not have","hasn`t": "has not",
"haven`t": "have not","he`d": "he would","he`d`ve": "he would have","he`ll": "he will","he`ll`ve": "he will have",
"he`s": "he is","how`d": "how did","how`d`y": "how do you","how`ll": "how will",
"how`s": "how does","i`d": "i would","i`d`ve": "i would have","i`ll": "i will","i`ll`ve": "i will have","i`m": "i am",
"i`ve": "i have","isn`t": "is not","it`d": "it would","it`d`ve": "it would have","it`ll": "it will","it`ll`ve": "it will have",
"it`s": "it is","let`s": "let us","ma`am": "madam","mayn`t": "may not","might`ve": "might have","mightn`t": "might not",
"mightn`t`ve": "might not have","must`ve": "must have","mustn`t": "must not","mustn`t`ve": "must not have","needn`t": "need not","needn`t`ve": "need not have",
"o`clock": "of the clock","oughtn`t": "ought not","oughtn`t`ve": "ought not have","shan`t": "shall not",
"sha`n`t": "shall not","shan`t`ve": "shall not have","she`d": "she would",
"she`d`ve": "she would have","she`ll": "she will","she`ll`ve": "she will have",
"she`s": "she is","should`ve": "should have","shouldn`t": "should not","shouldn`t`ve": "should not have","so`ve": "so have","so`s": "so is",
"that`d": "that would","that`d`ve": "that would have","that`s": "that is","there`d": "there would","there`d`ve": "there would have","there`s": "there is",
"they`d": "they would","they`d`ve": "they would have","they`ll": "they will","they`ll`ve": "they will have","they`re": "they are","they`ve": "they have",
"to`ve": "to have","wasn`t": "was not"," u ": " you "," ur ": " your "," n ": " and ","won`t": "would not",
"dis": "this","bak": "back","brng": "bring"}
def expanded_form(x):
  if x in contra_Expan_Dict.keys():
    return(contra_Expan_Dict[x])
  else:
    return(x)

def clean_with_re(x):
  x=str(x)
  x=re.sub(r'[^ a-z]','',x)  
  x=re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'," ", x)
  x=re.sub(r'[^\w ]+', "", x)
  x=re.sub(r"[,!@&\'?\.$%_]"," ", x)
  x=re.sub(r"\d+"," ", x)
  return(x)

def pre_processing(x):
    x=str(x).lower()
#    input_data["text_col_clean"] = input_data["text_col_clean"].apply(lambda x:word_tokenize(str(x)))
    x=[expanded_form(t) for t in str(x).split()]
    x=[t for t in x if t not in spacy_stopwords ]
    x=[t for t in str(x).split() if t not in stop ]
    x= clean_with_re(x)
    x=" ".join([t.lemma_ for t in spacy_model(str(x))if t.lemma_ !="-PRON-" ])
#    input_data["text_col_clean"]=input_data["text_col_clean"].apply(lambda x: " ".join(x) )
    return x

def vector(df):
    X = cv.fit_transform(df['Questions']).toarray()
    features = cv.get_feature_names()
    df_vec = pd.DataFrame(X, columns = features)
    return df_vec
    
df["Questions"]=df["Questions"].apply(lambda x:pre_processing(x))
df_vec=vector(df)
def answer(text):
    lemma = pre_processing(text)
    vec = cv.transform([lemma]).toarray()
    cosine_value = 1- pairwise_distances(df_vec,vec, metric = 'cosine' )
    index_value = cosine_value.argmax()
    if index_value==0:
        return 'Sorry, I am not understand you'
    else:
        return df['Answers'].loc[index_value]

def chat(text):
    if type(text)==int:
        return 'Plese emter a valid question type'
    elif text.lower()== 'hello':
        return 'Hello'
    elif text.lower()== 'thank you':
        return 'Welcome'
    elif type(text)==str:
        return answer(text)

try:
    text=int(input())
except:
    text=input()    
    
ans=chat(text)
print(ans)

