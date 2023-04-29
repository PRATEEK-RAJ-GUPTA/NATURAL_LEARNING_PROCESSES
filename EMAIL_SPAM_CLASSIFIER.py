# -*- coding: utf-8 -*-
"""
Created on Sat Apr 29 14:16:22 2023

@author: prate
"""

import pandas as pd

data=pd.read_csv('smsspamcollection/SMSSpamCollection',sep='\t',
                 names=['label','message'])

#DATA CLEANING AND PREPROCESSING

import re
import nltk
nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer

ps=PorterStemmer()
wordnet=WordNetLemmatizer()
corpus=[]

for i in range(0,len(data)):
    review=re.sub('[^a-zA-Z]',' ',data['message'][i])
    review=review.lower()
    review=review.split()
    review=[wordnet.lemmatize(word) for word in review if word not in stopwords.words('english')]
    review=' '.join(review)
    corpus.append(review)
    
    
#BAG OF WORDS

from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()
X=cv.fit_transform(corpus).toarray()

#DUMMIES VARABLES

#y=pd.get_dummies(data['label'])


    
