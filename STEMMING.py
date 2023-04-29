# -*- coding: utf-8 -*-
"""
Created on Sat Apr 29 12:02:57 2023

@author: prate
"""

import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords


#stopword==they them of from does not much value in different type of sentiment we are trying to be expressed


paragraph="""I have three visions for India. In 3000 years of our history people 
from all over the world have come and invaded us, captured our lands, conquered our minds. 
From Alexander onwards the Greeks, the Turks, the Moguls, the Portuguese, the British, the French, 
the Dutch, all of them came and looted us, took over what was ours. Yet we have not done this to any 
other nation.We have not conquered anyone. We have not grabbed their land, their culture and their
 history and tried to enforce our way of life on them. Why? Because we respect the freedom of others. 
 That is why my FIRST VISION is that of FREEDOM. I believe that India got its first vision of this in 1857, 
 when we started the war of Independence. It is this freedom that we must protect and nurture and build on. 
 If we are not free, no one will respect us. We have 10 percent growth rate in most areas. Our poverty levels are falling. 
 Our achievements are being globally recognised today. Yet we lack the self-confidence to see 
 ourselves as a developed nation, self-reliant and self-assured. Isn’t this incorrect? MY SECOND VISION for India is 
 DEVELOPMENT. For fifty years we have been a developing nation. It is time we see ourselves as a developed nation.
 We are among top five nations in the world in terms of GDP."""
 
sentences=nltk.sent_tokenize(paragraph)
 
 #creating the object 
stemmer=PorterStemmer() 
 
 ##now we are trying to remove the stopword from each sentence
 
for i in range(len(sentences)):
     words=nltk.word_tokenize(sentences[i])
     words=[stemmer.stem(word) for word in words if word not in set(stopwords.words('english'))]
     sentences[i]=' '.join(words)
     
##THE MAIN PROBLEM WITH STEMMING:
    ##PRODCUED INTERMIDEIATE OWRD WHICH DOES NOT HAVE ANY MEANING IN THE REAL WORLD
    
