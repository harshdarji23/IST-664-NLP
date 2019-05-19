#!/usr/bin/env python
# coding: utf-8

# In[1]:


import nltk
from nltk.tokenize import sent_tokenize
from nltk.corpus import sentence_polarity
import random


# In[16]:


reviewdata=open('C:/Users/Harsh Darji/Desktop/baby.txt').readlines()


# In[21]:


reviewTextContent=[]
reviewYear=[]


# In[26]:


pattern = r'''(?x)
    reviewText[:](.+)
    '''
year_pattern = r''' (?x)
   reviewTime[:](.+)
''' 


# In[36]:


for line in reviewdata:
    
    sentlist = []
    reviewyear_tokens = nltk.regexp_tokenize(line,year_pattern)
    reviewtext_tokens = nltk.regexp_tokenize(line,pattern)
    
    if(len(reviewtext_tokens)>0):
        current_review = reviewtext_tokens[0].strip("\n")
        reviewTextContent.append(current_review)
        
    if(len(reviewyear_tokens)>0):
        current_year = reviewyear_tokens[0].split(',')[1].strip('\n').strip(' ')
        reviewYear.append(current_year)


# In[39]:


finaldata=[]
for i in range(0, len(reviewYear)):
    if(reviewYear[i]== '2012'):
        finaldata.append(reviewTextContent[i])


# In[41]:


print("Total no of reviews",len(finaldata))


# In[42]:


finaldata[:5]


# In[ ]:





# In[58]:


import nltk
import re
from nltk.corpus import PlaintextCorpusReader
from nltk.tokenize import TweetTokenizer
from nltk.corpus import sentence_polarity
import random
from nltk.corpus import wordnet as wn
from nltk.corpus import sentiwordnet as swn
from nltk.corpus import stopwords

from nltk.stem import PorterStemmer
from nltk.metrics import precision
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.metrics import ConfusionMatrix


# In[59]:


from nltk.corpus import subjectivity


# In[65]:



# Sentence level tokenization
sentences=[nltk.sent_tokenize(sent) for sent in finaldata]


# In[66]:


# tokenize each review
# Sentence level tokenization, I am using Tweet Tokenizer because 
# if we use nltk.sent_tokenize, the words like didn't gets split, which we dont want.

tknzr = TweetTokenizer()
original_tokenized_review_sentences = []
for each_review in sentences:
    # tokenize each review into words by splitting them into different sentences
    sent_wordlevel=[tknzr.tokenize(sent) for sent in each_review]
    for each in sent_wordlevel:
        original_tokenized_review_sentences.append(each)
        
print(len(original_tokenized_review_sentences))
print(original_tokenized_review_sentences[-3:-1])


# In[67]:



# converting the sentences to lower case to have the uniformity during classification
tokenized_review_sentences = []
# sentences now has the originally selected sentences from the reiviews file
# and lower_case_sentences has the 
for sentence in original_tokenized_review_sentences:
    tokenized_review_sentences.append([item.lower() for item in sentence])
    
print(tokenized_review_sentences[-3:-1])


# In[68]:



print(len(tokenized_review_sentences))


# In[69]:


# Stop Words
stop_words = set(stopwords.words('english'))
print(stop_words)

negationwords = ['no', 'not', 'never', 'none', 'nowhere', 'nothing', 'noone', 'rather', 'hardly', 'scarcely', 'rarely', 'seldom', 'neither', 'nor']

neg_stop_words = []
print(type(stop_words))
for word in stop_words:
    if (word in negationwords) or (word.endswith("n't")):
        neg_stop_words.append(word)
    
#print(neg_stop_words)
#print(stop_words)      
neg_stop_words = set(neg_stop_words)
new_stop_words = []
new_stop_words = list(stop_words - neg_stop_words)


# In[70]:


breakdown3 = swn.senti_synset('besides.r.02')
print(breakdown3.pos_score())
print(breakdown3.neg_score())
print(breakdown3.obj_score())


# In[71]:


list(swn.senti_synsets("very"))


# In[72]:


breakdown3 = swn.senti_synset('very.s.01')
print(breakdown3.pos_score())
print(breakdown3.neg_score())
print(breakdown3.obj_score())


# In[73]:


# Sentiment Analysis


# In[74]:


# Classsification-Bag of Words


# In[75]:


sentences = sentence_polarity.sents()
print(sentence_polarity.categories())
documents = [(sent, cat) for cat in sentence_polarity.categories() 
    for sent in sentence_polarity.sents(categories=cat)]


# In[76]:



documents = [(sent, cat) for cat in sentence_polarity.categories() 
    for sent in sentence_polarity.sents(categories=cat)]


# In[77]:



random.shuffle(documents)


# In[78]:


all_words_list = [word for (sent,cat) in documents for word in sent]
all_words = nltk.FreqDist(all_words_list)
word_items = all_words.most_common(2000)
word_features = [word for (word, freq) in word_items]


# In[79]:


def document_features(document, word_features):
    document_words = set(document)
    features = {}
    for word in word_features:
        features['contains({})'.format(word)] = (word in document_words)
    return features


# In[80]:


featuresets = [(document_features(d,word_features), c) for (d,c) in documents]


# In[81]:


train_set, test_set = featuresets[1000:], featuresets[:1000]
classifier = nltk.NaiveBayesClassifier.train(train_set)
print (nltk.classify.accuracy(classifier, test_set))


# In[82]:


# Trying with 3000 most frequent bag of words
sentences = sentence_polarity.sents()
print(sentence_polarity.categories())
documents = [(sent, cat) for cat in sentence_polarity.categories() 
    for sent in sentence_polarity.sents(categories=cat)]

random.shuffle(documents)

all_words_list = [word for (sent,cat) in documents for word in sent]
all_words = nltk.FreqDist(all_words_list)
word_items = all_words.most_common(3000)
word_features = [word for (word, freq) in word_items]

def document_features(document, word_features):
    document_words = set(document)
    features = {}
    for word in word_features:
        features['contains({})'.format(word)] = (word in document_words)
    return features

featuresets = [(document_features(d,word_features), c) for (d,c) in documents]

train_set, test_set = featuresets[1000:], featuresets[:1000]
classifier = nltk.NaiveBayesClassifier.train(train_set)
print (nltk.classify.accuracy(classifier, test_set))


# In[84]:


# recalculating Document Feature after removing stop words

sentences = sentence_polarity.sents()
print(sentence_polarity.categories())
documents = [(sent, cat) for cat in sentence_polarity.categories() 
    for sent in sentence_polarity.sents(categories=cat)]

random.shuffle(documents)

# all_word_list after removing the stop words
all_words_list = [word for (sent,cat) in documents for word in sent if word not in new_stop_words]
all_words = nltk.FreqDist(all_words_list)
word_items = all_words.most_common(3000)
word_features = [word for (word, freq) in word_items]

# bag of words approach
def document_features(document, word_features):
    document_words = set(document)
    features = {}
    for word in word_features:
        features['contains({})'.format(word)] = (word in document_words)
    return features

# define the feature sets using the document_features
featuresets = [(document_features(d,word_features), c) for (d,c) in documents]

# Train and test your model for accuracy
train_set, test_set = featuresets[1000:], featuresets[:1000]
classifier = nltk.NaiveBayesClassifier.train(train_set)
print (nltk.classify.accuracy(classifier, test_set))


# In[83]:


# CLASSIFICATION-- SUBJECTIVITY COUNT FEATURES


# In[87]:


nltk.download('averaged_perceptron_tagger')
nltk.download('sentence_polarity')
nltk.download('punkt')


# In[88]:


from Subjectivity import *


# In[90]:



SLpath = 'subjclueslen1-HLTEMNLP05.tff'
SL = readSubjectivity(SLpath)
print(SL['absolute'])


# In[91]:


# define the features, to find out the feature_set
def SL_features(document, word_features, SL):
    document_words = set(document)
    features = {}
    for word in word_features:
        features['contains(%s)' % word] = (word in document_words)
        # count variables for the 4 classes of subjectivity
        weakPos = 0
        strongPos = 0
        weakNeg = 0
        strongNeg = 0
        for word in document_words:
            if word in SL:
                strength, posTag, isStemmed, polarity = SL[word]
                if strength == 'weaksubj' and polarity == 'positive':
                    weakPos += 1
                if strength == 'strongsubj' and polarity == 'positive':
                    strongPos += 1
                if strength == 'weaksubj' and polarity == 'negative':
                    weakNeg += 1
                if strength == 'strongsubj' and polarity == 'negative':
                    strongNeg += 1
                features['positivecount'] = weakPos + (2 * strongPos)
                features['negativecount'] = weakNeg + (2 * strongNeg)      
    return features


#define the feature set for performinh the classification
# word features here is the revised word features after removing the stop words
SL_featuresets = [(SL_features(d, word_features, SL), c) for (d,c) in documents]

print(SL_featuresets[0][0]['positivecount'])
print(SL_featuresets[0][0]['negativecount'])

train_set, test_set = SL_featuresets[1000:], SL_featuresets[:1000]
classifier = nltk.NaiveBayesClassifier.train(train_set)
print(nltk.classify.accuracy(classifier, test_set))


# In[92]:


# SENTIMENT--NEGATION


# In[93]:



negationwords = ['no', 'not', 'never', 'none', 'nowhere', 'nothing', 'noone', 'rather', 'hardly', 'scarcely', 'rarely', 'seldom', 'neither', 'nor']

def NOT_features(document, word_features, negationwords):
    features = {}
    for word in word_features:
        features['contains({})'.format(word)] = False
        features['contains(NOT{})'.format(word)] = False
    # go through document words in order
    for i in range(0, len(document)):
        word = document[i]
        if ((i + 1) < len(document)) and ((word in negationwords) or (word.endswith("n't"))):
            i += 1
            features['contains(NOT{})'.format(document[i])] = (document[i] in word_features)
        else:
            features['contains({})'.format(word)] = (word in word_features)
    return features


# In[94]:


#this word_features is the list of word_features after removing the stop words
NOT_featuresets = [(NOT_features(d, word_features, negationwords), c) for (d, c) in documents]
NOT_featuresets[0][0]['contains(NOTlike)']
NOT_featuresets[0][0]['contains(always)']

train_set, test_set = NOT_featuresets[1000:], NOT_featuresets[:1000]
classifier = nltk.NaiveBayesClassifier.train(train_set)
print(nltk.classify.accuracy(classifier, test_set))

classifier.show_most_informative_features(30)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




