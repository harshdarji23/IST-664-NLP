# classify part of speech based on sentence context

>>> import nltk
>>> from nltk.corpus import brown

# define features for the "i"th word in the sentence, including three types of suffix 
#     and one pre-word
# the pos features function takes the sentence of untagged words and the index of a word i
#   it creates features for word i, including the previous word i-1
def pos_features(sentence, i):    
    features = {"suffix(1)": sentence[i][-1:],
		    "suffix(2)": sentence[i][-2:],
		    "suffix(3)": sentence[i][-3:]}
    if i == 0:
        features["prev-word"] = "<START>"
    else:
        features["prev-word"] = sentence[i-1]
    return features 

# look at features of a specific word in a specific sentence
# first sentence of brown corpus
>>> sentence0 = brown.sents()[0]
>>> sentence0
# word 8 of sentence 0
>>> sentence0[8]

# pos features of the word 8 
>>> pos_features(sentence0, 8)

# get the POS tagged sentences with categories of news
>>> tagged_sents = brown.tagged_sents(categories='news')
>>> tag_sent0 = tagged_sents[0]
>>> tag_sent0

# the function nltk.tag.untag will take the tags off
>>> nltk.tag.untag(tag_sent0)
>>> for i,(word,tag) in enumerate(tag_sent0):
    print (i, word, tag)

# get feature sets of words appearing in the corpus, from untagged sentences.
# and then get their tags from corresponding tagged sentence
# use the Python function enumerate to pair the index numbers with sentence words 
#   for the pos features function
>>> featuresets = []
>>> for tagged_sent in tagged_sents:
	untagged_sent = nltk.tag.untag(tagged_sent)
	for i, (word, tag) in enumerate(tagged_sent):
		featuresets.append( (pos_features(untagged_sent, i), tag) )

# look at the feature sets of the first 10 words
>>> for f in featuresets[:10]:
	print (f)
	
# using naive Bayesian as classifier
# split data into a training set and a test set
>>> size = int(len(featuresets) * 0.1)
>>> train_set, test_set = featuresets[size:], featuresets[:size]
>>> len(train_set)
>>> len(test_set)

# train classifier on the training set
>>> classifier = nltk.NaiveBayesClassifier.train(train_set)

# evaluate the accuracy (this will take a little while)
>>> nltk.classify.accuracy(classifier, test_set)
# the result should be 0.78915962207856782, which is reasonable for features without the previous tag


### sentence segmentation
sents = nltk.corpus.treebank_raw.sents()
len(sents)
for sent in sents[:10]:
    print (sent)

# initialize an empty token list, an empty boundaries set and offset as the integer 0
tokens = [ ]
boundaries = set()
offset = 0
# make a list of tokens with sentence boundaries
#   the offset is set to the index of a sentence boundary
for sent in nltk.corpus.treebank_raw.sents():
      tokens.extend(sent)
      offset += len(sent)
      boundaries.add(offset - 1)

# look at tokens and boundaries
tokens[:40]
len(boundaries)
0 in boundaries
1 in boundaries
19 in boundaries
20 in boundaries
for num, tok in enumerate(tokens[:40]):
     print (num, tok, '\t', num in boundaries)

# feature extraction function
# token is a list of words and we get the features of the token at offset i
def punct_features(tokens, i):
    return {'next-word-capitalized': tokens[i+1][0].isupper(),
        'prevword': tokens[i-1].lower(),
        'punct': tokens[i],
        'prev-word-is-one-char': len(tokens[i-1]) == 1}

# feature dictionary for the period at index 20
tokens[20]
punct_features(tokens,20)

# Define featuresets of all candidate punctuation
Sfeaturesets = [(punct_features(tokens, i), (i in boundaries))
      for i in range(1, len(tokens) - 1)
      if tokens[i] in '.?!']

# look at the feature sets of the first 10 punctuation symbols
for sf in Sfeaturesets[:10]:
	print (sf)

# separate into training and test sets and build classifier
size = int(len(Sfeaturesets) * 0.1)
size

Strain_set, Stest_set = Sfeaturesets[size:], Sfeaturesets[:size]
Sclassifier = nltk.NaiveBayesClassifier.train(Strain_set)
nltk.classify.accuracy(Sclassifier, Stest_set)

# this is the . after Nov
Sclassifier.classify(punct_features(tokens, 18))
# this is the . after 29, which should be true!
Sclassifier.classify(punct_features(tokens, 20))
# this is the . after group
Sclassifier.classify(punct_features(tokens, 36))


# define function to use the trained classifier to label sentences
def segment_sentences(words):
      start = 0
      sents = []
      for i, word in enumerate(words):
          if word in '.?!' and i < len(words) - 1 Sclassifier.classify(punct_features(words, i)) == True:
              sents.append(words[start:i+1])
              start = i+1
      if start < len(words):
          sents.append(words[start:])
      return sents

len(tokens)
tokens[:50]

tinytokens = tokens[:1000]

for s in segment_sentences(tinytokens):
    print (s)

# compare to NLKT default sentence tokenizer, which works on raw text instead of tokens
from nltk.tokenize import sent_tokenize

rawtext = 'Pierre Vinken, 61 years old, will join the board as a nonexecutive director Nov. 29.  Mr. Vinken is chairman of Elsevier N.V., the Dutch publishing group.'
sents = nltk.sent_tokenize(rawtext)
sents


