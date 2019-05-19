# Python Examples for Week 8

import nltk

### WordNet in NLTK  ####

# import WordNet and shorten its names as "wn"
from nltk.corpus import wordnet as wn 

# given word "dog", returns the ids of the synsets
wn.synsets('dog')

# given a synset id, find words/lemma names (the synonyms) of the first noun sense of "dog"
wn.synset('dog.n.01').lemma_names()

# given a synset id, find lemmas of the synset (a lemma pairs a word with a synset)
wn.synset('dog.n.01').lemmas()
# find synset of a lemma
wn.lemma('dog.n.01.domestic_dog').synset()

# find lemma names for all senses of a word
for synset in wn.synsets('dog'):
	print (synset, ":  ", synset.lemma_names())


# find definition of the first noun sense of dog, or namely, the dog.n.01 synset
wn.synset('dog.n.01').definition()

# display an example of the synset
wn.synset('dog.n.01').examples()

# or show the definitions for all the synsets of a word
for synset in wn.synsets('dog'):
	print (synset, ":  ", synset.definition())

# or combine the synonyms/lemma names, definitions and examples
for synset in wn.synsets('dog'):
	print (synset, ":  ")
	print ('     ', synset.lemma_names())
	print ('     ', synset.definition())
	print ('     ', synset.examples())

##  Lexical relations between synsets in WordNet
# find hypernyms of synsets
dog1 = wn.synset('dog.n.01')
dog1.hypernyms()

# find hyponyms
dog1.hyponyms()

# the most general hypernym of a synset
dog1.root_hypernyms()

# from the wordnet browser, we see that dog1 has two more relations
dog1.part_meronyms()
# what is this?  check it out print (wn.synset('flag.n.07').lemma_names(),wn.synset('flag.n.07').definition(), wn.synset('flag.n.07').examples())
dog1.member_holonyms()

# look at another word, "good"
wn.synsets('good')

# find antonyms, sometimes need to specify for which lemma the antonym is needed
good1 = wn.synset('good.a.01')
# display synonyms of this synset
good1.lemma_names()

# the antonym function is defined only on the lemma, not the synset
# find antonym for the first lemma of the synset
good1.lemmas()
good1.lemmas()[0].antonyms() 

# find entailments of verbs
wn.synset('walk.v.01').entailments()
wn.synset('eat.v.01').entailments()

# trace paths of a synset by visiting its hypernyms
dog1.hypernyms()
paths=dog1.hypernym_paths()

# number of paths from the synset to the root concept "entity"
len(paths) 
# look at the first path
paths[0]
# or just list the names in the paths
#list the first path
[synset.name() for synset in paths[0]]

#list the second path 
[synset.name() for synset in paths[1]] 

# Word similarity

# define 3 different types of whales
right = wn.synset('right_whale.n.01')
minke = wn.synset('minke_whale.n.01')  orca = wn.synset('orca.n.01') 

# look at the paths of these three whales
right.hypernym_paths()
minke.hypernym_paths()
orca.hypernym_paths()

# find the least ancestor of right and minkeright.lowest_common_hypernyms(minke)
right.lowest_common_hypernyms(orca)


# the function min_depth gives the length of a path from a word to the top of the hierarchy
right.min_depth() wn.synset('baleen_whale.n.01').min_depth() wn.synset('entity.n.01').min_depth()

# the path similarity gives a similarity score between 0 and 1
right.path_similarity(minke) right.path_similarity(orca)

# define 2 more words and look at their similarity
tortoise = wn.synset('tortoise.n.01')
novel = wn.synset('novel.n.01')
# note the least ancestor of these two words
right.lowest_common_hypernyms(tortoise)
right.lowest_common_hypernyms(novel)

right.path_similarity(tortoise) right.path_similarity(novel)

# try Leacock-Chodorow Similarity
right.lch_similarity(orca)
right.lch_similarity(tortoise)
right.lch_similarity(novel)

# first get information content from a general corpus
from nltk.corpus import wordnet_ic
brown_ic = wordnet_ic.ic('ic-brown.dat')

# try Resnik Similarity
right.res_similarity(orca, brown_ic)
right.res_similarity(tortoise, brown_ic)
right.res_similarity(novel, brown_ic)


## SentiWordNet
from nltk.corpus import sentiwordnet as swn

list(swn.senti_synsets('breakdown'))
wn.synsets('breakdown')

breakdown3 = swn.senti_synset('breakdown.n.03')
print (breakdown3)

breakdown3.pos_score()
breakdown3.neg_score()
breakdown3.obj_score()

# some more exploration of sentiment scores of words
>>> dogswn1 = swn.senti_synset('dog.n.01')>>> print(dogswn1)
<dog.n.01: PosScore=0.0 NegScore=0.0>
>>> dogswn1.obj_score()
1.0

>>> goodswn1 = swn.senti_synset('good.a.01')
>>> print(goodswn1)
<good.a.01: PosScore=0.75 NegScore=0.0>
>>> goodswn1.obj_score()
0.25

# not all words in WordNet have been scored for sentiment in SentiWordNet
>>> wn.synsets('exuberant')
[Synset('ebullient.s.01'), Synset('excessive.s.02'), Synset('exuberant.s.03')]
>>> ex3 = swn.senti_synset('exuberant.s.03')
>>> print(ex3)
None





