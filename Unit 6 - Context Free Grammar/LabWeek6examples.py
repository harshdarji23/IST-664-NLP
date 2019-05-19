# This file has small examples that are meant to be run individually in the Python shell
# examples for lab in week 6 - Parsing with CFG grammars in NLTK

import nltk
from nltk import *


## write your own grammars
grammar = nltk.CFG.fromstring("""
  S -> NP VP
  VP -> V NP | V NP PP
  PP -> P NP
  V -> "saw" | "ate" | "walked"
  NP -> Prop | Det N | Det N PP
  Prop -> "John" | "Mary" | "Bob" 
  Det -> "a" | "an" | "the" | "my"
  N -> "man" | "dog" | "cat" | "telescope" | "park"
  P -> "in" | "on" | "by" | "with"
  """)

# top-down method: recursive descent parsing
rd_parser = nltk.RecursiveDescentParser(grammar)
senttext = "Mary saw Bob"

# tokenize the sentence by splitting on white space
# use nltk.word_tokenize() for more complex examples
sentlist = senttext.split()
trees = rd_parser.parse(sentlist)
# convert the generator to a list
treelist = list(trees)
# look at individual items
type(treelist[0])
# print the tree structures
for tree in treelist:
	print (tree)

# try an ambiguous sentence
sent2list = "John saw the man in the park with a telescope".split()
for tree in rd_parser.parse(sent2list):
	print (tree)

# extend the grammar with more words
groucho_grammar = nltk.CFG.fromstring("""
  S -> NP VP
  VP -> V NP | V NP PP
  PP -> P NP
  V -> "saw" | "ate" | "walked" | "shot"
  NP -> Prop | Det N | Det N PP
  Prop -> "John" | "Mary" | "Bob" | "I"
  Det -> "a" | "an" | "the" | "my"
  N -> "man" | "dog" | "cat" | "telescope" | "park" | "elephant" | "pajamas"
  P -> "in" | "on" | "by" | "with"
  """)

# try sent4 with the recursive descent parser on groucho grammar
sent4list = "I shot an elephant in my pajamas".split()
rd_parser = nltk.RecursiveDescentParser(groucho_grammar)
for tree in rd_parser.parse(sent4list):
	print (tree)

# extend the grammar for the flight grammar:
flight_grammar = nltk.CFG.fromstring("""
  S -> NP VP | VP
  VP -> V NP | V NP PP
  PP -> P NP
  V -> "saw" | "ate" | "walked" | "shot" | "book"
  NP -> Prop | Det N | Det N PP
  Prop -> "John" | "Mary" | "Bob" | "I"
  Det -> "a" | "an" | "the" | "my" | "that"
  N -> "man" | "dog" | "cat" | "telescope" | "park" | "elephant" | "pajamas" | "flight"
  P -> "in" | "on" | "by" | "with"
  """)

# make a recursive descent parser and parse the sentence
rd_parser = nltk.RecursiveDescentParser(flight_grammar)
sent5list = 'book that flight'.split()
for tree in rd_parser.parse(sent5list):
	print (tree)


## (Optional) Look at Dependency grammars in the NLTK book, section 8.5
# a dependency grammar for the groucho example
groucho_dep_grammar = nltk.DependencyGrammar.fromstring("""
  'shot' -> 'I' | 'elephant' | 'in'
  'elephant' -> 'an' | 'in'
  'in' -> 'pajamas'
  'pajamas' -> 'my'
  """)

print (groucho_dep_grammar)
pdp = nltk.ProjectiveDependencyParser(groucho_dep_grammar)
glist = 'I shot an elephant in my pajamas'.split()
trees = pdp.parse(glist)
for tree in trees:
    print (tree)


