# This file has small examples that are meant to be run individually in the Python shell
# examples for lab in week 7 - Parsing 

## Probabilistic CFG with verb subcategories
import nltk
from nltk import *
prob_grammar = nltk.PCFG.fromstring("""
  S -> NP VP [0.9]| VP  [0.1]
  VP -> TranV NP [0.3]
  VP -> InV  [0.3]
  VP -> DatV NP PP  [0.4]
  PP -> P NP   [1.0]
  TranV -> "saw" [0.2] | "ate" [0.2] | "walked" [0.2] | "shot" [0.2] | "book" [0.2]
  InV -> "ate" [0.5] | "walked" [0.5]
  DatV -> "gave" [0.2] | "ate" [0.2] | "saw" [0.2] | "walked" [0.2] | "shot" [0.2]
  NP -> Prop [0.2]| Det N [0.4] | Det N PP [0.4]
  Prop -> "John" [0.25]| "Mary" [0.25] | "Bob" [0.25] | "I" [0.25] 
  Det -> "a" [0.2] | "an" [0.2] | "the" [0.2] | "my" [0.2] | "that" [0.2]
  N -> "man" [0.15] | "dog" [0.15] | "cat" [0.15] | "park" [0.15] | "telescope" [0.1] | "flight" [0.1] | "elephant" [0.1] | "pajamas" [0.1]
  P -> "in" [0.2] | "on" [0.2] | "by" [0.2] | "with" [0.2] | "through" [0.2]
  """)

viterbi_parser = nltk.ViterbiParser(prob_grammar)
for tree in viterbi_parser.parse(['John', 'saw', 'a', 'telescope']):
    print (tree)
## Last week�s Exercise
# Define sentences for the exercise (the last sentence is newly added here)

sentex1 = "I prefer a flight through Houston".split()
sentex2 = "Jack walked with the dog".split()
sentex3 = "I want to book that flight".split()
sentex4 = "John gave the dog a bone".split()

# extend the flight grammar:
flight_grammar = nltk.CFG.fromstring("""
  S -> NP VP | VP
  VP -> V NP | V NP PP | V PP
  PP -> P NP | P VP
  V -> "saw" | "ate" | "walked" | "shot" | "book" | "want" | "prefer"
  NP -> Prop | Det N | Det N PP
  Prop -> "John" | "Mary" | "Bob" | "I" | "Houston" | "Jack"
  Det -> "a" | "an" | "the" | "my" | "that"
  N -> "man" | "dog" | "cat" | "telescope" | "park" | "elephant" | "pajamas" | "flight" 
  P -> "in" | "on" | "by" | "with" | "through" | "to"
  """)

# redefine rd_parser when you change the flight grammar
rd_parser = nltk.RecursiveDescentParser(flight_grammar)
for tree in rd_parser.parse(sentex3):
    print (tree)
   
