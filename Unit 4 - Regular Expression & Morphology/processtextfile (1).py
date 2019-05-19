'''
	This program reads and processes text from a file.
	To change the file, change the filepath variable
	The program gets the raw text, tokenizes and lowercases the tokens.
	It puts the tokens in a frequency distribution and displays the 30 most frequent.
'''
# open python and nltk packages needed for processing
import nltk
import re

# put the full path to the file here (or can use relative path from the directory of the program)
#filepath = '/Users/njmccrac/NLPfall2016/labs/LabExamplesWeek4/CrimeAndPunishment.txt'
#filepath = 'H:\NLPclass\LabExamplesWeek4\CrimeAndPunishment.txt'
filepath = 'CrimeAndPunishment.txt'

def alpha_filter(w):
  # pattern to match word of non-alphabetical characters
  pattern = re.compile('^[^a-z]+$')
  if (pattern.match(w)):
    return True
  else:
    return False

# open the file, read the text and close it
f = open(filepath, 'r')
filetext = f.read()
f.close()

# tokenize by the regular word tokenizer
filetokens = nltk.word_tokenize(filetext)

# choose to treat upper and lower case the same
#    by putting all tokens in lower case
filewords = [w.lower() for w in filetokens]


# display the first words
print ("Display first 50 words from file:")
print (filewords[:50])

# setup to process bigrams
from nltk.collocations import *
bigram_measures = nltk.collocations.BigramAssocMeasures()
       
finder = BigramCollocationFinder.from_words(filewords)
# choose to use both the non-alpha word filter and a stopwords filter
finder.apply_word_filter(alpha_filter)
finder.apply_word_filter(lambda w: w in stopwords)

# score by frequency and display the top 50 bigrams
scored = finder.score_ngrams(bigram_measures.raw_freq)
print ()
print ("Bigrams from file with top 50 frequencies")
for item in scored[:20]:
        print (item)

# score by PMI and display the top 50 bigrams
# only use frequently occuring words in mutual information
finder.apply_freq_filter(5)
scored = finder.score_ngrams(bigram_measures.pmi)

print ("\nBigrams from file with top 50 mutual information scores")
for item in scored[:20]:
        print (item)


