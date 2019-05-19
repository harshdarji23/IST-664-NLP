# program to get text from Gutenberg files

import nltk
from nltk import FreqDist

fileno = 1

nfile = nltk.corpus.gutenberg.fileids( ) [fileno]
ntext = nltk.corpus.gutenberg.raw(nfile)
print('First 200 characters of text', ntext[:200])

ntokens = nltk.word_tokenize(ntext)
print('First 100 tokens', ntokens[:100])

nwords = [w.lower( ) for w in ntokens]

ndist = FreqDist(nwords)

print('30 Most Frequent Words:')
nitems = ndist.most_common(30)
for item in nitems:
    print(item[0], item[1])
