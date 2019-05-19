# these are the python lines for the Week 9 lab, the first lab on classification

import nltk

# define a feature extraction function for each name
def gender_features(word):
    return{'last_letter': word[-1]}

gender_features('Shrek')

from nltk.corpus import names
names.words('male.txt')[:20]
namesgender = ([(name, 'male') for name in names.words('male.txt')] +
          [(name, 'female') for name in names.words('female.txt')])
len(namesgender)
namesgender[:20]
namesgender[7924:]


import random
random.shuffle(namesgender)
namesgender[:20]

# featuresets represent each name as features and a label
featuresets = [(gender_features(n), g) for (n, g) in namesgender]
featuresets[:20]

# create training and test sets, run a classifier and show the accuracy
train_set, test_set = featuresets[500:], featuresets[:500]
classifier = nltk.NaiveBayesClassifier.train(train_set)

# classify accuracy function runs the classifier on the test set and reports
#   comparisons between predicted labels and actual/gold labels
print(nltk.classify.accuracy(classifier, test_set))

# classify new instances
classifier.classify(gender_features('Neo'))
classifier.classify(gender_features('Trinity'))

classifier.show_most_informative_features(20)

#creating lots of features
def gender_features2(name):
    features = {}
    features["firstletter"] = name[0].lower()
    features["lastletter"] = name[-1].lower()
    for letter in 'abcdefghijklmnopqrstuvwxyz':
        features["count({})".format(letter)] = name.lower().count(letter)
        features["has({})".format(letter)] = (letter in name.lower())
    return features

features = gender_features2('Shrek')
len(features)
features

# create feature sets using this function
featuresets2 = [(gender_features2(n), g) for (n, g) in namesgender]

for (n, g) in namesgender[:2]:
    print(n, gender_features2(n), '\n')

# create new training and test sets, classify and look at accuracy
train_set, test_set = featuresets2[500:], featuresets2[:500]
classifier = nltk.NaiveBayesClassifier.train(train_set)
print(nltk.classify.accuracy(classifier, test_set))


## Experiment
# go back and separate the names into training and test
train_names = namesgender[500:]
test_names = namesgender[:500]

# use our original features to train a classify and test on the development test set
train_set = [(gender_features(n), g) for (n, g) in train_names]
test_set = [(gender_features(n), g) for (n, g) in test_names]
classifier = nltk.NaiveBayesClassifier.train(train_set)
print(nltk.classify.accuracy(classifier, test_set))

# save the classifier accuracy for use in the exercise

# define a function that will compare the classifier labels with the gold standard labels
def geterrors(test):
    errors = []
    for (name, tag) in test:
        guess = classifier.classify(gender_features(name))
        if guess != tag:
            errors.append( (tag, guess, name) )
    return errors

errors = geterrors(test_names)
len(errors)

# define a function to print the errors
def printerrors(errors):
    for (tag, guess, name) in sorted(errors):
        print('correct={:<8s} guess={:<8s} name={:<30s}'.format(tag, guess, name))

printerrors(errors[:40])

# evaluation measures showing performance of classifier

from nltk.metrics import *

reflist = []
testlist = []
for (features, label) in test_set:
    reflist.append(label)
    testlist.append(classifier.classify(features))

reflist[:30]
testlist[:30]

# define and print confusion matrix

cm = ConfusionMatrix(reflist, testlist)
print(cm)

# define a set of item identifiers that are gold labels and a set of item identifiers that are predicted labels

reffemale = set()
refmale = set()
testfemale = set()
testmale = set()

for i, label in enumerate(reflist):
    if label == 'female': reffemale.add(i)
    if label == 'male': refmale.add(i)

for i, label in enumerate(testlist):
    if label == 'female': testfemale.add(i)
    if label == 'male': testmale.add(i)

reffemale
testfemale
refmale
testmale

# compute precision, recall and F-measure for each label

def printmeasures(label, refset, testset):
    print(label, 'precision:', precision(refset, testset))
    print(label, 'recall:', recall(refset, testset)) 
    print(label, 'F-measure:', f_measure(refset, testset))

printmeasures('female', reffemale, testfemale)
printmeasures('male', refmale, testmale)

# another feature extraction function for the exercise
def gender_features3(word):
    return {'suffix1': word[-1],'suffix2': word[-2]}

gender_features3('Shrek')