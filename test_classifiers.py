##################################
## Import statements
import os
import json
import pickle
import random
import logging
import operator
from collections import defaultdict

import nltk
from nltk.corpus import stopwords # contains lists of words that have little to no meaning
from nltk.tokenize import word_tokenize # smart splitting function

# http://mines.humanoriented.com/classes/2010/fall/csci568/portfolio_exports/lguo/decisionTree.html
from nltk.classify import DecisionTreeClassifier

# https://stackoverflow.com/questions/10059594/a-simple-explanation-of-naive-bayes-classification
from nltk import NaiveBayesClassifier


##################################
## Set up our logger 
log = logging.getLogger('tester')
log.setLevel(logging.INFO)

logconsole_formatter = logging.Formatter('%(message)s')
logfile_formatter = logging.Formatter('##### %(asctime)s - %(name)s - %(levelname)s:\n %(message)s')

stream_handler = logging.StreamHandler()
stream_handler.setFormatter(logconsole_formatter)
stream_handler.setLevel(logging.INFO)

file_handler = logging.FileHandler('tester.log')
file_handler.setFormatter(logfile_formatter)
file_handler.setLevel(logging.INFO)

log.addHandler(stream_handler)
log.addHandler(file_handler)


##################################
## Directories of interest
install_dir = os.path.abspath(os.path.dirname(__file__))
classifiers_dir = os.path.join(install_dir, 'classifiers')
tokens_dir = os.path.join(install_dir, 'tokens')
twitter_dir = os.path.join(install_dir, 'tutorial_data', 'corpora', 'twitter_samples')
twitter_data = [f[:f.rfind('.')] for f in os.listdir(twitter_dir) if f.endswith('.json')]


##################################
## GLOBALS
TOKEN_DIST = defaultdict(int)
NEGATIVE = 'neg'
POSITIVE = 'pos'


##################################
## Classes and functions

class FalseDict(dict):

    def __init__(self, iterable=None):
        '''Special dictionary object that will return False when
        trying to access keys that don't exist.
        This class has near identical behaviour as
        
        >>> import collections
        >>> d = collections.defaultdict(bool)
        
        but I discovered that after writing it. It's a good example of how to 
        subclass a standard container in order to customize its behaviour, so I'm
        leaving it here.
        '''
        if iterable is not None:
            super(FalseDict, self).__init__(iterable)
        else:
            super(FalseDict, self).__init__()
            
            
    def __getitem__(self, key):
        try:
            val = dict.__getitem__(self, key)
        except KeyError:
            val = False
        return val
        
    def get(self, key):
        return self.__getitem__(key)
        


def load_twitter_data(filename):
    '''Convenience function that will load a json file from the 
    twitter_dir given it's name (without extension)
    '''
    if filename not in twitter_data:
        raise ValueError("filename invalid. must be in {}".format(twitter_data))
    data = []
    with open(os.path.join(twitter_dir, filename + '.json'), 'r') as f:
        total_lines = 0
        loaded_lines = 0
        for line in f.readlines():
            try:
                data.append(json.loads(line))
                loaded_lines += 1
            except:
                pass
            finally:
                total_lines += 1
        log.info('loaded {} of {} lines of data from {}'.format(loaded_lines, total_lines, filename))
    return data
    
    
def tokenized_tweets(twitter_data):
    '''Extracts and tokenizes the 'text' value from each
    row in twitter_data, and updates the TOKEN_DIST global
    '''
    tweets = []
    for data_row in twitter_data:
        tokenized = {w.lower() for w in word_tokenize(data_row['text']) if w.lower() not in stopwords.words('english')}
        for tkn in tokenized:
            TOKEN_DIST[tkn] += 1
        tweets.append(tokenized)
    return tweets

    
def top_n_tokens(n):
    '''returns the top n number of tokens from the TOKEN_DIST
    global defaultdict
    '''
    top_n = [word for word, count in sorted(TOKEN_DIST.items(), key=operator.itemgetter(1))][:n+1]
    return top_n
    
        
def find_features(tkns, word_features):
    '''Given a set of word tokens, this will create a
    FalseDict object that will serve as a feature set.
    '''
    features = FalseDict()
    for w in word_features:
        if w in tkns:
            features[w] = True
    return features
    
def pickle_it(obj, fpath):
    '''pickles an object to the given fpath
    '''
    with open(fpath, 'wb') as f:
        pickle.dump(obj, f)


def save_classifier(obj, fname):
    '''pickles an object into the classifiers_dir
    '''
    fpath = os.path.join(classifiers_dir, fname)
    pickle_it(obj, fpath)
    log.info('saved classifier at {}'.format(fpath))

def save_tokens(obj, fname):
    '''pickles an object into the tokens_dir
    '''
    fpath = os.path.join(tokens_dir, fname)
    pickle_it(obj, fpath)
    log.info('saved tokens at {}'.format(fpath))
    
    
def main():
    log.info('loading negative_tweets')
    neg_data = load_twitter_data('negative_tweets')
    neg_tokens = tokenized_tweets(neg_data)
    log.info('tokenized negative_tweets')

    log.info('loading positive_tweets')
    pos_data = load_twitter_data('positive_tweets')
    pos_tokens = tokenized_tweets(pos_data)
    log.info('tokenized positive_tweets')
    
    log.info('extracting the top 5000 words')
    top_5000 = top_n_tokens(5000)

    log.info('creating feature set from negative tweets - labeled with {}'.format(NEGATIVE))
    neg_features = [(find_features(tkns, top_5000), NEGATIVE) for tkns in neg_tokens]
    
    log.info('creating feature set from positive tweets - labeled with {}'.format(POSITIVE))
    pos_features = [(find_features(tkns, top_5000), POSITIVE) for tkns in pos_tokens]

    log.info('combining and shuffling feature sets')
    all_features = neg_features + pos_features
    random.shuffle(all_features)

    log.info('splitting the combined feature sets ~(70/30)')
    split_idx = int(len(all_features) * .7)
    training_set = all_features[:split_idx]
    testing_set = all_features[split_idx:]
    log.info('training data set contains {} features'.format(len(training_set)))
    log.info('testing data set contains {} features'.format(len(testing_set)))

    log.info("\n#####  STARTING CLASSIFIER TRAINING #####")
    log.info("this will take a while\n")
    
    DT_classifier = DecisionTreeClassifier.train(training_set)
    log.info(("DT_classifier accuracy percent:", (nltk.classify.accuracy(DT_classifier, testing_set))*100))
    save_classifier(DT_classifier, 'dt_classifier.pickle')
    
    NB_classifier = nltk.NaiveBayesClassifier.train(training_set)
    log.info(("Original Naive Bayes Algo accuracy percent:", (nltk.classify.accuracy(NB_classifier, testing_set))*100))
    save_classifier(NB_classifier, 'nb_classifier.pickle')
    
    save_tokens(top_5000, 'top_5000.pickle')

    
if __name__ == '__main__':
    main()
