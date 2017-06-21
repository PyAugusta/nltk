import os
import json
import pickle
import random
import logging

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.classify import DecisionTreeClassifier
from nltk import NaiveBayesClassifier


log = logging.getLogger('tester')
log.setLevel(logging.INFO)
log_formatter = logging.Formatter('##### %(asctime)s - %(name)s - %(levelname)s:\n %(message)s\n')

stream_handler = logging.StreamHandler()
stream_handler.setFormatter(log_formatter)
stream_handler.setLevel(logging.INFO)

log.addHandler(stream_handler)


install_dir = os.path.abspath(os.path.dirname(__file__))
classifiers_dir = os.path.join(install_dir, 'classifiers')
tokens_dir = os.path.join(install_dir, 'tokens')
twitter_dir = os.path.join(install_dir, 'tutorial_data', 'corpora', 'twitter_samples')
twitter_data = [f[:f.rfind('.')] for f in os.listdir(twitter_dir) if f.endswith('.json')]


TOKENS = set()
NEGATIVE = 0
POSITIVE = 1


class FalseDict(dict):

    def __init__(self, iterable=None):
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
        log.info('Loaded {} of {} lines of data from {}'.format(loaded_lines, total_lines, filename))
    return data
    
    
def tokenized_tweets(twitter_data):
    tweets = []
    for data_row in twitter_data:
        tokenized = set([w.lower().strip() for w in word_tokenize(data_row['text']) if w.lower() not in stopwords.words('english')])
        tweets.append(tokenized)
        TOKENS.update(tokenized)
    return tweets

        
def find_features(tkns):
    features = FalseDict()
    for tkn in TOKENS:
        if tkn in tkns:
            features[tkn] = True
    return features
    
def pickle_it(obj, fpath):
    with open(fpath, 'wb') as f:
        pickle.dump(obj, f)


def save_classifier(obj, fname):
    fpath = os.path.join(classifiers_dir, fname)
    pickle_it(obj, fpath)
    log.info('saved classifier at {}'.format(fpath))

def save_tokens(obj, fname):
    fpath = os.path.join(tokens_dir, fname)
    pickle_it(obj, fpath)
    log.info('saved tokens at {}'.format(fpath))
    
    
def main():
    log.info('loading negative_tweets')
    neg_data = load_twitter_data('negative_tweets')
    neg_tokens = tokenized_tweets(neg_data[:1000])

    log.info('loading positive_tweets')
    pos_data = load_twitter_data('positive_tweets')
    pos_tokens = tokenized_tweets(pos_data[:1000])

    log.info('creating feature set from negative tweets - labeled with {}'.format(NEGATIVE))
    neg_features = [(find_features(tkns), NEGATIVE) for tkns in neg_tokens]
    
    log.info('creating feature set from positive tweets - labeled with {}'.format(POSITIVE))
    pos_features = [(find_features(tkns), POSITIVE) for tkns in pos_tokens]

    all_features = neg_features + pos_features
    random.shuffle(all_features)

    half = int(len(all_features) / 2)

    training_set = all_features[:half]
    testing_set = all_features[half:]

    DT_classifier = DecisionTreeClassifier.train(training_set)
    log.info(("DT_classifier accuracy percent:", (nltk.classify.accuracy(DT_classifier, testing_set))*100))
    save_classifier(DT_classifier, 'dt_classifier.pickle')
    
    NB_classifier = nltk.NaiveBayesClassifier.train(training_set)
    log.info(("Original Naive Bayes Algo accuracy percent:", (nltk.classify.accuracy(NB_classifier, testing_set))*100))
    save_classifier(NB_classifier, 'nb_classifier.pickle')
    
    save_tokens(TOKENS, 'tokens.pickle')

    
if __name__ == '__main__':
    main()
