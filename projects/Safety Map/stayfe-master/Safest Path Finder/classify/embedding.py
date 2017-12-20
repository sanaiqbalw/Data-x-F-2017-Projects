# -*- coding: utf-8 -*-

import multiprocessing
import csv, json, re
from collections import defaultdict
from math import floor

import numpy as np
from gensim.models.word2vec import Word2Vec
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

import spacy
from spacy.en import English, STOP_WORDS
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS
import string
import re


# load english 
parser = English()
# stopwords
stopwords = ENGLISH_STOP_WORDS.union(STOP_WORDS)
# List of symbols we don't care about
symbols = " ".join(string.punctuation).split(" ") + ["-----", "---", "...", "“", "”", "'ve"]
# part of speech
useful_tags = set(['ADJ', 'VERB', 'ADV', 'NOUN', 'PROPN', 'PART', 'ADP', 'CONJ', 'INTJ', 'AUX'])
delimiters = ['. ', '\n', '.\n']
# min words
min_token_length = 2
# set random seed
np.random.seed(111)

# sklearn-compatible mean embedding
class MeanEmbeddingVectorizer(object):
    def __init__(self, word2vec):
        self.word2vec = word2vec
        # if a text is empty we should return a vector of zeros
        # with the same dimensionality as all the other vectors
        self.dim = len(next(iter(word2vec.values())))

    def fit(self, X, y):
        return self

    def transform(self, X):
        return np.array([
            np.mean([self.word2vec[w] for w in words if w in self.word2vec]
                    or [np.zeros(self.dim)], axis=0)
            for words in X
        ])

# sklearn-compatible tf-idf embedding
class TfidfEmbeddingVectorizer(object):
    def __init__(self, word2vec):
        self.word2vec = word2vec
        self.word2weight = None
        self.dim = len(next(iter(word2vec.values())))

    def fit(self, X, y):
        tfidf = TfidfVectorizer(analyzer=lambda x: x)
        tfidf.fit(X)
        # if a word was never seen - it must be at least as infrequent
        # as any of the known words - so the default idf is the max of
        # known idf's
        max_idf = max(tfidf.idf_)
        self.word2weight = defaultdict(
            lambda: max_idf,
            [(w, tfidf.idf_[i]) for w, i in tfidf.vocabulary_.items()])

        return self

    def transform(self, X):
        return np.array([
                np.mean([self.word2vec[w] * self.word2weight[w]
                         for w in words if w in self.word2vec] or
                        [np.zeros(self.dim)], axis=0)
                for words in X
            ])

# process document
class DocParser(object):
    def __init__(self, nlp, useful_tags, delimiters, min_token_length, **kwargs):
        self.nlp = nlp
        self.useful_tags = useful_tags
        self.delimiters = delimiters
        self.min_token_length = min_token_length
        self.splitter = self.splitterer()

    def cleanText(self, text):
        text = text.strip().replace("\n", " ").replace("\r", " ")
        mentionFinder = re.compile(r"@[a-z0-9_]{1,15}", re.IGNORECASE)
        text = mentionFinder.sub("@MENTION", text)
        text = text.replace("&amp;", "and").replace("&gt;", ">").replace("&lt;", "<")
        text = text.lower()
        return text

    def splitterer(self):
        escaped_delimiters = '|'.join([re.escape(delimiter) for delimiter in self.delimiters])
        return lambda value: re.split(escaped_delimiters, value)

    def text2tokens(self, text):
        tokens = []
        doc = self.nlp(text)
        for word in doc:
            if word.lemma_ in stopwords:
                continue
            if word.pos_ not in self.useful_tags:
                continue
            tokens.append("{}/{}".format(word.lemma_, word.pos_))
        return tokens

    def text2sentences(self, text):
        text = self.cleanText(text)
        candidates = self.splitter(text.strip())
        sentences = []
        for candidate in candidates:
            while len(candidate) and (candidate[-1] == '.' or candidate[-1] == ' '):
                candidate = candidate.strip(' ').strip('.')
            if not candidate:
                continue
            # remove link
            candidate = re.sub(r"\[\[(?:[^\]|]*\|)?([^\]|]+)\]\]", r'\1', candidate)
            tokens = self.text2tokens(candidate)
            if len(tokens) < self.min_token_length:
                continue
            sentences += tokens
        return sentences

# sklearn-compatible Word Mover's Distance metric
def wmd(X, Y=None):
    pass

# build word2vec from vocab
def build_word2vec(vocab, filename='word2vec', size=200):
    config = {
        'min_count': 5,
    	'window': 5,
    	'size': size,
    	'sg': 1,
    	'batch_words': 1000,
    	'iter': 100,
    	'workers': multiprocessing.cpu_count(),
    }
    # let vocab be a list of tokenized texts (i.e. list of lists of tokens)
    model = Word2Vec(vocab, **config)
    # trim unneeded model memory = use (much) less RAM
    model.init_sims(replace=True)
    model.save(filename)
    return model

# load pretrained word2vec
def load_word2vec(filename='word2vec'):
    model = Word2Vec.load(filename)
    return model

# preprocess data
def preprocess(parser):
    data_dir = '../data/keyword_filtered.csv'
    data = {}

    with open(data_dir, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # process content
            content = parser.text2sentences(row['Body_Text'])
            key = row['Crime_Type']
            if key not in data:
                data[key] = []
            data[key].append(content)

    return data

# save data
def save_data(data, name='dataset'):
    vectors, labels = [], []

    for key in data.keys():
        size = len(data[key])
        vectors += data[key]
        labels += [key for _ in range(size)]

    vectors = np.array(vectors)
    labels = np.array(labels)

    # save as npz file
    np.savez_compressed('{}_all'.format(name), X=vectors, y=labels)

# split data to train & test
def split_data(data, name='dataset', valid=0.1, test=0.1):
    train_data, train_label = [], []
    valid_data, valid_label = [], []
    test_data, test_label = [], []

    for key in data.keys():
        # shuffle content
        np.random.shuffle(data[key])
        size = len(data[key])
        t = int(floor(size * (1 - valid - test)))
        v = int(floor(size * (1 - valid)))
        # split to train & test
        train_data += data[key][:t]
        valid_data += data[key][t:v]
        test_data += data[key][v:]
        train_label += [key for _ in range(t)]
        valid_label += [key for _ in range(t, v)]
        test_label += [key for _ in range(size - t - v)]

    # save as npz file
    np.savez_compressed('{}_train'.format(name), X=train_data, y=train_label)
    np.savez_compressed('{}_valid'.format(name), X=valid_data, y=valid_label)
    np.savez_compressed('{}_test'.format(name), X=test_data, y=test_label)

# load all data
def load_all(name='dataset'):
    data = np.load('{}_all.npz'.format(name))
    return data

# load train & test data
def load_split(name='dataset'):
    train = np.load('{}_train.npz'.format(name))
    valid = np.load('{}_valid.npz'.format(name))
    test = np.load('{}_test.npz'.format(name))
    return train, valid, test

if __name__ == '__main__':
    # initialize parser
    parser = DocParser(nlp=parser,
                       useful_tags=useful_tags,
                       delimiters=delimiters,
                       min_token_length=min_token_length)

    # preprocessing
    data = preprocess(parser)
    save_data(data)
    split_data(data)

    # train word2vec on all the texts
    data = load_all()
    build_word2vec(data['X'])
