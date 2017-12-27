
#  wordChar_prep.py
#
#   corpus_tokenizer(), dict_count() created by Jake K on 11/1/17.
#   Vocab class, get_input_data() by Taehoon Kim (https://github.com/carpedm20/lstm-char-cnn-tensorflow)
#   basic_tokenizer(), create_vocabulary(), initialize_vocabulary(), process_glove() by Ethan Socher (Stanford CS224n)
#   clean_str() by Yoon Kim (https://github.com/yoonkim/CNN_sentence/blob/master/process_data) modified by Jacob Karcz


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gzip
import os
import re
import tarfile
import argparse
import codecs
import collections
from tensorflow.python.platform import gfile
import cPickle as pickle
from tqdm import *
import numpy as np
from os.path import join as pjoin

# Data Pre-Processing Script
    #Useful Resources
        # HOW TO GUIDE!!!!!! YAYAYAYAYAYYA
                # https://ireneli.eu/2017/01/17/tensorflow-07-word-embeddings-2-loading-pre-trained-vectors/
                    # also https://stackoverflow.com/questions/35687678/using-a-pre-trained-word-embedding-word2vec-or-glove-in-tensorflow
        # M & C prep:
        # assignment 4 prep:
        # CNN-LSTM prep:
        #  map dict to embeds: https://stackoverflow.com/questions/38665556/matching-words-and-vectors-in-gensim-word2vec-model
        # Keras LSTM w GloVe: https://www.kaggle.com/lystdo/lb-0-18-lstm-with-glove-and-magic-features
        # GloVe in TF??? https://github.com/GradySimon/tensorflow-glove
        #  Fundamentals of Deep Learning Ch.6 (146), Ch. 7 (pp 194, 153, 185)
        #  Hands on Machine Learning pp 421 - 428
        #  Intro to ML Python Ch 8 (mostly starting p 314)
_PAD = b"<pad>"
_SOS = b"<sos>"
_UNK = b"<unk>"
_START_VOCAB = [_PAD, _SOS, _UNK]

PAD_ID = 0
SOS_ID = 1
UNK_ID = 2


class Vocab:

    def __init__(self, token2index=None, index2token=None):
        self._token2index = token2index or {}
        self._index2token = index2token or []

    def feed(self, token):
        if token not in self._token2index:
            # allocate new index for this token
            index = len(self._token2index)
            self._token2index[token] = index
            self._index2token.append(token)

        return self._token2index[token]

    @property
    def size(self):
        return len(self._token2index)

    def token(self, index):
        return self._index2token[index]

    def __getitem__(self, token):
        index = self.get(token)
        if index is None:
            raise KeyError(token)
        return index

    def get(self, token, default=None):
        return self._token2index.get(token, default)

    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump((self._token2index, self._index2token), f, pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, filename):
        with open(filename, 'rb') as f:
            token2index, index2token = pickle.load(f)

        return cls(token2index, index2token)


def get_input_data(data_dir, max_word_length, eos='+'):

    char_vocab = Vocab()
    char_vocab.feed(' ')  # blank is at index 0 in char vocab
    char_vocab.feed('{')  # start is at index 1 in char vocab
    char_vocab.feed('}')  # end   is at index 2 in char vocab

    word_vocab = Vocab()
    word_vocab.feed('|')  # <unk> is at index 0 in word vocab

    actual_max_word_length = 0

    word_tokens = collections.defaultdict(list)
    char_tokens = collections.defaultdict(list)

    for fname in ('train', 'valid', 'test'):
        print('reading', fname)
        with codecs.open(os.path.join(data_dir, fname + '_tokenized.txt'), 'r', 'utf-8') as f:
            for line in f:
                line = line.strip()
                line = line.replace('}', '').replace('{', '').replace('|', '')
                line = line.replace('<unk>', ' | ') # why do they use the bos symbol and why <unk> tags!!!?!?!?!??!
                if eos:
                    line = line.replace(eos, '')

                for word in line.split():
                    if len(word) > max_word_length - 2:  # space for 'start' and 'end' chars
                        word = word[:max_word_length-2]

                    word_tokens[fname].append(word_vocab.feed(word))

                    char_array = [char_vocab.feed(c) for c in '{' + word + '}']
                    #print(char_array)
                    char_tokens[fname].append(char_array)

                    actual_max_word_length = max(actual_max_word_length, len(char_array))

                if eos:
                    word_tokens[fname].append(word_vocab.feed(eos))

                    char_array = [char_vocab.feed(c) for c in '{' + eos + '}']
                    char_tokens[fname].append(char_array)

    assert actual_max_word_length <= max_word_length

    print()
    print('actual longest token length is:', actual_max_word_length)
    print('size of word vocabulary:', word_vocab.size)
    print('size of char vocabulary:', char_vocab.size)
    print('number of tokens in train:', len(word_tokens['train']))
    print('number of tokens in valid:', len(word_tokens['valid']))
    print('number of tokens in test:', len(word_tokens['test']))

    # now we know the sizes, create tensors
    word_tensors = {}
    char_tensors = {}
    for fname in ('train', 'valid', 'test'):
        assert len(char_tokens[fname]) == len(word_tokens[fname])

        word_tensors[fname] = np.array(word_tokens[fname], dtype=np.int32)
        char_tensors[fname] = np.zeros([len(char_tokens[fname]), actual_max_word_length], dtype=np.int32)

        for i, char_array in enumerate(char_tokens[fname]):
            char_tensors[fname] [i,:len(char_array)] = char_array

    return word_vocab, char_vocab, word_tensors, char_tensors, actual_max_word_length

def basic_tokenizer(sentence):
    words = []
    for space_separated_fragment in sentence.strip().split():
        words.extend(re.split(" ", space_separated_fragment))
    return [w for w in words if w]

def corpus_tokenizer(corpus, destination):
    words = []
    with gfile.GFile(destination, mode="wb") as tokenized_file:
        with open(corpus, mode='rb') as fp:
            for line in fp:
                for space_separated_fragment in line.strip().split():
                    #shorter than the if bellow https://stackoverflow.com/questions/3645931/python-padding-punctuation-with-white-spaces-keeping-punctuation
                    #if '.' or ',' or ';' or ':' or '?' or '!' or '\'' or '\'s' or '\'ve' '\'ll' in space_separated_fragment:
                        #tokenized_file.write(words.extend(re.split(" ", space_separated_fragment)) + b"\n")
                    tokenized_file.write(clean_str(space_separated_fragment)+b' ')
                tokenized_file.write('\n')

def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original mostly taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data
    more info: https://stackoverflow.com/questions/3645931/python-padding-punctuation-with-white-spaces-keeping-punctuation
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " ( ", string)
    string = re.sub(r"\)", " ) ", string)
    string = re.sub(r"\?", " ? ", string)
    string = re.sub(r"\s{2,}", " ", string) #what does this one do?
    return string.strip()

def dict_count(dict_path):
    with open(dict_path, mode="rb") as f:
        count=0
        for line in f:
            count += 1
    return count

def create_vocabulary(vocabulary_path, data_paths, tokenizer=None):
    if not gfile.Exists(vocabulary_path):
        print("Creating vocabulary %s from data %s" % (vocabulary_path, str(data_paths)))
        vocab = {}
        for path in data_paths:
            with open(path, mode="rb") as f:
                counter = 0
                for line in f:
                    counter += 1
                    if counter % 100000 == 0:
                        print("processing line %d" % counter)
                    tokens = tokenizer(line) if tokenizer else basic_tokenizer(line)
                    for w in tokens:
                        if w in vocab:
                            vocab[w] += 1
                        else:
                            vocab[w] = 1
        vocab_list = _START_VOCAB + sorted(vocab, key=vocab.get, reverse=True)
        print("Vocabulary size: %d" % len(vocab_list))
        with gfile.GFile(vocabulary_path, mode="wb") as vocab_file:
            for w in vocab_list:
                vocab_file.write(w + b"\n")

def initialize_vocabulary(vocabulary_path):
    # map vocab to word embeddings
    if gfile.Exists(vocabulary_path):
        rev_vocab = []
        with gfile.GFile(vocabulary_path, mode="r") as f:
            rev_vocab.extend(f.readlines())
        rev_vocab = [line.strip('\n') for line in rev_vocab]
        vocab = dict([(x, y) for (y, x) in enumerate(rev_vocab)])
        return vocab, rev_vocab
    else:
        raise ValueError("Vocabulary file %s not found.", vocabulary_path)

def process_glove(glove_dim, vocab_list, save_path, size=4e5, random_init=True):
    """ saves a file containing only the vectors that co-occur in vocab and vector files
    :param vocab_list: [vocab]
    :return:
    """
    if not gfile.Exists(save_path + ".npz"):
        glove_path = "../GloVe/shakespeare_d200/vectors.txt"
        #glove_path = "../GloVe/wikipedia/glove.6B.200d.txt"
        #glove_path = "../GloVe/web_crawl/glove.840B.300d.txt"
        if random_init:
            #glove = np.random.randn(len(vocab_list), glove_dim)
            glove = np.random.randn(67062, glove_dim)
        else:
            glove = np.zeros((len(vocab_list), glove_dim))
        found = 0
        with open(glove_path, 'r') as fh:
            for line in tqdm(fh, total=size):
                array = line.lstrip().rstrip().split(" ") #!
                word = array[0]
                vector = list(map(float, array[1:]))
                if word in vocab_list:
                    idx = vocab_list.index(word)
                    glove[idx, :] = vector
                    found += 1
                if word.capitalize() in vocab_list:
                    idx = vocab_list.index(word.capitalize())
                    glove[idx, :] = vector
                    found += 1
                if word.upper() in vocab_list:
                    idx = vocab_list.index(word.upper())
                    glove[idx, :] = vector
                    found += 1

        #print("{}/{} of word vocab have corresponding vectors in {}".format(found, len(vocab_list), glove_path))
        print("{}/{} of word vocab have corresponding vectors in {}".format(found, 67062, glove_path))
        np.savez_compressed(save_path, glove=glove)
        print("saved trimmed glove matrix at: {}".format(save_path))


""" {GloVe Init} """
# check out qa_train.py, first few lines of main


""" {char init} """

""" {Word init} """

#match vocab to embeddings... how?
