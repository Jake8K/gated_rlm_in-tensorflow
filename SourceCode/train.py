
#  train.py
#   training file for for gated rlm
#
#  Created by Jake K on 11/1/17.
#   Based on Gated RLM by Miyamoto and Cho
#       ==> The model description is here: https://arxiv.org/abs/1606.01700
#       ==> The base code is here: https://github.com/nyu-dl/dl4mt-tutorial
#   initialize_model(), initialize_vocab(), & print_samples() borrowed from Stanford CS224n Final Project

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from datetime import datetime
import json
import time
import random

import tensorflow as tf
import numpy as np
import argparse
from tqdm import tqdm
from gated_rlm import *
from os.path import join as pjoin

from base import Model, Progressbar
from data_reader import DataReader
from wordChar_prep import *
from data_preprocess import *
from gated_rlm import *
from layers import *

import logging

logging.basicConfig(level=logging.INFO)

tf.app.flags.DEFINE_integer("max_train_samples", 0, "Max number of training samples (0--load all).")
tf.app.flags.DEFINE_integer("max_val_samples", 0, "Max number of validation samples (0--load all).")
tf.app.flags.DEFINE_integer("embedding_size", 200, "Size of the pretrained GloVe embeddings.")
tf.app.flags.DEFINE_string("data_dir", "../data/", "shakespeare data directory (default ../data)")
tf.app.flags.DEFINE_string("train_dir", "train", "Training directory to save the model parameters (default: ./train).")
tf.app.flags.DEFINE_string("log_dir", "log", "Path to store log and flag files (default: ./log)")
tf.app.flags.DEFINE_integer("print_every", 10, "How many iterations to do per print.")
tf.app.flags.DEFINE_integer("keep", 0, "How many checkpoints to keep, 0 indicates keep all.")
tf.app.flags.DEFINE_string("word_vocab_path", "../data/MnC_dicts/word_dict.pkl", "Path to vocab file (default: ../data/MnC_dicts/word_dict.pkl)")
tf.app.flags.DEFINE_string("char_vocab_path", "../data/MnC_dicts/char_dict.pkl", "Path to vocab file (default: ../data/MnC_dicts/char_dict.pkl)")
tf.app.flags.DEFINE_string("embed_path", "", "Path to the trimmed GLoVe embedding (default: ./data/GloVe_vectors.trimmed.{vocab_dim}d.npz)")
tf.app.flags.DEFINE_boolean("restart", False, "[Do]/[Do not] attempt to restore the model.")
tf.app.flags.DEFINE_string("restore_path", "", "Directory Path to load model weights if Restart = TRUE. (default: /tmp/trainin_dir)")
tf.app.flags.DEFINE_integer("verify_only", 0, "Print N random samples and exit.")
tf.app.flags.DEFINE_boolean("check_embeddings", False, "Check embedding ids for our of bound conditions")

FLAGS = tf.app.flags.FLAGS


def initialize_model(session, model, train_dir):
    if FLAGS.restore_path and FLAGS.restart:
        ckpt = tf.train.get_checkpoint_state(FLAGS.restore_path)
    else:
        ckpt = tf.train.get_checkpoint_state(train_dir)

    v2_path = ckpt.model_checkpoint_path + ".index" if ckpt else ""''
    logging.info("Restart Flag = %s" % (FLAGS.restart))
    saver = tf.train.Saver()
    if FLAGS.restart and ckpt and (tf.gfile.Exists(ckpt.model_checkpoint_path) or tf.gfile.Exists(v2_path)):
        logging.info("Reading model parameters from %s" % ckpt.model_checkpoint_path)
        saver.restore(session, ckpt.model_checkpoint_path)
    else:
        logging.info("Created model with fresh parameters.")
        session.run(tf.global_variables_initializer())
        logging.info('Num params: %d' % sum(v.get_shape().num_elements() for v in tf.trainable_variables()))

    return model

def initialize_vocab(vocab_path): #seems super similar to the pre-processing in M and C's file
    if tf.gfile.Exists(vocab_path):
        rev_vocab = []
        with tf.gfile.GFile(vocab_path, mode="rb") as f:
            rev_vocab.extend(f.readlines())
        rev_vocab = [line.strip('\n') for line in rev_vocab]
        vocab = dict([(x, y) for (y, x) in enumerate(rev_vocab)])
        return vocab, rev_vocab
    else:
        raise ValueError("Vocabulary file %s not found.", vocab_path)

def load_training_data(data_path, max_vocab, max_samples=0):
    #prefix_path = pjoin(FLAGS.data_dir, prefix)

    char_path  = data_path+"MnC_dicts/char_dict.pkl"
    if not tf.gfile.Exists(char_path):
        raise ValueError("char vocab file %s not found.", char_path)

    word_path = data_path+"MnC_dicts/char_dict.pkl"
    if not tf.gfile.Exists(word_path):
        raise ValueError("word vocab file %s not found.", word_path)

    train_path = data_path+"train_tokenized.txt"
    if not tf.gfile.Exists(train_path):
        raise ValueError("training file %s not found.", train_path)

    eval_path = data_path+"valid_tokenized.txt"
    if not tf.gfile.Exists(eval_path):
        raise ValueError("development file %s not found.", eval_path)



    tic = time.time()
    logging.info("Loading SQUAD data from %s" % data_path)

    char_file = open(char_path, mode="rb")
    word_file = open(word_path, mode="rb")
    train_file = open(train_path, mode="rb")
    eval_file = open(eval_path, mode="rb")

    valid_range = range(0, max_vocab)

    train_data = []
    dev_data = []



    toc = time.time()
    logging.info("Complete: %d samples loaded in %f secs)" % (samples, toc - tic))
    logging.info("Question length histogram (10 in each bucket): %s" % str(c_buckets));
    logging.info("Context length histogram (100 in each bucket): %s" % str(q_buckets));
    logging.info("Median context length: %d" % len(data[counter//2][0]));

    return train_data, dev_data

def print_sample(sample, rev_vocab):
    print("Context:")
    print(" ".join([rev_vocab[s] for s in sample[0]]))
    print("Question:")
    print(" ".join([rev_vocab[s] for s in sample[1]]))
    print("Answer:")
    print(" ".join([rev_vocab[s] for s in sample[0][sample[2][0]:sample[2][1]+1]]))

def print_samples(data, n, rev_vocab):
    all_samples = range(len(data))
    for ix in random.sample( all_samples, n) if n > 0 else all_samples:
        print_sample(data[ix], rev_vocab)

def main(args):
    if args:
        restore = args

    embed_path = FLAGS.embed_path or "../data/GloVe_vectors.trimmed.200d.npz"
    embeddingz = np.load(embed_path)
    embeddings = embeddingz['glove']
    embeddingz.close()
    assert embeddings.shape[1] == 200 #(embedding size)

    vocab_len = embeddings.shape[0]

    vocab_path = "../data/MnC_dicts/word_dict.pkl"
    vocab, rev_vocab = initialize_vocab(vocab_path)

    with open( '../data/MnC_dicts/char_dict.pkl' , 'rb') as chars:
        char_dict = pkl.load(chars)
    with open( '../data/MnC_dicts/word_dict.pkl' , 'rb') as words:
        word_dict = pkl.load(words)
    train = load_file('../data/train_tokenized.txt')
    dev = load_file('../data/valid_tokenized.txt')

    data = []
    data.append(char_dict)
    data.append(word_dict)
    data.append(train)
    data.append(dev)

    #X_char, X_char_trash, X_mask, spaces, last_chars = prepare_char_data(text_to_char_index(input_txt, char_dict, '|'), text_to_char_index(input_txt, char_dict, '|'))
    #X_word, x_mask = prepare_word_data(text_to_word_index(input_txt, word_dict))

    if FLAGS.verify_only:
	print_samples(train, FLAGS.verify_only, rev_vocab)

        return

    global_train_dir = '/tmp/trainin_dir'
    # Adds symlink to {train_dir} from /tmp/cs224n-squad-train to canonicalize the
    # file paths saved in the checkpoint. This allows the model to be reloaded even
    # if the location of the checkpoint files has moved, allowing usage with CodaLab.
    # This must be done on both train.py and qa_answer.py in order to work.
    if os.path.exists(global_train_dir):
        os.unlink(global_train_dir)
    if not os.path.exists(FLAGS.train_dir):
        os.makedirs(FLAGS.train_dir)
    os.symlink(os.path.abspath(FLAGS.train_dir), global_train_dir)
    train_dir = global_train_dir

    #model = Gated_RNN_LM(sess, word_dict, char_dict, pretrained_embeddings=embeddings, word_tensors=X_word, char_tensors=X_char, max_word_length=20 )
    #model.build_model()

    if not os.path.exists(FLAGS.log_dir):
        os.makedirs(FLAGS.log_dir)
    file_handler = logging.FileHandler(pjoin(FLAGS.log_dir, "log.txt"))
    logging.getLogger().addHandler(file_handler)

    with open(os.path.join(FLAGS.log_dir, "flags.json"), 'w') as fout:
        json.dump(FLAGS.__flags, fout)

    with tf.Session() as sess:
        print("settings: optimizer: {}".format(FLAGS.optimizer))
        """{INIT}"""
        print("==> Building model:")
        model = Gated_LM_Model(train_dir, embeddings, word_dict, char_dict)
        model.build_model()
        print("==> Initializing model:")
        initialize_model(sess, model, train_dir)

        """{TRAIN}"""
        print("==> Training model:")
        model.train(sess, data) # (self, sess, dataset))

        """{EVAL}"""
        #model.evaluate(sess, model.prep_data(dev), log=True)
        #qa.evaluate_answer(sess, qa.preprocess_sequence_data(val), log=True)

if __name__ == "__main__":
    tf.app.run()
