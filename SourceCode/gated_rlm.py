#  Gated_LSTM_model.py
#
#
#  Created by Jake K on 11/1/17.
#   Based on Gated RLM by Miyamoto and Cho
#       ==> The model description is here: https://arxiv.org/abs/1606.01700
#       ==> The base code is here: https://github.com/nyu-dl/dl4mt-tutorial
#   helper functions borrowed from CS224n assignment 4 (and modified or unused)
#

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
from datetime import datetime
from random import shuffle
import logging
import sys #Progbar
import os
import cPickle as pkl


import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
from tensorflow.contrib import rnn
from sklearn.cross_validation import KFold
from tensorflow.python.ops import variable_scope as vs

from base import Model, Progressbar, variable_summaries, mask_softmax
from layers import     bidirectional_lstm, GloVe, gate, lstm_lm
from wordChar_prep import *
from data_preprocess import *

tf.app.flags.DEFINE_integer("epochs", 100, "Number of epochs to train.")
#tf.app.flags.DEFINE_float  ("cross_id_bias", -1, "ID coefficient to init attention multiplier matrix. (Use -1 to avoid using matrix altogether)")
tf.app.flags.DEFINE_integer("maxlen", 60, "max seq length (est line length)")
tf.app.flags.DEFINE_integer("dim_word", 200, "Word Vector Length")
tf.app.flags.DEFINE_integer("dim_char", 200, "Char Vector Length")
tf.app.flags.DEFINE_integer("char_lstm_size", 200, "Hidden Units in bidirectional LSTM.")
tf.app.flags.DEFINE_integer("lstm_lm_size", 200, "Hidden Units in Language Modeling LSTM.")

tf.app.flags.DEFINE_boolean("log_losses", False, "Collect batch losses for plotting.")
tf.app.flags.DEFINE_string ("optimizer", "adam", "adam / sgd / [rmsProp?]")
tf.app.flags.DEFINE_float  ("learning_rate", 0.01, "Learning rate.") #for some reason 1 in M&C
tf.app.flags.DEFINE_float  ("learning_rate_decay", 0.9999, "Learning rate.") # 2.1 in M n C
tf.app.flags.DEFINE_integer("lr_decay_start", 7, "Learning rate.") # 2.1 in M n C
tf.app.flags.DEFINE_float  ("gradient_clipping", 5.0, "Learning rate.") # 2.1 in M n C
tf.app.flags.DEFINE_integer("pretrain", 2, "(in Gate) the first m epochs: word only, the next m epochs: char only, 0 to disable")
tf.app.flags.DEFINE_integer("is_train", 2, "flag for train(1.) or test(0.)")
tf.app.flags.DEFINE_integer("patience", 3, "for early stopping")

tf.app.flags.DEFINE_boolean("train_embeddings", False, "Train embedding vectors")
tf.app.flags.DEFINE_float  ("dropout", 0.50, "Fraction of units randomly dropped on non-recurrent lstm-lm connections.")

tf.app.flags.DEFINE_boolean("evaluate_epoch", True, "Run full EM/F1 evaluation at the end of each epoch.")
tf.app.flags.DEFINE_boolean("save_epoch", True, "Auto-save the model at the end of each epoch.")
tf.app.flags.DEFINE_integer("seve_freq", 10000, "Save Params every x updates.")
tf.app.flags.DEFINE_integer("valid_freq", 10000, "Validate every y updates.")
tf.app.flags.DEFINE_string ("saveto", "gate_word_char_p_shkspr.npz", "newest model (default: gate_word_char_p_shkspr.npz).")
tf.app.flags.DEFINE_string ("savebestto", "gate_word_char_p_shkspr_best.npz", "best model (default: gate_word_char_p_shkspr_best.npz).")

    #opts['saveto'] == FLAGS.train_dir

tf.app.flags.DEFINE_integer("batch_size", 32, "Batch size to use during training.")
tf.app.flags.DEFINE_float  ("max_gradient_norm", 5.0, "Clip gradients to this norm.") #VERIFY

tf.app.flags.DEFINE_string("train_data", "../data/train_tokenized.txt", "path to training data (default: ../data/train_tokenized.txt).")
tf.app.flags.DEFINE_string("dev_data", "../data/valid_tokenized.txt", "path to validation data (default: ../data/valid_tokenized.txt).")
tf.app.flags.DEFINE_string("test_data", "../data/test_tokenized.txt", "path to test data (default: ../data/test_tokenized.txt).")

#tf.app.flags.DEFINE_string("word_vocab_path", "../data/MnC_dicts/word_dict.pkl", "Path to word vocab file (default: ../data/MnC_dicts/word_dict.pkl)")
#tf.app.flags.DEFINE_string("char_vocab_path", "../data/MnC_dicts/char_dict.pkl", "Path to char vocab file (default: ../data/MnC_dicts/char_dict.pkl)")

logging.basicConfig(level=logging.INFO)

FLAGS = tf.app.flags.FLAGS



class Gated_LM_Model(object):
    def __init__(self, train_dir, glove, word_dict, char_dict):
        self.maxlen = FLAGS.maxlen #longest sequence
        self.batch_size = FLAGS.batch_size
        self.pretrained_embeddings = glove
        self.dim_word = FLAGS.dim_word
        self.dim_char = FLAGS.dim_char
        self.words = word_dict
        self.chars = char_dict
        print("initializing model. maxlen: {}, batch_size: {}, dim_word: {}, dim_char {}".format(self.maxlen, self.batch_size, self.dim_word, self.dim_char))
        print("                    embeddings shape: [{}x{}], char vocab: {}, word vocab: {}".format(self.pretrained_embeddings.shape[0] ,self.pretrained_embeddings.shape[1], len(self.chars), len(self.words)))
        """Add placeholder variables to tensorflow computational graph.
        See for more information:
        https://www.tensorflow.org/versions/r0.7/api_docs/python/io_ops.html#placeholders
        """
        #self.X_char   = tf.placeholder(dtype=tf.int64, shape=[None, self.maxlen], name="X_char")
        #self.X_word   = tf.placeholder(dtype=tf.int64, shape=[None, self.maxlen], name="X_word")
        #self.X_spaces = tf.placeholder(dtype=tf.float32, shape=[None, self.maxlen], name="spaces")
        #self.X_last_chars = tf.placeholder(dtype=tf.float32, shape=[None, self.maxlen], name="last_chars")
        #self.label_words =  tf.placeholder(dtype=tf.float32, shape=[None, self.maxlen], name="label_words")

        #self.seq_len_placeholder = tf.placeholder(tf.int64, [self.maxlen], "batch_vector") #char_batch_len_vector
        #print ("char_batch_vector dims: ", self.seq_len_placeholder.get_shape().as_list()) #[10,000, 200]
        # NOTE: TO DO! self.n_timesteps = max_time = longest_seq?
        # NOTE: TO DO! self.n_samples = ????
        # OBJECTIVE: reshape all tensors to be dim
        """[n_time_steps, n_samples (batch_size), tensor_dim (ie dim_char)]"""

        """can I use ints or must they be shared tensor variables? (don't forget to do reuse=T in scope!)"""
        self.is_train = 0. #tf.Variable(np.float32(0.))        # why
        self.pretrain_mode = 0. # tf.Variable(np.float32(0.)) # line 348 of word_char_lm.py

    def build_model(self):
    #with tf.variable_scope("Gated_RNN-LM"):
        print ("==> Building Model...")
        #print ("word vocab: ")
        #print (self.words)
        #print ("char vocab: ")
        #print (self.chars)

        # ==== set up placeholder tokens ========
        #char in
        self.X_char = tf.placeholder(tf.float32, [self.batch_size, self.maxlen, self.dim_char], "X_char") #[batch, seq, max_word_len]
        print ("\tchar_input dims: ", self.X_char.get_shape().as_list())

        #word in
        self.X_word = tf.placeholder(tf.int64, [self.batch_size, self.maxlen], "X_word") #[batch, seq, max_word_len]
        #self.one_hot_word=tf.one_hot(self.word_input, self.n_words)
        print ("\tword_input dims: ", self.X_word.get_shape().as_list())

        self.seq_len_placeholder = tf.placeholder(tf.int64, [self.maxlen], "seq_len_placeholder")
        self.word_labels_placeholder = tf.placeholder(dtype=tf.int64, shape=[None,2], name='word_labels') #aka span?
        self.pred_len_placeholder = tf.placeholder(tf.float32, [self.batch_size], "pred_len")

        """
        self.char_input = tf.placeholder(dtype=tf.int64, name="X_char")
        self.n_timesteps = 201 #self.char_input.get_shape()[1] #bc tf is batch major (not time-major) by default
        self.n_samples = 36 #self.char_input.get_shape()[0]
        #print ("char_input dims: ", self.char_input.get_shape().as_list())

        self.word_input = tf.placeholder(dtype=tf.int64, name="X_word")
        #print ("word_input dims: ", self.word_input.get_shape().as_list()) #[32, None, 200]
        """

        #self.label_words = tf.placeholder(tf.int64, [self.batch_size, self.sequence_length, self.dim_word], "Y_hat")


        #char_W = tf.get_variable("char_embed", [self.n_char, self.dim_char]) #[self.char_vocab_size, self.char_embed_dim])
        #print ("char_W dims: ", char_W.get_shape().as_list()) #[51, 200]
        #word_W = tf.get_variable("word_embed", [self.n_words, self.dim_word])
        #print ("word_W dims: ", word_W.get_shape().as_list()) #[10,000, 200]

        ###################
        """ {char LSTM} """ #https://github.com/tensorflow/tensorflow/issues/799
        ###################
        #char_batch_vector = tf.placeholder(tf.int64, [self.dim_char], "batch_vector")
        print ("\tchar_batch_vector dims: ", self.seq_len_placeholder.get_shape().as_list()) #[10,000, 200]
        self.Cemb = bidirectional_lstm(self=self, char_dict=self.chars, batch_tensor=self.seq_len_placeholder, n_hidden=200, X_char=self.X_char)

        ######################
        """ {Word Look Up} """
        #####################
        self.Wemb = GloVe(self=self, X_word=self.X_word, pretrained_embeds=self.pretrained_embeddings )

        ##############
        """{ GATE }"""
        ##############
        #try simply concatenating Wemb and Cemb first
        self.X_wt = gate(self=self, word_emb=self.Wemb, char_emb=self.Cemb, pretrain_mode=self.pretrain_mode)

        ##################
        """ {LSTM LM } """
        ##################
        self.final_output, self.final_state = lstm_lm(self=self, input_X=self.X_wt, n_hidden=FLAGS.lstm_lm_size, num_layers=2)
        #test crap        self.word = lstm_lm(self=self, lm_input=self.Wemb, n_hidden=self.lstm_lm_size, num_layers=2)

        ###################
        """ { Softmax } """
        ###################
        state_size = FLAGS.lstm_lm_size # * 4 #(num_units * num_gates )
        num_classes = self.dim_char
        with tf.variable_scope('softmax') as scope:
            W = tf.get_variable('W', [state_size, num_classes])
            b = tf.get_variable('b', [num_classes], initializer=tf.constant_initializer(0.0))
        self.softmax = tf.nn.softmax(self.final_output, dim=1)
        self.logits = tf.reshape(tf.matmul(tf.reshape(self.final_output, [-1, state_size]), W) + b, [self.batch_size, self.maxlen, self.dim_char]) #[batch_size, num_steps, num_classes])
        self.prediction = tf.nn.softmax(self.logits)

        variable_summaries(self.prediction)
        self.prediction = mask_softmax(self.prediction, 1, self.pred_len_placeholder)
        self.softmax = tf.nn.softmax(self.prediction, dim=1)

        #multiword = tf.argmax( self.softmax, 1 )
        #sngleword = tf.argmax( tf.reduce_prod( self.softmax, 2, keep_dims=True ), 1)
        #start = tf.slice(multiword, [0,0], [-1,1])
        #end   = tf.slice(multiword, [0,1], [-1,1])
        #conflicts = tf.greater(start, end)
        #self.result = tf.to_int32( tf.where( tf.tile(conflicts, [1,2]), tf.tile(sngleword, [1,2]), multiword ))
        self.result =  tf.argmax( self.softmax, 1 )


        #########################
        """ { Optimization } """
        ########################
        #https://www.tensorflow.org/versions/r0.12/api_docs/python/train/optimizers
        # TODO: Word_LableS!!!!!!!!!!!!!

        # NOTE: self.losses, self.total_loss, and self.self.train_step...minimize() go together
        #NOTE: selt.setup_loss goes with impimizer and gradient clipping

        self.losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.word_labels_placeholder, logits=self.logits)
        self.total_loss = tf.reduce_mean(self.losses)

        self.setup_loss()

        step = tf.Variable(0, trainable=False)
        rate = tf.train.exponential_decay(FLAGS.learning_rate, step, 1, FLAGS.learning_rate_decay)

        if FLAGS.optimizer is 'sgd':
            # SGD #https://www.tensorflow.org/api_docs/python/tf/train/GradientDescentOptimizer
            self.train_step = tf.train.GradientDescentOptimizer(FLAGS.learning_rate).minimize(self.total_loss)
            optimizer = tf.train.GradientDescentOptimizer(rate)

        elif FLAGS.optimizer is 'adam':
            # Adam #https://www.tensorflow.org/versions/r1.1/api_docs/python/tf/train/AdamOptimizer
            self.train_step = tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(self.total_loss) #add lr decay?
            optimizer = tf.train.AdamOptimizer(rate)

        elif FLAGS.optimizer is 'adagrad':
            # Adagrad #https://www.tensorflow.org/versions/r1.1/api_docs/python/tf/train/AdagradOptimizer
            self.train_step = tf.train.AdagradOptimizer(FLAGS.learning_rate).minimize(self.total_loss)
            optimizer = tf.train.AdagradOptimizer(rate)

        grads, vars = zip(*optimizer.compute_gradients(self.loss))
        if FLAGS.max_gradient_norm:
            capped_grads, self.grad_norm = tf.clip_by_global_norm(grads, FLAGS.max_gradient_norm)
        else:
            capped_grads = grads
            self.grad_norm = tf.global_norm(grads)

        self.train_op = optimizer.apply_gradients(zip(capped_grads, vars))


        """ [To Do notes from M & C] ""
        # mask for final loss: 1 if the last char of a word, 0 otherwise
        final_loss_mask = 1 - x_last_chars
        final_loss_mask = final_loss_mask.flatten()

        # cost
        x_flat = label_words.flatten()
        x_flat_idx = tensor.arange(x_flat.shape[0]) * options['n_words'] + x_flat
        cost = -tensor.log(probs.flatten()[x_flat_idx] + 1e-8) * final_loss_mask # only last chars of words
        cost = cost.reshape([x_f.shape[0], x_f.shape[1]]) # reshape to n_steps x n_samples
        cost = cost.sum(0)                                # sum up NLL of words in a sentence
        cost = cost.mean()                                # take mean of sentences
        "" [END To Do from M & C] """

        """ {{{{ check out qa_model.py lines 517 to 534 }}}} """

    def setup_loss(self):
        #by Ethan Socher (Professor, CS224n)
        """
        Set up your loss computation here
        :return:
        """
        #print("  ==> setting up cost")
        max_encoded = self.softmax.get_shape().as_list()[1]
        print("\tsoftmax shape is: {}".format(max_encoded))
        with vs.variable_scope("loss"):
            print("\tmodel's final output shape: {}".format(self.final_output.get_shape().as_list()))
            #assert self.final_output.get_shape().as_list() == [None, max_encoded, 2]
            logits = tf.transpose(self.final_output, perm=[0, 2, 1])
            print("\tlogits: {}".format(self.final_output.get_shape().as_list()))
            #assert logits.get_shape().as_list() == [None, 2, max_encoded]

            #assert self.span_placeholder.get_shape().as_list() == [None, 2]
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=self.word_labels_placeholder)
            self.loss = tf.reduce_mean(losses)

            #tf.summary.scalar('cross_entroy_loss',self.loss)
        """
        with vs.variable_scope("em"):
            ""
            in M&C
            x_flat_idx = tensor.arange(x_flat.shape[0]) * options['n_words'] + x_flat
            cost = -tensor.log(probs.flatten()[x_flat_idx] + 1e-8) * final_loss_mask # only last chars of words
            cost = cost.reshape([x_f.shape[0], x_f.shape[1]]) # reshape to n_steps x n_samples
            cost = cost.sum(0)                                # sum up NLL of words in a sentence
            cost = cost.mean()                                # take mean of sentences
            ""

            eq = tf.equal( self.result, self.word_labels_placeholder )
            assert eq.get_shape().as_list() == [None, 2]

            both = tf.reduce_min( tf.to_float(eq), 1)
            assert both.get_shape().as_list() == [None]

            self.em = tf.reduce_mean(both)
            tf.summary.scalar('EM_Accuracy', self.em)
            """
            #setup tensorboard
        self.summary = tf.summary.merge_all()



    def optimize(self, session, x_f, x_r, x_spaces, x_last_chars, x_word_input, label_words):
        #by Ethan Socher (Professor, CS224n) edited by Jacob Karcz
        """
        Takes in actual data to optimize your model
        Equivalent to a feed-dict / step() function
        :return:
        """
        input_feed = {
            self.X_char : x_f,  #[batch, seq, char-dim]
            self.X_word : x_word_input, #[batch, seq, word-dim]
            self.seq_len_placeholder : x_spaces, #? will that work????
            self.word_labels_placeholder: label_words

            #self.encoder.c_ids_placeholder: c_ids,
            #self.encoder.q_ids_placeholder: q_ids,

            #self.c_len_placeholder: c_len,
            #self.q_len_placeholder: q_len,

            #self.dropout_placeholder: FLAGS.dropout,#TODO???

            #self.span_placeholder : span,
        }

        output_feed = [self.train_op, self.loss, self.total_loss, self.summary] #TODO??? gradient and param norm???
        #output_feed = [self.train_op, self.loss, self.em, self.summary]

        outputs = session.run(output_feed, input_feed)

        return outputs[1:] # All but the optimizer

    def test(self, session, valid):
        #by Ethan Socher (Professor, CS224n)
        """
        in here you should compute a cost for your validation set
        and tune your hyperparameters according to the validation set performance
        :return:
        """
        c_ids, c_len, q_ids, q_len, span = valid

        input_feed = {
            self.X_char : x_f,  #[batch, seq, char-dim]
            self.X_word : x_word_input, #[batch, seq, word-dim]
            self.seq_len_placeholder : x_spaces, #? will that work????
            self.word_labels_placeholder: label_words
            #self.encoder.c_ids_placeholder: c_ids,
            #self.encoder.q_ids_placeholder: q_ids,

            #self.c_len_placeholder: c_len,
            #self.q_len_placeholder: q_len,

            #self.dropout_placeholder: 0, #TODO???

            #self.span_placeholder : span,
        }

        output_feed = [self.loss, self.em] #TODO??? gradient and param norm???

        outputs = session.run(output_feed, input_feed)

        return outputs


    def decode(self, session, test_x):
        #by Ethan Socher (Professor, CS224n)
        """
        Returns the probability distribution over different positions in the paragraph
        so that other methods like self.answer() will be able to work properly
        :return:
        """
        c_ids, c_len, q_ids, q_len = test_x

        input_feed = {
            self.encoder.c_ids_placeholder: [c_ids],
            self.encoder.q_ids_placeholder: [q_ids],

            self.c_len_placeholder: [c_len],
            self.q_len_placeholder: [q_len],

            self.dropout_placeholder: 0,
        }

        output_feed = [self.result]

        outputs = session.run(output_feed, input_feed)
        return outputs

    def answer(self, session, test_x):
        #by Ethan Socher (Professor, CS224n)
        a = self.decode(session, test_x)
        a_s = a[0][0][0]
        a_e = a[0][0][1]
        # a_s = np.argmax(yp, axis=1)
        # a_e = np.argmax(yp2, axis=1)

        return (a_s, a_e)

    def validate(self, session, valid_dataset):
        #by Ethan Socher (Professor, CS224n)
        """
        Iterate through the validation dataset and determine what
        the validation cost is.

        This method calls self.test() which explicitly calculates validation cost.

        How you implement this function is dependent on how you design
        your data iteration function

        :return:
        """
        loss_validation = 0

        batch_count = (len(valid_dataset[0]) + FLAGS.batch_size-1) // FLAGS.batch_size
        prog = Progbar(target=batch_count)
        losses = []
        # run over the minibatch size for validation dataset
        for i, batch in enumerate(get_minibatches(valid_dataset, FLAGS.batch_size)):
            loss_validation, em_validation = self.test(session, batch)

            losses.append([loss_validation, em_validation])
            prog.update(i + 1, [("Validation loss", loss_validation), ("Validation em", em_validation*100)])

        mean = np.mean(losses, axis = 0)
        logging.info("Validation logged mean: loss : %f, EM = %f %%", mean[0], mean[1]*100)

        return losses

    def evaluate_answer(self, session, dataset, samples = 100, log=False):
        #by Ethan Socher (Professor, CS224n)
        """
        Evaluate the model's performance using the harmonic mean of F1 and Exact Match (EM)
        with the set of true answer labels

        This step actually takes quite some time. So we can only sample 100 examples
        from either training or testing set.

        :param session: session should always be centrally managed in train.py
        :param dataset: a representation of our data, in some implementations, you can
                        pass in multiple components (arguments) of one dataset to this function
        :param log: whether we print to std out stream
        :return:
        """
        samples = min(samples, len(dataset[0]))

        c_ids, c_len, q_ids, q_len, span = dataset

        f1 = 0.
        em = 0.

        for index in range(samples):
            a_s, a_e = self.answer(session, (c_ids[index], c_len[index], q_ids[index], q_len[index]))
            answers = c_ids[index][a_s: a_e+1]
            p_s, p_e = span[index]
            true_answer = c_ids[index][p_s: p_e+1]

            answers = " ".join(str(a) for a in answers)
            true_answer = " ".join(str(ta) for ta in true_answer)

            f1 += f1_score(answers, true_answer)
            em += exact_match_score(' '.join(str(a) for a in answers), ' '.join(str(ta) for ta in true_answer))
            #logging.info("answers %s, true_answer %s" % (answers, true_answer))

        f1 /=samples
        em /=samples

        if log:
            logging.info("F1: {:.2%}, EM: {:.2%}, for {} samples".format(f1, em, samples))

        return f1, em

    def preprocess_sequence_data(self, dataset):
        #by Ethan Socher (Professor, CS224n)
        max_c = FLAGS.max_c
        max_q = FLAGS.max_q

        stop = next(( idx for idx, xi in enumerate(dataset) if len(xi[0])>max_c), len(dataset))
        assert len(dataset[stop-1][0])<=max_c

        c_ids = np.array([xi[0]+[0]*(max_c-len(xi[0])) for xi in dataset[:stop]], dtype=np.int32)
        q_ids = np.array([xi[1]+[0]*(max_q-len(xi[1])) for xi in dataset[:stop]], dtype=np.int32)

        span = np.array([xi[2] for xi in dataset[:stop]], dtype=np.int32)

        c_len  = np.array([len(xi[0]) for xi in dataset[:stop]], dtype=np.int32)
        q_len  = np.array([len(xi[1]) for xi in dataset[:stop]], dtype=np.int32)

        data_size = c_ids.shape[0]

        assert q_ids.shape[0] == data_size
        assert  c_ids.shape == (data_size, max_c)
        assert  q_len.shape == (data_size,)
        assert  c_len.shape == (data_size,)
        assert   span.shape == (data_size,2)

        return [c_ids, c_len, q_ids, q_len, span]


    def run_epoch(self, session, epoch, writer, train, dev):
        #by Ethan Socher (Professor, CS224n)
        batch_count = (len(train[0]) + FLAGS.batch_size-1) // FLAGS.batch_size
        prog = Progbar(target=batch_count)
        losses = []
        # run over the minibatch size
        for i, batch in enumerate(get_minibatches(train, FLAGS.batch_size)):
            loss_train, em_train, summary = self.optimize(session, batch)

            writer.add_summary(summary, epoch*batch_count+i)

            loss_dev, em_dev = self.test(session, dev)

            losses.append([loss_train, loss_dev])
            prog.update(i + 1, [("train loss", loss_train), ("train em", em_train*100), ("dev loss", loss_dev), ("dev em", em_dev*100)])

        mean = np.mean(losses, axis = 0)
        logging.info("Logged mean epoch losses: train : %f dev : %f ", mean[0], mean[1])

        return losses

    def train_squad(self, session, dataset):
        #by Ethan Socher (Professor, CS224n)
        """
        Implement main training loop

        TIPS:
        You should also implement learning rate annealing (look into tf.train.exponential_decay)
        Considering the long time to train, you should save your model per epoch.

        More ambitious appoarch can include implement early stopping, or reload
        previous models if they have higher performance than the current one

        As suggested in the document, you should evaluate your training progress by
        printing out information every fixed number of iterations.

        We recommend you evaluate your model performance on F1 and EM instead of just
        looking at the cost.

        :param session: it should be passed in from train.py
        :param dataset: a representation of our data, in some implementations, you can
                        pass in multiple components (arguments) of one dataset to this function
        :return:
        """

        # some free code to print out number of parameters in your model
        # it's always good to check!
        # you will also want to save your model parameters in self.train_dir
        # so that you can use your trained model to make predictions, or
        # even continue training

        tic = time.time()
        params = tf.trainable_variables()
        num_params = sum(map(lambda t: np.prod(tf.shape(t.value()).eval()), params))
        toc = time.time()

        results_path = "results/{:%Y%m%d_%H%M%S}/".format(datetime.now())
        model_path = results_path + "model.weights/"
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        saver2 = tf.train.Saver()
        saver = tf.train.Saver()
        logging.info("Number of params: %d (retreival took %f secs)" % (num_params, toc - tic))

        corpus = self.preprocess_sequence_data(dataset)

        train, valid = list( get_minibatches(corpus, len(corpus[0])*9//10))
        dev =  get_minibatches(valid, 16).next()

        train_writer = tf.summary.FileWriter(FLAGS.log_dir + '/train', session.graph)
        val_writer   = tf.summary.FileWriter(FLAGS.log_dir + '/val')

        # run the number of epochs for the train
        losses = []
        losses_validation = []
        best_f1 = 0

        for epoch in range(FLAGS.epochs):
            logging.info("Epoch %d out of %d", epoch + 1, FLAGS.epochs)
            loss = self.run_epoch(session, epoch, train_writer, train, dev)

            if FLAGS.evaluate_epoch:
                logging.info("Starting Answer Evaluation")
                f1, em = self.evaluate_answer(session, valid, 100, log=True)

                logging.info("Starting data validation step")
                loss_validation = self.validate(session, valid)

            if f1 > best_f1:
                logging.info("New Best F1 Score!!! %f", f1*100)
                best_f1 = f1
                if FLAGS.save_epoch:
                    logging.info("Checkpoint Saved %s %s" %(self.train_dir, model_path))
                    saver.save(session, self.train_dir+"/model.weights")
                    saver2.save(session, model_path+"model.weights") #save to current working directory as well just to be safe!

            if FLAGS.log_losses:
                losses.append(loss)
                losses_validation.append(loss_validation)

        #TODO plot or return losses

        logging.info("Best F1 Score = %f" % (best_f1*100))
        saver.save(session, self.train_dir+"/model.weights")

        best_f1_path = results_path[:-1]+"_"+str(round(best_f1,2))+"_f1"
        os.rename(model_path[:-1],  best_f1_path)
        os.rename("log/log.txt",    best_f1_path+"/log.txt")
        os.rename("log/flags.json", best_f1_path+"/flags.json")

    #-------------------------------------------------------------
    # training
    #-------------------------------------------------------------

    def train(self, session, data):
        """ training process starts here """

        print ("==> Training a gated word & char language model")
        tic = time.time()
        params = tf.trainable_variables()
        num_params = sum(map(lambda t: np.prod(tf.shape(t.value()).eval()), params))
        toc = time.time()

        results_path = "results/{:%Y%m%d_%H%M%S}/".format(datetime.now())
        model_path = results_path + "model.weights/"
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        saver2 = tf.train.Saver()
        saver = tf.train.Saver()
        logging.info("Number of params: %d (retreival took %f secs)" % (num_params, toc - tic))

        #---------------------------------------------------------
        # prepare ingredients
        #---------------------------------------------------------

        print("==> Loading dictionaries: ")
        bos = '|'

        # load word dictionary
        print("     word dict")
        if FLAGS.word_vocab_path:
            with open(FLAGS.word_vocab_path, 'rb') as f:
                word_dict = pkl.load(f) # word -> index
            word_idict = dict()
            for kk, vv in word_dict.iteritems():
                word_idict[vv] = kk     # index -> word

        # load character dictionary
        print("     char dict")
        if FLAGS.char_vocab_path:
            with open(FLAGS.char_vocab_path, 'rb') as f:
                char_dict = pkl.load(f) # word -> index
                char_dict[bos] = len(char_dict) # add the BOS symbol
            char_idict = dict()
            for kk, vv in char_dict.iteritems():
                char_idict[vv] = kk     # index -> char
        print("Done")

        """setup options:
        format: opts[0] = '|', [1] = maxlen, [2] = n_char, [3] = n_word
        ------------------------------------------------------------"""
        opts = []
        opts.append(bos)
        maxlen = FLAGS.maxlen
        opts.append(maxlen)
        n_char = len(char_dict) + 1
        opts.append(n_char)
        n_words = len(word_dict) + 1
        opts.append(n_words)

        # reload options?


        # load training data
        train = load_file(path=FLAGS.train_data)
        #print("training data:\n{}".format(train))

        # initialize params
        #print("==> Building model:")
        #params = init_params(opts)

        # reload parameters
        if FLAGS.restart and os.path.exists(FLAGS.restore_path):
            params = load_params(FLAGS.restore_path, params)

        # convert params to Theano shared variabel
        #tparams = init_tparams(params)

        # build computational graph
        #trng, is_train, pretrain_mode, x_f, x_r, x_spaces, x_last_chars, x_word_input, label_words, cost \
        #                                                                          = build_model(tparams, opts)
        #inps = [x_f, x_r, x_spaces, x_last_chars, x_word_input, label_words]

        print("==> Building f_cost...")
        #f_cost = theano.function(inps, cost)
        f_cost = self.setup_loss() #might wanna move this to the build() function!!!
        #print("Done")

        # get gradients
        #print("==> Computing gradient...")
        #grads = tensor.grad(cost, wrt=itemlist(tparams))

        # gradient clipping
        #print("gradient clipping...")
        #grad_norm = tensor.sqrt(tensor.sum([tensor.sum(g**2.) for g in grads]))
        #tau = opts['gradclip']
        #grad_clipped = []
        #for g in grads:
        #    grad_clipped.append(tensor.switch(tensor.ge(grad_norm, tau), g * tau / grad_norm, g))
        #print("Done")

        # build optimizer
        #lr = tensor.scalar(name='lr')
        #print("==> Building optimizers...")
        #f_grad_shared, f_update = eval(opts['optimizer'])(lr, tparams, grad_clipped, inps, cost)
        #print("Done")

        #---------------------------------------------------------
        # start optimization
        #---------------------------------------------------------

        print("==> Optimization:")

        # reload history
        history_errs = []
        #if opts['reload_'] and os.path.exists(opts['saveto']):
        #    history_errs = list(numpy.load(opts['saveto'])['history_errs'])
        best_p = None
        bad_counter = 0

        # load validation and test data
        if FLAGS.dev_data:
            valid_lines = []
            with open(FLAGS.dev_data, 'r') as f:
                for l in f:
                    valid_lines.append(l)
            n_valid_lines = len(valid_lines)
        if FLAGS.test_data:
            test_lines = []
            with open(FLAGS.test_data, 'r') as f:
                for l in f:
                    test_lines.append(l)
            n_test_lines = len(test_lines)

        # initialize some values
        uidx = 0                 # update counter
        estop = False            # early stopping flag
        m = FLAGS.pretrain     # pretrain for m epochs using word/char only
        lrate = FLAGS.learning_rate
        lr_decayed = FLAGS.learning_rate
        batch_size = FLAGS.batch_size
        pretrain_mode = FLAGS.pretrain
        is_train = FLAGS.is_train

        # outer loop: epochs
        for eidx in xrange(FLAGS.epochs):

            n_samples = 0  # sample counter

            # shuffle training data every epoch
            #print("==> Shuffling sentences...")
            #shuffle(train)
            #print("Done")

            # learning rate decay
            if eidx >= FLAGS.lr_decay_start:
                lr_decayed /= FLAGS.learning_rate_decay

            # set pretraining mode
            if eidx in [e for e in range(m)]:
                #pretrain_mode.set_value(0.)
                pretrain_mode = 0
            elif eidx in [e + m for e in range(m)]:
                lrate = .1
                #pretrain_mode.set_value(1.)
                pretrain_mode = 1
            else:
                lrate = lr_decayed
                #pretrain_mode.set_value(2.)
                pretrain_mode = 2
            print("pretrain_mode = {}, epoch = {}, lr = {}".format(pretrain_mode, eidx, lrate))

            # training iterator
            kf_train = KFold(len(train), n_folds=int(len(train)/(batch_size-1)), shuffle=False)
            print("kf_train:\n{}".format(kf_train))


            # inner loop: batches
            for stuff, index in kf_train:
                #print("index: \n{}\nstuff: \n{}".format(index, stuff))
            #for _, index in enumerate(kf_train):
            #for index, minibatch in self.get_minibatches(train, batch_size):
                n_samples += len(index)
                uidx += 1

                # is_train=1 at training time
                FLAGS.is_train = 1



                # get a batch
                x = [(train[i]) for i in index]
                #x = [self.minibatch(train, i) for i in index]


                # get vectors
                x_f_, x_r_, x_spaces_, x_last_chars_, x_word_input_, label_words_ \
                                                          = txt_to_inps(x, char_dict, word_dict, opts)
                #print ("x_f_:\n{}\n\n, x_r_:\n{}\n\n, x_spaces_:\n{}\n\n, x_last_chars_:\n{}\n\n, x_word_input_:\n{}\n\n, label_words_:\n{}\n\n".format(x_f_, x_r_, x_spaces_, x_last_chars_, x_word_input_, label_words_))

                #reshape from time-major to batch-major
                print ("x_f.shape[0]: {}, x_f.shape[1]: {}".format(x_f_.shape[0], x_f_.shape[1]))
                print ("x_word_input_.shape[0]: {}, x_word_input_.shape[1]: {}".format(x_word_input_.shape[0], x_word_input_.shape[1]))
                x_f_ = numpy.reshape(x_f_, (x_f_.shape[1], x_f_.shape[0]))
                x_r_ = numpy.reshape(x_r_, (x_r_.shape[1], x_r_.shape[0]))
                x_spaces_ = numpy.reshape(x_spaces_, (x_spaces_.shape[1], x_spaces_.shape[0]))
                x_last_chars_ = numpy.reshape(x_last_chars_, (x_last_chars_.shape[1], x_last_chars_.shape[0]))
                #x_word_ = numpy.reshape(x_word_, (x_word_.shape[1], x_word_.shape[0]))
                x_word_input_ = numpy.reshape(x_word_input_, (x_word_input_.shape[1], x_word_input_.shape[0]))
                label_words_ = numpy.reshape(label_words_, (label_words_.shape[1], label_words_.shape[0]))
                print ("x_f.shape[0]: {}, x_f.shape[1]: {}".format(x_f_.shape[0], x_f_.shape[1]))
                print ("x_word_input_.shape[0]: {}, x_word_input_.shape[1]: {}".format(x_word_input_.shape[0], x_word_input_.shape[1]))
                #print ("x_f_:\n{}\n\n, x_r_:\n{}\n\n, x_spaces_:\n{}\n\n, x_last_chars_:\n{}\n\n, x_word_input_:\n{}\n\n, label_words_:\n{}\n\n".format(x_f_, x_r_, x_spaces_, x_last_chars_, x_word_input_, label_words_))

                # compute cost
                cost = self.optimize(session, x_f_, x_r_, x_spaces_, x_last_chars_, x_word_input_, label_words_)

                # update parameters
                #f_update(lrate)

                # check cost
                if numpy.isnan(cost) or numpy.isinf(cost):
                    print("NaN detected")
                    return 1., 1., 1.

                # display cost
                if numpy.mod(uidx, opts['dispFreq']) == 0:
                    print("epoch = {}, update = {}, cost = {}".format(eidx, uidx, cost))

                # save params
                if numpy.mod(uidx, opts['saveFreq']) == 0:
                    print("Saving...")
                    if best_p is not None:
                        params = best_p
                    else:
                        params = unzip(tparams)
                    numpy.savez(FLAGS.saveto, history_errs=history_errs, **params)
                    pkl.dump(opts, open('%s.pkl' % FLAGS.saveto, 'wb'))
                    print("Done")

                # compute validation/test perplexity
                if numpy.mod(uidx, FLAGS.valid_freq) == 0:
                    print ("Computing Dev/Test Perplexity")

                    # is_train=0 at valid/test time
                    FLAGS.is_train = 0
                    valid_err = perplexity(f_cost, valid_lines, char_dict, word_dict, opts)
                    test_err = perplexity(f_cost, test_lines, char_dict, word_dict, opts)
                    history_errs.append([valid_err, test_err])

                    # save the best params
                    if len(history_errs) > 1:
                        if uidx == 0 or valid_err <= numpy.array(
                                history_errs)[:, 0].min():
                            best_p = unzip(tparams)
                            print("Saving best params...")
                            numpy.savez(FLAGS.savebestto, history_errs=history_errs, **params)
                            pkl.dump(opts, open('%s.pkl' % FLAGS.savebestto, 'wb'))
                            print("Done")
                            bad_counter = 0
                        if len(history_errs) > FLAGS.patience and valid_err >= numpy.array(
                                    history_errs)[:-FLAGS.patience, 0].min():
                            bad_counter += 1
                            if bad_counter > FLAGS.patience:
                                print("Early Stop!")
                                estop = True
                                break

                    print("valid = {}, test = {}, ".format(valid_err, test_err))


            # inner loop: end

            print("Seen %d samples" % n_samples)

            # early stopping
            if estop:
                break

        # outer loop: end

        if best_p is not None:
            zipp(best_p, tparams)

        # compute validation/test perplexity at the end of training
        is_train.set_value(0.)
        valid_err = perplexity(f_cost, valid_lines, char_dict, word_dict, opts)
        test_err = perplexity(f_cost, test_lines, char_dict, word_dict, opts)
        print("valid = {}, test = {}, ".format(valid_err, test_err))

        # save everithing
        params = copy.copy(best_p)
        numpy.savez(opts['saveto'], zipped_params=best_p, valid_err=valid_err,
                    test_err=test_err, history_errs=history_errs, **params)

        return valid_err, test_err
