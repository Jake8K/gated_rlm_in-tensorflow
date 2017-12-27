
#  layers.py
#   layers for gated rlm model
#
#  Created by Jake K on 11/1/17.
#   Based on Gated RLM by Miyamoto and Cho
#       ==> The model description is here: https://arxiv.org/abs/1606.01700
#       ==> The base code is here: https://github.com/nyu-dl/dl4mt-tutorial


import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.contrib import legacy_seq2seq

import numpy as np

#class bidirectional_lstm()
""" use new o'reily book p 422 for reference on creating a layer!"""
def bidirectional_lstm(self, char_dict, batch_tensor, n_hidden, X_char, train=False): #self, input placeholder, cell size, embeddings, training True to train embs
    """ instantiates a bidirectional LSTM for char-level word embeddings
    Parameters
        self
        char_dict    : a dict of chars, dtype={}, dim={[ ]}
        batch_tensor : 1-D tensor
        n_hidden     : size of LSTM cells, number of hidden units
        X_char     : a tensor with untrained or pretrained char-level embeddings, dim={[ ]}, dtype=
                     #maybe also have embed_size???
        train        : boolean of whether to train the char embeddings or use pre-trained char embeddings
    Returns
        Cemb: character-level word embedding (LSTM output)
        states: the final state of the bidirectional LSTM
    """
    """input tensor dims are reshaped to [n_timesteps, n_samples, dim_char]"""
    print("  ==> building bidirectional char LSTM")
    # from M & C: char-based embeddings / bidirectional LSTM:
    #emb_c = ['char_lookup_f'][x_f.flatten()].reshape([n_timesteps, n_samples, options['dim_char']])
        #proj_f = get_layer('char_lstm_x')[1](tparams, emb_c, options, 'bi_lstm_f', x_spaces)
    #emb_c = tparams['char_lookup_r'][x_r.flatten()].reshape([n_timesteps, n_samples, options['dim_char']])
        #proj_r = get_layer('char_lstm_x')[1](tparams, emb_c, options, 'bi_lstm_r', x_spaces)
    #proj = tensor.concatenate([proj_f[0], proj_r[0]], axis=2)
    #Cemb = get_layer('fc')[1](tparams, proj, options, 'combine_hs', activ='linear')

    #emb_c = ['char_lookup_f'][x_f.flatten()].reshape([n_timesteps, n_samples, options['dim_char']])
    print ("\tchar_dict length: {}".format(len(char_dict)))
    print ("\tbatch_tensor dims for bi-LSTM: {}".format(batch_tensor.get_shape().as_list()))
    print ("\tX_char dims for bi-LSTM: {}".format(X_char.get_shape().as_list()))


    #xavier weights initializer
    xavier = tf.contrib.layers.xavier_initializer()

    # WHY is it currently size [32, ?, 200]
    """
    #cher embeddings (this might just be for words...)
    if train:
        embeddings = tf.Variable(char_dict, name='trainable_char_embs', dtype=tf.float32)
    else:
        embeddings = tf.constant(char_dict, name='pretrained_char_embs', dtype=tf.float32)
        #embeddings = tf.Variable(char_dict, name='trainable_char_embs', dtype=tf.float32)
        print ("char embeddings dims for bi-LSTM: ", embeddings.get_shape().as_list())


    #https://www.tensorflow.org/api_docs/python/tf/nn/embedding_lookup
    temp_bs = tf.placeholder(dtype=tf.int64, shape=embeddings.get_shape().as_list(), name='q_ids')
    print("bs_tensor dims are: ", temp_bs.get_shape().as_list())
    temp_vocab_idx = tf.Variable(temp_bs, name="temporary_char_vocab_index_tensor_for_lookup", dtype=tf.int64)
    #temp_vocab_idx = tf.reshape(temp_vocab_idx, [batch_tensor.get_shape().as_list()[0], 200])
    char_vectors = tf.nn.embedding_lookup(params=embeddings, ids=temp_vocab_idx) #Value (dict) passed to parameter 'indices' has DataType float32 not in list of allowed values: int32, int64
    #char_vectors = tf.nn.embedding_lookup(params=embeddings, ids=char_dict) #Value (dict) passed to parameter 'indices' has DataType float32 not in list of allowed values: int32, int64
"""

    #lstm
    with tf.variable_scope("char_LSTM") as scope:
        fw_cell = tf.nn.rnn_cell.LSTMCell(num_units=n_hidden, initializer=xavier)
        bw_cell = tf.nn.rnn_cell.LSTMCell(num_units=n_hidden, initializer=xavier)
        outputs, states = tf.nn.bidirectional_dynamic_rnn( #https://www.tensorflow.org/api_docs/python/tf/nn/bidirectional_dynamic_rnn
                cell_fw=fw_cell, cell_bw=bw_cell, inputs=X_char, time_major=True,
                dtype=tf.float32, sequence_length=batch_tensor, scope=scope)
    """fc = fully_connected( add fully connected layer like original model)""" # returns: The tensor variable representing the result of the series of operations.
    out_fw, out_bw = outputs
    states_fw, states_bw = states
    char_LSTM_output = tf.concat([out_fw, out_bw], axis=-1) #axis 0-2= dim 0-2, -1 to 2 concats all dimensins starting @ last
    Cemb = char_LSTM_output
    Cemb_size = n_hidden * 2

    return Cemb


#class GloVe:
def GloVe(self, X_word, pretrained_embeds ):
    #Wemb = ['word_lookup'][x_word_input.flatten()].reshape([n_timesteps, n_samples, dim-word])
    print("  ==> building GloVE embeddings word lookup table")
    print ("\tglove embedding dims: [{}, {}]".format(pretrained_embeds.shape[0], pretrained_embeds.shape[1]))
    print ("\tX_word dims for GloVE: {}".format(X_word.get_shape().as_list()))
    with tf.variable_scope("word_lookup") as scope:
        """
        # init embed Lookup (do this once at the start instead of at this layer!)
        w_init = tf.random_uniform([self.n_words, self.dim_word], -1.0, 1.0) # double chec their INIT! (also prob dtype=tf.int64)
        self.word_embeddings = tf.Variable(w_init) #could also use tf.get_variable(name, [vocab_size, size],  dtype=tf.int64)

        # get embeddings in/out
        self.x_word_input = tf.placeholder(tf.int64, [self.batch_size, self.seq_length], "x_word_input") #dim [batch size, time_step, features]
        self.word_outputs = Wemb = tf.nn.embedding_lookup(self.word_embeddings, self.x_word_input)
        # NOTE: emb placeholders! https://stackoverflow.com/questions/35687678/using-a-pre-trained-word-embedding-word2vec-or-glove-in-tensorflow
        """
        self.embed_size = 200
        assert self.embed_size == self.pretrained_embeddings.shape[1]
        #self.word_ids_placeholder = tf.placeholder(dtype=tf.int32, shape=[None, self.maxlen], name='word_ids')
        self.embeddings = tf.constant(self.pretrained_embeddings, name='glove_embeddings', dtype=tf.float32)

        self.word_vectors = Wemb = tf.nn.embedding_lookup(params=self.embeddings, ids=X_word)
        assert self.word_vectors.get_shape().as_list() == [self.batch_size, self.maxlen, self.embed_size]


    return Wemb


#class Gate:
def gate(self, char_emb, word_emb, pretrain_mode): #dim = dim_word
    """ instantiates the gating function described my Myiamoto and Cho
    Parameters
        self?
        char_emb    : the output of the char-level bidirectional LSTM (Cemb)
        word_emb    : the output of the word-embedding lookup (Wemb)
        pretrain_mode : M & C used a tensor holding vals 0, 1, 2... hoping to just use an int
                        0 ( ), 1 ( ), 2( )
    Returns
        X: The gated value of the char and word level embeddings (a special concatenation)
    """
    print("  ==> building the gate layer")
    word_emb = tf.to_float(word_emb, 'Wemb_foat')
    print ("\tC_emb dims for gate: {}".format(char_emb.get_shape().as_list()))
    print ("\tW_emb dims for gate: {}".format(word_emb.get_shape().as_list()))
    with tf.variable_scope('Gate'): #Oreily book uses name scope (name_scope(name), name is an arg)
        #seed_int = random.randint()
        #init = tf.random_uniform(shape=dim, maxval=0.2, minval=0.1, dtype=float32, seed=seed_int)
        V_shape = word_emb.get_shape()#  #should be a 3D tesnor
        V_shape = [self.batch_size, self.maxlen, self.dim_word]
        init = tf.random_uniform(shape=V_shape, maxval=0.2, minval=0.1, dtype=tf.float32) #shape is same as Wemb
        V_g = tf.Variable(init, name='V_g')
        b_g = tf.Variable(tf.zeros([1, ]), dtype='float32', name='b_g')
        #G_g = tf.nn.sigmoid(tf.matmul(tf.constant(word_emb, shape=[200, 200]),
        #                                tf.constant(V_g, shape=[200, 200]),
        #                                a_is_sparse=True, b_is_sparse=True)
        #                            + b_g, 'G_wt')
        #G_g = tf.nn.sigmoid(tf.matmul(word_emb, V_g) + b_g, 'G_wt')
        G_g = tf.nn.sigmoid(tf.tensordot(word_emb, V_g, 3) + b_g, 'G_wt')


        # https://www.tensorflow.org/api_guides/python/control_flow_ops#Control_Flow_Operations
        def X_val(x):
            return x
        nested_ifelse = tf.cond( #https://www.tensorflow.org/api_docs/python/tf/cond
            #tf.equal(pretrain_mode, np.float32(0.)), if using tensor pretrain_mode
            tf.equal(pretrain_mode, 0.), #or just pretrain_mode == 0,
            name=None,
            true_fn=lambda: X_val(word_emb),
            false_fn=lambda: X_val(char_emb))
        X_wt = tf.cond(
                        #tf.less_equal(pretrain_mode, np.float32(1.)),
                        tf.less_equal(pretrain_mode, 1.), # or just pretrain_mode <= 1,
                        strict=False,
                        name=None,
                        true_fn=lambda: nested_ifelse,
                        false_fn=lambda: G_g[:, :, None] * char_emb + (1. - G_g)[:, :, None] * word_emb)
        """ send in pretrain _mode as int and implement the behavior w regular ifelse except for the last one?"""
        return X_wt

#class LSTM_LM:
def lstm_lm(self, input_X, n_hidden, num_layers):
    print("  ==> building the language modeling LSTM layers")
    print ("\tX_wt dims for LSTM-LM: {}".format(input_X.get_shape()))
    with tf.variable_scope('LSTM') as scope:
        """{TO DO} [add initial state] [add gradient clipping] [give it the shape of input_X] """
        # add a fully connected layer here? how to get gate output to be lstm input?
        input_X = tf.reshape(input_X, [self.batch_size, self.maxlen, self.dim_char])
        lstm_cell = rnn.LSTMCell(num_units=n_hidden) # NOTE: add clipping from paper!
        lstm_lm = rnn.MultiRNNCell([lstm_cell] * 2, state_is_tuple=True) #time_major=True)#, inputs=lm_input) #stacks it so it's 2 layers *might wanna make it dynamic *
        self.LSTM_LM_output, self.LSTM_LM_state = tf.nn.dynamic_rnn(lstm_lm, input_X, dtype=tf.float32)
        # not sure this actually goes here
        #outputs, _ = rnn(self.LSTM_LM, self.gate_ouput, dtype=tf.float32) <--- the logits
        #self.true_outputs = tf.placeholder(tf.int64, [self.batch_size, self.seq_length]) #aybe float 32? also check dimensionaliy

        """
        # [pre_softmax] NOT SURE AT ALL ABOUT THIS, but since they have dropout set to false... maybe don't even include it?
        # add a fully connected layer here? https://www.tensorflow.org/api_docs/python/tf/contrib/layers/fully_connected
        if training and (args.output_keep_prob < 1.0 or args.input_keep_prob < 1.0):
            # wrong->self.LSTM_LM  = rnn.DropoutWrapper(cell, input_keep_prob=args.input_keep_prob, output_keep_prob=args.output_keep_prob) # seed=None)?
            tf.contrib.layers.dropout(inputs, # https://www.tensorflow.org/api_docs/python/tf/contrib/layers/dropout
                                    keep_prob=0.5,
                                        noise_shape=None,
                                        is_training=True,
                                        outputs_collections=None,
                                        scope=None)

        # [SoftMax]
        # check https://github.com/spiglerg/RNN_Text_Generation_Tensorflow/blob/master/rnn_tf.py
        with tf.variable_scope('softmax'):
            W = tf.get_variable('W', [lstm_lm_size, n_words])
            b = tf.get_variable('b', [n_words], initializer=tf.constant_initializer(0.0))

        # https://www.tensorflow.org/api_docs/python/tf/contrib/layers/softmax
        self.preds = tf.contrib.layers.softmax(logits, scope=None)

        logit = get_layer('fc')[1](tparams, proj_h, options, 'pre_softmax', activ='linear')
        logit_shp = logit.shape
        probs = tensor.nnet.softmax(logit.reshape([logit_shp[0] * logit_shp[1], logit_shp[2]]))"""
    return self.LSTM_LM_output, self.LSTM_LM_state
