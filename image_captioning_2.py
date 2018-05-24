# coding=utf-8

import numpy as np
import tensorflow as tf
import math
import os
import ipdb
import pandas as pd
import cPickle

from tensorflow.models.rnn import rnn_cell
import tensorflow.python.platform
from keras.preprocessing import sequence


class Config(object):
    n_epochs = 1000
    batch_size = 128
    hidden_dim = 256
    embedding_dim = 256
    context_dim = 512
    context_shape = [196, 512]
    pretrained_model_path = './model/model-8'
    annotation_path = './data/annotations.pickle'
    feat_path = './data/feats.npy'
    model_path = './model/'


class CaptionGenerator(object):
    def init_weights(self, dim_1, dim_2, name=None, stddev=1.0):
        return tf.Variable(tf.truncated_normal([dim_1, dim_2], stddev=stddev / math.sqrt(float(dim_1)), name=name))

    def init_bias(self, dim_2, name=None):
        return tf.Variable(tf.zeros([dim_2]), name=name)

    def __init__(self, config, n_words, n_steps, bias_init_vector=None):
        self._n_words = n_words
        self._n_steps = n_steps
        self._config = config

        with tf.device('/cpu:0'):
            self._Word_embedding = tf.Variable(
                tf.random_uniform([self._n_words, self._config.embedding_dim], -1.0, 1.0),
                name='Word_embedding')

        self._init_hidden_W = self.init_weights(self._config.context_dim, self._config.hidden_dim, name='init_hidden_W')
        self._init_hidden_b = self.init_bias(self._config.hidden_dim, name='inti_hidden_b')

        self._init_c_W = self.init_weights(self._config.context_dim, self._config.hidden_dim, name='init_c_W')
        self._init_c_b = self.init_bias(self._config.hidden_dim, name='init_c_b')

        self._lstm_W = self.init_weights(self._config.embedding_dim, self._config.hidden_dim * 4, name='lstm_W')
        self._lstm_U = self.init_weights(self._config.embedding_dim, self._config.hidden_dim * 4, name='lstm_U')
        self._lstm_b = self.init_bias(self._config.hidden_dim * 4, name='lstm_b')

        self._image_encode_W = self.init_weights(self._config.context_dim, self._config.hidden_dim * 4,
                                                 name='image_encode_W')
        self._image_attention_W = self.init_weights(self._config.context_dim, self._config.context_dim,
                                                    name='image_attention_W')
        self._hidden_attention_W = self.init_weights(self._config.hidden_dim, self._config.context_dim,
                                                     name='hidden_attention_W')
        self._pre_attention_b = self.init_bias(self._config.context_dim, name='pre_attention_b')

        self._attention_W = self.init_weights(self._config.context_dim, 1, name='attention_W')
        self._attention_b = self.init_bias(1, name='attention_b')

        self._decode_lstm_W = self.init_weights(self._config.hidden_dim, self._config.embedding_dim,
                                                name='decode_lstm_W')
        self._decode_lstm_b = self.init_bias(self._config.embedding_dim, name='decode_lstm_b')
        self._decode_word_W = self.init_weights(self._config.embedding_dim, self._n_words, name='decode_word_W')

        if bias_init_vector is not None:
            self._decode_word_b = tf.Variable(bias_init_vector.astype(np.float32), name='decode_word_b')
        else:
            self._decode_word_b = self.init_bias(self._n_words, name='decode_word_b')

    def get_initial_lstm_h_and_c(self, context):
        initial_h = tf.nn.tanh(tf.matmul(context, self._init_hidden_W) + self._init_hidden_b)
        initial_c = tf.nn.tanh(tf.matmul(context, self._init_c_W) + self._init_c_b)
        return initial_h, initial_c

    def create_model(self):
        context = tf.placeholder('float32', [self._config.batch_size, self._config.context_shape[0],
                                             self._config.context_shape[1]])
        sentence = tf.placeholder('int32', [self._config.batch_size, self._n_steps])
        mask = tf.placeholder('float32', [self._config.batch_size, self._n_steps])

        h, c = self.get_initial_lstm_h_and_c(tf.reduce_mean(context, 1))

        context_flat = tf.reshape(context, [-1, self._config.context_dim])
        context_encode = tf.matmul(context_flat, self._image_attention_W)
        context_encode = tf.reshape(context_encode, [-1, self._config.context_shape[0], self._config.context_shape[1]])

        loss = 0.0

        for i in range(self._n_steps):
            if i == 0:
                word_emb = tf.zeros([self._config.batch_size, self._config.embedding_dim])
            else:
                tf.get_variable_scope().reuse_variables()
                with tf.device("/cpu:0"):
                    word_emb = tf.nn.embedding_lookup(self._Word_embedding, sentence[:, i - 1])

            x_t = tf.matmul(word_emb, self._lstm_W) + self._lstm_b

            labels = tf.expand_dims(sentence[:, i], 1)
            indices = tf.expand_dims(tf.range(0, self._config.batch_size, 1), 1)
            concated = tf.concat(1, [indices, labels])
            onehot_labels = tf.sparse_to_dense(concated, tf.pack([self._config.batch_size, self._n_words]), 1.0, 0.0)

            context_encode = context_encode + tf.expand_dims(tf.matmul(h, self._hidden_attention_W),
                                                             1) + self._pre_attention_b
            context_encode = tf.nn.tanh(context_encode)

            context_encode_flat = tf.reshape(context_encode, [-1, self._config.context_dim])
            alpha = tf.matmul(context_encode_flat, self._attention_W) + self._attention_b
            alpha = tf.reshape(alpha, [-1, self._config.context_shape[0]])
            alpha = tf.nn.softmax(alpha)

            context_weighted = tf.reduce_sum(context * tf.expand_dims(alpha, 2), 1)
            pre_lstm = tf.matmul(h, self._lstm_U) + x_t + tf.matmul(context_weighted, self._image_encode_W)
            i, f, o, new_c = tf.split(1, 4, pre_lstm)

            i = tf.nn.sigmoid(i)
            f = tf.nn.sigmoid(f)
            o = tf.nn.sigmoid(o)
            new_c = tf.nn.tanh(new_c)

            c = f * c + i * new_c
            h = o * tf.nn.tanh(new_c)

            logits = tf.matmul(h, self._decode_lstm_W) + self._decode_lstm_b
            logits = tf.nn.relu(logits)
            logits = tf.nn.dropout(logits, keep_prob=0.5)

            logit_words = tf.matmul(logits, self._decode_word_W) + self._decode_word_b
            logit_cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logit_words, onehot_labels)
            logit_cross_entropy_mask = logit_cross_entropy * mask[:, i]

            current_loss = tf.reduce_sum(logit_cross_entropy_mask)
            loss += current_loss

        loss = loss / tf.reduce_sum(mask)
        return loss, context, sentence, mask

    def create_generator(self, max_len):
        context = tf.placeholder('float32', [-1, self._config.context_shape[0], self._config.context_shape[1]])
        h, c = self.get_initial_lstm_h_and_c(tf.reduce_mean(context, 1))

        context_encode = tf.matmul(tf.squeeze(context), self._image_attention_W)
        generator_words = []
        logit_list = []
        alpha_list = []
        word_embedding = tf.zeros([1, self._config.embedding_dim])

        for i in range(max_len):
            x_t = tf.matmul(word_embedding, self._lstm_W) + self._lstm_b
            context_encode = context_encode + tf.matmul(h, self._hidden_attention_W) + self._pre_attention_b
            context_encode = tf.nn.tanh(context_encode)

            alpha = tf.matmul(context_encode, self._attention_W) + self._attention_b
            alpha = tf.reshape(alpha, [-1, self._config.context_shape[0]])
            alpha = tf.nn.softmax(alpha)

            context_weighted = tf.reduce_sum(tf.squeeze(context) * alpha, 0)
            context_weighted = tf.expand_dims(context_weighted, 0)
            pre_lstm = tf.matmul(h, self._lstm_U) + x_t + tf.matmul(context_weighted, self._image_encode_W)
            i, f, o, new_c = tf.split(1, 4, pre_lstm)

            i = tf.nn.sigmoid(i)
            f = tf.nn.sigmoid(f)
            o = tf.nn.sigmoid(o)
            new_c = tf.nn.tanh(new_c)

            c = i * c + f * new_c
            h = o * tf.nn.tanh(new_c)

            logits = tf.matmul(h, self._decode_lstm_W) + self._decode_lstm_b
            logits = tf.nn.relu(logits)

            logit_words = tf.matmul(logits, self._decode_word_W) + self._decode_word_b
            max_prob_word = tf.argmax(logit_words, 1)
            print(max_prob_word.shape())

            with tf.device('/cpu:0'):
                word_embedding = tf.nn.embedding_lookup(self._Word_embedding, max_prob_word)

            generator_words.append(max_prob_word)
            logit_list.append(logit_words)

        return context, generator_words, logit_list, alpha_list


