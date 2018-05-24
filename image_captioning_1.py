# coding=utf-8

import numpy as np
import tensorflow as tf
import json
from copy import deepcopy
import os, time, sys
import cPickle as pickle
from json import encoder

encoder.FLOAT_REPR = lambda o: format(o, '.3f')
from utils import train_data_iterator, sample

with open('data_files/val_image_id2feature.pkl', 'r') as f:
    val_image_id2feature = pickle.load(f)


class Config(object):
    image_dim = 1024
    hidden_dim = embedding_dim = 512
    lr = 0.001
    max_epochs = 100
    batch_size = 256
    keep_prob = 0.8
    layers = 2


class Model(object):
    def __init__(self, config):
        self.config = config
        self.load_data()
        self.vocab_size = len(self.id2char)

        self._sentence_placeholder = tf.placeholder(tf.int32, shape=[self.config.batch_size, None], name='sentence')
        self._image_placeholder = tf.placeholder(tf.float32, shape=[self.config.batch_size, self.config.image_dim],
                                                 name='image')
        self._targets_placeholder = tf.placeholder(tf.int32, shape=[self.config.batch_size, None], name='targets')
        self._dropout_placeholder = tf.placeholder(tf.float32, name='dropout')

        with tf.variable_scope('CNN'):
            w1 = tf.get_variable('w1', shape=[self.config.image_dim, self.config.embedding_dim])
            b1 = tf.get_variable('b1', shape=[self.config.batch_size, self.config.embedding_dim])
            image_input = tf.expand_dims(tf.nn.sigmoid(tf.matmul(self._image_placeholder, w1) + b1), 1)
            print('image:{0}'.format(image_input.shape()))

        with tf.variable_scope('sentence_input'):
            word_embeddings = tf.get_variable('word_embeddings', shape=[self.vocab_size, self.config.embedding_dim])
            sentences_input = tf.nn.embedding_lookup(word_embeddings, self._sentence_placeholder)
            print('sentences_inputs:{0}'.format(sentences_input.shape()))

        with tf.variable_scope('all_inputs'):
            all_inputs = tf.concat(1, [image_input, sentences_input])
            print('combined:{0}'.format(all_inputs.shape()))

        cell_lstm = tf.nn.rnn_cell_BasicLSTMCell(self.config.hidden_dim, forget_bias=1,
                                                 input_size=self.config.embedding_dim)
        cell_lstm_dropout = tf.nn.rnn_cell.DropoutWrapper(cell_lstm, input_keep_prob=self._dropout_placeholder,
                                                          output_keep_prob=self._dropout_placeholder)
        layers_lstm = tf.nn.rnn_cell.MutilRNNCell([cell_lstm_dropout for _ in range(self.config.layers)])
        initial_state = layers_lstm.zero_state(self.config.batch_size, tf.float32)
        outputs, final_state = tf.nn.dynamic_rnn(layers_lstm, all_inputs, initial_state=initial_state, scope='LSTM')
        output = tf.reshape(outputs, [-1, self.config.hidden_dim])
        self._final_state = final_state
        print('Outputs (raw):{0}'.format(outputs.shape()))
        print('Final state:{0}'.format(final_state.shape()))
        print('Output (reshaped):{0}'.format(output.shape()))

        with tf.variable_scope('softmax'):
            w2 = tf.get_variable('w2', shape=[self.config.hidden_dim, self.vocab_size])
            b2 = tf.get_variable('b2', shape=[self.vocab_size])
            logits = tf.matmul(output, w2) + b2
        print('Logits:{0}'.format(logits.shape()))

        self.logits = logits
        self._prediction = prediction = tf.argmax(self.logits, 1)
        print('Prediction:{0}'.format(prediction.shape()))

        targets_reshaped = tf.reshape(self._targets_placeholder, [-1])
        print('Targets (raw):{0}'.format(self._targets_placeholder.shape()))
        print('Targets (reshaped):{0}'.format(targets_reshaped.shape()))

        with tf.variable_scope('loss'):
            # _targets is [-1, ..., -1] so that the first and last logits are not used
            # these correspond to the img step and the <eos> step
            self.loss = loss = tf.reduce_sum(
                tf.nn.sparse_softmax_cross_entropy_with_logits(logits, targets_reshaped, name='loss'))
            print('Loss:{0}'.format(loss.shape()))
        with tf.variable_scope('optimizer'):
            optimizer = tf.train.AdamOptimizer(self.config.lr)
            self.train_op = optimizer.minimize(loss)

    def load_data(self, type='train'):
        if type == 'train':
            with open('data_files/index2token.pkl', 'r') as f:
                self.id2char = pickle.load(f)
            with open('data_files/preprocessed_train_captions.pkl', 'r') as f:
                self.train_captions, self.train_caption_id2sentence, self.train_caption_id2image_id = pickle.load(f)
            with open('data_files/train_image_id2feature.pkl', 'r') as f:
                self.train_image_id2feature = pickle.load(f)

    def run_epoch(self, session, train_op):
        total_steps = sum(1 for x in train_data_iterator(self.train_captions, self.train_caption_id2sentence,
                                                         self.train_caption_id2image_id, self.train_image_id2feature,
                                                         self.config))
        total_loss = []
        if not train_op:
            train_op = tf.no_op()
        start = time.time()

        for step, (sentences, images, targets) in enumerate(
                train_data_iterator(self.train_captions, self.train_caption_id2sentence, self.train_caption_id2image_id,
                                    self.train_image_id2feature, self.config)):

            feed = {self._sentence_placeholder: sentences,
                    self._image_placeholder: images,
                    self._targets_placeholder: targets,
                    self._dropout_placeholder: self.config.keep_prob}
            loss, _ = session.run([self.loss, train_op], feed_dict=feed)
            total_loss.append(loss)

            if (step % 50) == 0:
                print('%d/%d: loss = %.2f time elapsed = %d' % (step, total_steps, np.mean(total_loss), time.time() - start))

        print('Total time: %ds' % (time.time() - start))
        return total_loss


