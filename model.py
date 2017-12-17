#!/usr/bin/python3 -u

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

import reader
from tensorflow.python.client import device_lib

class Config(object):
  init_scale = 0.04
  learning_rate = 0.1
  max_grad_norm = 10
  num_layers = 2
  num_steps = 35
  hidden_size = 1024
  max_epoch = 15
  keep_prob = 0.75
  lr_decay = 0.5
  batch_size = 64
  input_vocab_size = 28773
  output_vocab_size = 27648

class Input(object):
  def __init__(self, config, input_data, output_data, name=None):
    self.batch_size = batch_size = config.batch_size
    self.num_steps = num_steps = config.num_steps
    self.epoch_size = ((len(input_data) // batch_size) - 1) // num_steps
    self.input_data, self.targets = reader.producer(
        input_data, output_data, batch_size, num_steps, name=name)

class Model(object):
  def __init__(self, is_training, config, input_, l2=False, w2v=None):
    self._is_training = is_training
    self._input = input_
    self._rnn_params = None
    self._cell = None
    self.batch_size = input_.batch_size
    self.num_steps = input_.num_steps
    size = config.hidden_size
    input_vocab_size = config.input_vocab_size
    output_vocab_size = config.output_vocab_size

    #with tf.device("/cpu:0"):
    #  embedding = tf.get_variable(
    #      "embedding", [input_vocab_size, 256], dtype=tf.float32)
    #  inputs = tf.nn.embedding_lookup(embedding, input_.input_data)

    #if is_training and config.keep_prob < 1:
    #  inputs = tf.nn.dropout(inputs, config.keep_prob)
    
    _inputs = tf.one_hot(input_.input_data, input_vocab_size)
    self.word2vec = tf.convert_to_tensor(w2v, name="word2vec", dtype=tf.float32)
    inputs = tf.reshape(tf.matmul(tf.reshape(_inputs, [input_.batch_size* input_.num_steps, -1]), self.word2vec),
             [input_.batch_size, input_.num_steps, -1])

    output, state = self._build_rnn_graph(inputs, config, is_training)

    stdv = np.sqrt(1. / output_vocab_size)
    initializer = tf.random_uniform_initializer(-stdv * 0.8, stdv * 0.8)
    softmax_w = tf.get_variable(
        "softmax_w", [size, output_vocab_size], initializer=initializer, dtype=tf.float32)
    softmax_b = tf.get_variable(
        "softmax_b", [output_vocab_size], initializer=tf.constant_initializer(0.0, dtype=tf.float32),dtype=tf.float32)
    logits = tf.nn.xw_plus_b(output, softmax_w, softmax_b)
    logits = tf.reshape(logits, [self.batch_size, self.num_steps, output_vocab_size])

    self._output_probs = tf.nn.softmax(logits)
    self._prediction = tf.argmax(self._output_probs, axis=2, output_type=tf.int32)
    #self._accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self._output_probs, axis = 2, output_type=tf.int32), input_.targets), tf.float32))

    self._weights = tf.multiply(0.0, tf.cast(tf.not_equal(input_.targets, 0), tf.float32)) + 1.0
    print("self._weights: ", self._weights)
    loss = tf.contrib.seq2seq.sequence_loss(
        logits,
        input_.targets,
        self._weights,
        average_across_timesteps=False,
        average_across_batch=True)

    #L2正则
    train_vars = tf.trainable_variables()
    for v in train_vars:
        print("v.name: ", v.name, v.shape)
    lossL2 = tf.add_n([ tf.nn.l2_loss(v) for v in train_vars
                    if 'embedding' not in v.name and 'bias' not in v.name and 'softmax_b' not in v.name])
    print("l2: ", l2)
    if l2:
        self._cost = tf.reduce_sum(loss) + 0.001 * tf.reduce_mean(lossL2)   
    else:
        self._cost = tf.reduce_sum(loss)
    self._final_state = state

    if not is_training:
      return

    self._lr = tf.Variable(0.0, trainable=False)
    tvars = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(tf.gradients(self._cost, tvars),
                                      config.max_grad_norm)
    optimizer = tf.train.GradientDescentOptimizer(self._lr)
    self._train_op = optimizer.apply_gradients(
        zip(grads, tvars),
        global_step=tf.train.get_or_create_global_step())

    self._new_lr = tf.placeholder(
        tf.float32, shape=[], name="new_learning_rate")
    self._lr_update = tf.assign(self._lr, self._new_lr)

  def _build_rnn_graph(self, inputs, config, is_training):
    def make_cell():
      #cell = tf.contrib.rnn.LSTMBlockCell(config.hidden_size, forget_bias=1.0)
      cell = tf.contrib.rnn.LSTMCell(config.hidden_size, use_peepholes=False)
      if is_training and config.keep_prob < 1:
        cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=config.keep_prob)
      return cell

    cell = tf.contrib.rnn.MultiRNNCell(
        [make_cell() for _ in range(config.num_layers)], state_is_tuple=True)

    self._initial_state = cell.zero_state(config.batch_size, tf.float32)
    state = self._initial_state
    inputs = tf.unstack(inputs, num=self.num_steps, axis=1)
    outputs, state = tf.contrib.rnn.static_rnn(cell, inputs,
                                initial_state=self._initial_state)
    output = tf.reshape(tf.concat(outputs, 1), [-1, config.hidden_size])
    return output, state

  def assign_lr(self, session, lr_value):
    session.run(self._lr_update, feed_dict={self._new_lr: lr_value})

  @property
  def input(self):
    return self._input

  @property
  def targets(self):
    return self._input.targets
  
  @property
  def prediction(self):
    return self._prediction

  @property
  def outputs(self):
    return self._output_probs
  
  @property
  def initial_state(self):
    return self._initial_state

  @property
  def cost(self):
    return self._cost

  @property
  def final_state(self):
    return self._final_state

  @property
  def lr(self):
    return self._lr

  @property
  def train_op(self):
    return self._train_op

  @property
  def initial_state_name(self):
    return self._initial_state_name

  @property
  def final_state_name(self):
    return self._final_state_name

