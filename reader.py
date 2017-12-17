from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os
import sys
import emoji
import tensorflow as tf
import numpy as np

Py3 = sys.version_info[0] == 3

def _read_words(filename):
  with tf.gfile.GFile(filename, "r") as f:
    if Py3:
      return f.read().split()
      #return f.read().replace("\n", "<eos>").split()
    else:
      return f.read().decode("utf-8").split()
      #return f.read().decode("utf-8").replace("\n", "<eos>").split()

def _build_vocab(data_path, filename):
  data = _read_words(filename)

  counter = collections.Counter(data)
  count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))

  words, _ = list(zip(*count_pairs))
  word_to_id = dict(zip(words, range(len(words))))
  #for k,v in word_to_id.items():
  #  print(k, v)

  emoji_list = []
  if "input" in filename:
      filepath = os.path.join(data_path, "input_vocab")
      print(filepath)
      a = open(filepath, "w")
      for k,v in word_to_id.items():
          #print(k.encode('utf-8'), v)
          a.write(k + " " + str(v) + "\n")
          if k in emoji.UNICODE_EMOJI:
              emoji_list.append(v)
      a.close()
  if "output" in filename:
      filepath = os.path.join(data_path, "output_vocab")
      print(filepath)
      a = open(filepath, "w")
      for k,v in word_to_id.items():
          #print(k.encode('utf-8'), v)
          a.write(k + " " + str(v) + "\n")
          if k in emoji.UNICODE_EMOJI:
              print(k, v)
              emoji_list.append(v)
      a.close()

  return word_to_id, emoji_list


def _file_to_word_ids(filename, word_to_id):
  data = _read_words(filename)
  return [word_to_id[word] for word in data if word in word_to_id]

def raw_data(data_path=None):

  train_path = os.path.join(data_path, "train.txt")
  valid_path = os.path.join(data_path, "valid.txt")
  test_path = os.path.join(data_path, "test.txt")

  word_to_id, emoji_list = _build_vocab(data_path, train_path)
  train_data = _file_to_word_ids(train_path, word_to_id)
  valid_data = _file_to_word_ids(valid_path, word_to_id)
  test_data = _file_to_word_ids(test_path, word_to_id)
  vocabulary = len(word_to_id)
  print("vocab_len: ", vocabulary)
  return train_data, valid_data, test_data, vocabulary, emoji_list

def producer(input_raw_data, output_raw_data, batch_size, num_steps, name=None):
  with tf.name_scope(name, "PTBProducer", [input_raw_data, output_raw_data, batch_size, num_steps]):
    input_raw_data = tf.convert_to_tensor(input_raw_data, name="input_raw_data", dtype=tf.int32)
    output_raw_data = tf.convert_to_tensor(output_raw_data, name="output_raw_data", dtype=tf.int32)

    data_len = tf.size(input_raw_data)
    batch_len = data_len // batch_size
    input_data = tf.reshape(input_raw_data[0 : batch_size * batch_len],
                      [batch_size, batch_len])
    output_data = tf.reshape(output_raw_data[0 : batch_size * batch_len],
                      [batch_size, batch_len])

    epoch_size = (batch_len - 1) // num_steps
    assertion = tf.assert_positive(
        epoch_size,
        message="epoch_size == 0, decrease batch_size or num_steps")
    with tf.control_dependencies([assertion]):
      epoch_size = tf.identity(epoch_size, name="epoch_size")

    i = tf.train.range_input_producer(epoch_size, shuffle=False).dequeue()
    x = tf.strided_slice(input_data, [0, i * num_steps],
                         [batch_size, (i + 1) * num_steps])
    x.set_shape([batch_size, num_steps])
    y = tf.strided_slice(output_data, [0, i * num_steps + 1],
                         [batch_size, (i + 1) * num_steps + 1])
    y.set_shape([batch_size, num_steps])
    return x, y

def gen_w2v_matrix(vocab_len, input_data):
  word2id = {}
  with open(input_data+"/input_vocab", "r") as text:
    for line in text:
      words = line.split()
      word2id[words[0]] = words[1]

  w2v_matrix = np.zeros((vocab_len, 512), dtype=float)
  with open(input_data+"/word2feat_512", "r") as text:
    for line in text:
      words = line.split()
      _id = word2id[words[0]]
      for i in range(1,513):
        w2v_matrix[int(_id)][i-1] = float(words[i])

  return w2v_matrix
