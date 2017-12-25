from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os
import re
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

def has_numbers(inputString):
    return bool(re.search(r'\d', inputString))

def clean_str(line):
    line = line.replace("`", "'").replace("‘", "'").replace("_", " ").replace("’","'").replace("“","\"").replace("”","\"").replace("^", "").replace("+", "").replace("-", "").replace("\*", "") 
 
    #line = re.sub(r"[^A-Za-z0-9,\']", " ", line) 
    line = re.sub(r"\.", " ", line)
    line = re.sub(r"!", " ", line)
    line = re.sub(r"\?", " ", line)
    line = re.sub(r"\(", " ", line)
    line = re.sub(r"\)", " ", line)
    line = re.sub(r"\"", " ", line)
    line = re.sub(r"\'s", " \'s", line)
    line = re.sub(r"\'S", " \'S", line)
    line = re.sub(r"\'m", " \'m", line)
    line = re.sub(r"\'M", " \'M", line)
    line = re.sub(r"\'ve", " \'ve", line)
    line = re.sub(r"\'VE", " \'VE", line)
    line = re.sub(r"n\'t", " n\'t", line)
    line = re.sub(r"N\'T", " N\'T", line)
    line = re.sub(r"\'re", " \'re", line)
    line = re.sub(r"\'RE", " \'RE", line)
    line = re.sub(r"\'d", " \'d", line)
    line = re.sub(r"\'D", " \'D", line)
    line = re.sub(r"\'ll", " \'ll", line)
    line = re.sub(r"\'LL", " \'LL", line) 

    return line

def test_data(data_path=None):

    test_path = os.path.join(data_path, "test.txt")
    vocab_path = os.path.join(data_path, "input_vocab")
    output_vocab_path = os.path.join(data_path, "output_vocab")
    print(test_path, vocab_path, output_vocab_path)

    word2id = {}
    vocab_len = 0
    with open(vocab_path, "r") as text:
        for line in text:
            line = line.strip()
            vocab_len += 1
            words = line.split()
            word2id[words[0]] = int(words[1])

    output_word2id = {}
    output_vocab_len = 0
    emoji_list = []
    with open(output_vocab_path, "r") as text:
        for line in text:
            line = line.strip()
            output_vocab_len += 1
            words = line.split()
            output_word2id[words[0]] = int(words[1])
            if words[0] in emoji.UNICODE_EMOJI:
                emoji_list.append(int(words[1]))
    
    test_list = []
    output_test_list = []
    with open(test_path, "r") as text:
        for line in text:
            line = line.strip()
            line = clean_str(line)
            line = ' '.join(line.split()) 
            if (len(line)) < 2:
                continue
            print(line)
            for sent in line.split():
                if has_numbers(sent):
                    test_list.append(word2id["<num>"])
                elif sent == ",":
                    test_list.append(word2id["<pun>"])
                else:
                    if word2id.get(sent) == None:
                        if sent in emoji.UNICODE_EMOJI:
                            test_list.append(word2id["<emoji>"])
                        else:
                            test_list.append(word2id["<unk>"])
                    else:
                        test_list.append(word2id[sent])
                if output_word2id.get(sent) == None:
                    output_test_list.append(output_word2id["<unk>"])
                else:
                    output_test_list.append(output_word2id[sent])

    #for word in test_list:
    #    print(word)

    print("output_vocab_len: ", output_vocab_len)
    return test_list, output_test_list, vocab_len, emoji_list

