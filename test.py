#!/usr/bin/python3 -u

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import os
import numpy as np
import tensorflow as tf

import model
import reader

flags = tf.flags
logging = tf.logging
flags.DEFINE_string("input_data", None,
                    "Where the input training/test data is stored.")
flags.DEFINE_string("save_path", None,
                    "Model output directory.")
FLAGS = flags.FLAGS

def test_epoch(session, model, emoji_list):
  start_time = time.time()
  costs = 0.0
  iters = 0
  Total_num1 = 0.0
  Emoji_num1 = 0.0
  Emoji_right_predict1 = 0.0
  Word_right_predict1 = 0.0
  Wrong_predict1 = 0.0 
  Total_num3 = 0.0
  Emoji_num3 = 0.0
  Emoji_right_predict3 = 0.0 
  Word_right_predict3 = 0.0 
  Wrong_predict3 = 0.0 
  state = session.run(model.initial_state)

  fetches = {
      "cost": model.cost,
      "final_state": model.final_state,
      "target": model.targets,
      "output": model.outputs,
      "predict": model.prediction,
  }

  for step in range(model.input.epoch_size):
    feed_dict = {}
    for i, (c, h) in enumerate(model.initial_state):
      feed_dict[c] = state[i].c
      feed_dict[h] = state[i].h

    vals = session.run(fetches, feed_dict)
    cost = vals["cost"]
    state = vals["final_state"]
    predicts = vals["predict"]
    outputs = vals["output"]
    targets = vals["target"]

    total_num1 = 0.0001
    emoji_num1 = 0.0001
    emoji_right_predict1 = 0.0001 
    word_right_predict1 = 0.0001 
    wrong_predict1 = 0.0001 
    
    total_num3 = 0.000
    emoji_num3 = 0.000
    emoji_right_predict3 = 0.000
    word_right_predict3 = 0.000 
    wrong_predict3 = 0.000
    for i in range(model.batch_size):
      for j in range(model.num_steps):
        total_num3 += 1
        total_num1 += 1
        top3 = np.argsort(outputs[i][j])[-3:]
        if 1 in top3:
          top3 = np.argsort(outputs[i][j])[-4:]   #predict not contain <unk>
        if targets[i][j] in emoji_list:     #<unk>
          emoji_num1 += 1 
          emoji_num3 += 1 
        if targets[i][j] == 1:     #<unk>
          wrong_predict3 += 1 
        elif targets[i][j] in top3:
          if targets[i][j] in emoji_list:
            emoji_right_predict3 += 1
          word_right_predict3 += 1
        elif targets[i][j] not in top3:
          wrong_predict3 += 1 

        top1 = np.argsort(outputs[i][j])[-1:]
        if 1 in top1:
          top1 = np.argsort(outputs[i][j])[-2:]   #predict not contain <unk>
        if targets[i][j] == 1:     #<unk>
          wrong_predict1 += 1 
        elif targets[i][j] in top1:
          if targets[i][j] in emoji_list:
            emoji_right_predict1 += 1
          word_right_predict1 += 1
        elif targets[i][j] not in top1:
          wrong_predict1 += 1 

    Total_num1 = Total_num1 + total_num1 
    Emoji_num1 = Emoji_num1 + emoji_num1 
    Emoji_right_predict1 = Emoji_right_predict1 + emoji_right_predict1 
    Word_right_predict1 = Word_right_predict1 + word_right_predict1  
    Wrong_predict1 = Wrong_predict1 + wrong_predict1 
    Total_num3 = Total_num3 + total_num3 
    Emoji_num3 = Emoji_num3 + emoji_num3 
    Emoji_right_predict3 = Emoji_right_predict3 + emoji_right_predict3 
    Word_right_predict3 = Word_right_predict3 + word_right_predict3  
    Wrong_predict3 = Wrong_predict3 + wrong_predict3 
 
    #print(total_num1, emoji_num1, emoji_right_predict1, word_right_predict1, wrong_predict1, "emoji_recall_1: ", 1.0*emoji_right_predict1/emoji_num1, 
    #      "word_recall_1: ", 1.0*word_right_predict1/total_num1, "wrong_1: ", 1.0*wrong_predict1/total_num1) 
    #print(total_num3, emoji_num3, emoji_right_predict3, word_right_predict3, wrong_predict3, "emoji_recall_3: ", 1.0*emoji_right_predict3/emoji_num3, 
    #      "word_recall_3: ", 1.0*word_right_predict3/total_num3, "wrong_3: ", 1.0*wrong_predict3/total_num3) 

    costs += cost
    iters += model.input.num_steps

    #print("%.3f perplexity: %.5f cost: %f speed: %.0f wps" %
    #      (step * 1.0 / model.input.epoch_size, np.exp(costs / iters), costs/iters,
    #      iters * model.input.batch_size / (time.time() - start_time)))

  print(Total_num1, Emoji_num1, Emoji_right_predict1,  Word_right_predict1, Wrong_predict1, "Emoji_recall_1: ", 1.0*Emoji_right_predict1/Emoji_num1, 
        "Word_recall_1: ", 1.0*Word_right_predict1/Total_num1, "Wrong_1: ", 1.0*Wrong_predict1/Total_num1) 
  print(Total_num3, Emoji_num3, Emoji_right_predict3,  Word_right_predict3, Wrong_predict3, "Emoji_recall_3: ", 1.0*Emoji_right_predict3/Emoji_num3, 
        "Word_recall_3: ", 1.0*Word_right_predict3/Total_num3, "Wrong_3: ", 1.0*Wrong_predict3/Total_num3) 

  return np.exp(costs / iters)

def main(_):
  if not FLAGS.save_path:
    raise ValueError("Must set --save_path to data directory")
  if not FLAGS.input_data:
    raise ValueError("Must set --input_data to data directory")

  input_test_data, vocab_len, emoji_list = reader.test_data(FLAGS.input_data)
  w2v_matrix = reader.gen_w2v_matrix(vocab_len, FLAGS.input_data)
  print("len(emoji_list): ",  len(emoji_list))
  print("vocab_len: ",  vocab_len)
  #input_data = input_test_data
  #output_data[:-1] = input_test_data[1:]
  #output_data[-1] = input_test_data[0]
  
  eval_config = model.TestConfig()
  initializer = tf.random_uniform_initializer(-eval_config.init_scale, eval_config.init_scale)
  with tf.Graph().as_default():
    with tf.name_scope("Test"): 
      test_input = model.Input(config=eval_config, input_data=input_test_data, output_data=input_test_data, name="Test")
      with tf.variable_scope("Model", reuse=None, initializer=initializer):  
        mtest = model.Model(is_training=False, config=eval_config, input_=test_input, l2=False, w2v=w2v_matrix)
    
    sv = tf.train.Supervisor(logdir=FLAGS.save_path)
    config_proto = tf.ConfigProto(allow_soft_placement=False)
    with sv.managed_session(config=config_proto) as session:
      test_perplexity = test_epoch(session, mtest, emoji_list)
  

if __name__ == "__main__":
  tf.app.run()
