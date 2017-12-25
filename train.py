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

#np.set_printoptions(threshold=np.inf)  

flags = tf.flags
logging = tf.logging
flags.DEFINE_string("l2_regular", "False", "Use l2 or not.")
flags.DEFINE_string("mode", "train", "Train a model or test using a trained model.")
flags.DEFINE_string("input_data", None,
                    "Where the input training/test data is stored.")
flags.DEFINE_string("output_data", None,
                    "Where the output training/test data is stored.")
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
    
    total_num3 = 0.0001
    emoji_num3 = 0.0001
    emoji_right_predict3 = 0.0001 
    word_right_predict3 = 0.0001 
    wrong_predict3 = 0.0001
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

    Total_num1 = Total_num1 + total_num1 - 0.0001
    Emoji_num1 = Emoji_num1 + emoji_num1 - 0.0001
    Emoji_right_predict1 = Emoji_right_predict1 + emoji_right_predict1 - 0.0001 
    Word_right_predict1 = Word_right_predict1 + word_right_predict1 - 0.0001 
    Wrong_predict1 = Wrong_predict1 + wrong_predict1 - 0.0001 
    Total_num3 = Total_num3 + total_num3 - 0.0001
    Emoji_num3 = Emoji_num3 + emoji_num3 - 0.0001
    Emoji_right_predict3 = Emoji_right_predict3 + emoji_right_predict3 - 0.0001 
    Word_right_predict3 = Word_right_predict3 + word_right_predict3 - 0.0001 
    Wrong_predict3 = Wrong_predict3 + wrong_predict3 - 0.0001 
 
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

def train_epoch(session, model, eval_op=None, lr=0.0):
  start_time = time.time()
  costs = 0.0
  iters = 0
  state = session.run(model.initial_state)

  fetches = {
      "cost": model.cost,
      "final_state": model.final_state,
  }
  if eval_op is not None:
    fetches["eval_op"] = eval_op

  for step in range(model.input.epoch_size):
    feed_dict = {}
    for i, (c, h) in enumerate(model.initial_state):
      feed_dict[c] = state[i].c
      feed_dict[h] = state[i].h

    vals = session.run(fetches, feed_dict)
    cost = vals["cost"]
    state = vals["final_state"]

    costs += cost
    iters += model.input.num_steps

    print("cost: ", cost, " lr: ", lr)
    print("%.3f perplexity: %.5f cost: %f speed: %.0f wps" %
          (step * 1.0 / model.input.epoch_size, np.exp(costs / iters), costs/iters,
          iters * model.input.batch_size / (time.time() - start_time)))

  return np.exp(costs / iters)

def main(_):
  if FLAGS.mode == "train": 
    if not FLAGS.output_data:
      raise ValueError("Must set --output_data to data directory")
    if not FLAGS.input_data:
      raise ValueError("Must set --input_data to data directory")
    #os.environ["CUDA_VISIBLE_DEVICES"] = '{}'.format(3)

    input_raw_data = reader.raw_data(FLAGS.input_data)
    input_train_data, input_valid_data, input_test_data, vocab_len, _ = input_raw_data
    w2v_matrix = reader.gen_w2v_matrix(vocab_len, FLAGS.input_data)
    output_raw_data = reader.raw_data(FLAGS.output_data)
    output_train_data, output_valid_data, output_test_data, _ , emoji_list = output_raw_data

    print("len(emoji_list): ",  len(emoji_list))

    config = model.Config()
    eval_config = model.Config()
    eval_config.batch_size = 1
    eval_config.num_steps = 1

    with tf.Graph().as_default():
      initializer = tf.random_uniform_initializer(-config.init_scale, config.init_scale)

      with tf.name_scope("Train"):
        train_input = model.Input(config=config, input_data=input_train_data, output_data=output_train_data, name="TrainInput")
        with tf.variable_scope("Model", reuse=None, initializer=initializer):
          m = model.Model(is_training=True, config=config, input_=train_input, l2=FLAGS.l2_regular, w2v=w2v_matrix)
        tf.summary.scalar("Training Loss", m.cost)
        tf.summary.scalar("Learning Rate", m.lr)

      with tf.name_scope("Valid"):
        valid_input = model.Input(config=config, input_data=input_valid_data, output_data=output_valid_data, name="ValidInput")
        with tf.variable_scope("Model", reuse=True, initializer=initializer):
          mvalid = model.Model(is_training=False, config=config, input_=valid_input, l2=FLAGS.l2_regular, w2v=w2v_matrix)
        tf.summary.scalar("Validation Loss", mvalid.cost)

      with tf.name_scope("Test"):
        test_input = model.Input(
            config=eval_config, input_data=input_test_data, output_data=output_test_data, name="TestInput")
        with tf.variable_scope("Model", reuse=True, initializer=initializer):
          mtest = model.Model(is_training=False, config=eval_config,
                           input_=test_input, l2=FLAGS.l2_regular, w2v=w2v_matrix)

      ppl = 2000.0
      sv = tf.train.Supervisor(logdir=FLAGS.save_path)
      config_proto = tf.ConfigProto(allow_soft_placement=False)
      config_proto.gpu_options.allow_growth = True
      with sv.managed_session(config=config_proto) as session:
          lr_decay = 1.0
          for i in range(config.max_epoch):
            m.assign_lr(session, config.learning_rate * lr_decay)

            lr = session.run(m.lr)
            print("Epoch: %d Learning rate: %f" % (i + 1, lr))
            train_perplexity = train_epoch(session, m, eval_op=m.train_op, lr=lr)
            print("Epoch: %d Train Perplexity: %.3f" % (i + 1, train_perplexity))
            valid_perplexity = train_epoch(session, mvalid)
            print("Epoch: %d Valid Perplexity: %.3f" % (i + 1, valid_perplexity))

            if i != 0 and (ppl - valid_perplexity) / valid_perplexity  < 0.02:
              lr_decay = lr_decay * config.lr_decay
            ppl = valid_perplexity  
          #valid_perplexity = test_epoch(session, mvalid, emoji_list)
          test_perplexity = test_epoch(session, mtest, emoji_list)
          print("Test Perplexity: %.3f" % test_perplexity)

          if FLAGS.save_path:
            print("Saving model to %s." % FLAGS.save_path)
            sv.saver.save(session, FLAGS.save_path, global_step=sv.global_step)
      
  elif FLAGS.mode == "test":  
    if not FLAGS.save_path:
      raise ValueError("Must set --save_path to data directory")
    if not FLAGS.input_data:
      raise ValueError("Must set --input_data to data directory")

    input_test_data, vocab_len, emoji_list = reader.test_data(FLAGS.input_data)
    w2v_matrix = reader.gen_w2v_matrix(vocab_len, FLAGS.input_data)
    
    print("len(emoji_list): ",  len(emoji_list))

    eval_config = model.TestConfig()
    initializer = tf.random_uniform_initializer(-eval_config.init_scale, eval_config.init_scale)
    with tf.name_scope("Test"):
      test_input = model.Input(config=eval_config, input_data=input_test_data, output_data=input_test_data, name="TestInput")
      with tf.variable_scope("Model", reuse=None, initializer=initializer):
        mtest = model.Model(is_training=False, config=eval_config, input_=test_input, l2=False, w2v=w2v_matrix)

    sess_config = tf.ConfigProto()
    sess_config.gpu_options.allow_growth = True
    with tf.Session(config=sess_config) as sess:
      tf.global_variables_initializer().run()
      saver = tf.train.Saver(tf.global_variables())
      ckpt = tf.train.get_checkpoint_state(FLAGS.save_path)
      if ckpt and ckpt.model_checkpoint_path:
        print(ckpt, ckpt.model_checkpoint_path)
        saver.restore(sess, ckpt.model_checkpoint_path)
        test_perplexity = test_epoch(sess, mtest, emoji_list)
    
if __name__ == "__main__":
  tf.app.run()
