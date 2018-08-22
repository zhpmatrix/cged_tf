import os
import argparse
import pickle
import math
import numpy as np
import tensorflow as tf

from os.path import join
from itertools import chain
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from preprocess import *

class FLAGS:
		pass

tf.set_random_seed(-1)


config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth = True

def load_data():
    with open(FLAGS.dict_path, 'rb') as f:
        
        char2id     = pickle.load(f)
        id2char     = pickle.load(f)
        pos2id      = pickle.load(f)
        id2pos      = pickle.load(f)
        word2id     = pickle.load(f)
        id2word     = pickle.load(f)
        tag2id      = pickle.load(f)
        id2tag      = pickle.load(f)
    
    with open(FLAGS.data_path, 'rb') as f:
    
        train_data_char     = pickle.load(f)
        dev_data_char       = pickle.load(f)
        test_data_char      = pickle.load(f)
        
        train_data_word_old = pickle.load(f)
        dev_data_word_old   = pickle.load(f)
        test_data_word_old  = pickle.load(f)
        
        train_data_word_cur      = pickle.load(f)
        dev_data_word_cur        = pickle.load(f)
        test_data_word_cur       = pickle.load(f)
        
        train_data_pos      = pickle.load(f)
        dev_data_pos        = pickle.load(f)
        test_data_pos       = pickle.load(f)
        
        train_data_tag      = pickle.load(f)
        dev_data_tag        = pickle.load(f)
        test_data_tag       = pickle.load(f)
    
    return  train_data_char, dev_data_char, test_data_char, char2id, id2char,train_data_word_old, dev_data_word_old, test_data_word_old, train_data_word_cur, dev_data_word_cur, test_data_word_cur, word2id, id2word, train_data_pos, dev_data_pos, test_data_pos,pos2id, id2pos,train_data_tag, dev_data_tag, test_data_tag,tag2id, id2tag

def load_dict():
	with open(FLAGS.dict_path, 'rb') as f:
		char2id     = pickle.load(f)
		id2char     = pickle.load(f)
		pos2id      = pickle.load(f)
		id2pos      = pickle.load(f)
		word2id     = pickle.load(f)
		id2word     = pickle.load(f)
		tag2id      = pickle.load(f)
		id2tag      = pickle.load(f)

	return char2id, id2char, pos2id, id2pos, word2id, id2word, tag2id, id2tag

def get_data(data_x, data_y):
    """
    Split data from loaded data
    :param data_x:
    :param data_y:
    :return: Arrays
    """
    print('Data X Length', len(data_x), 'Data Y Length', len(data_y))
    print('Data X Example', data_x[0])
    print('Data Y Example', data_y[0])
    
    train_x, test_x, train_y, test_y = train_test_split(data_x, data_y, test_size=0.2, random_state=40)
    train_x, dev_x, train_y, dev_y = train_test_split(train_x, train_y, test_size=0.2, random_state=40)
    
    print('Train X Shape', train_x.shape, 'Train Y Shape', train_y.shape)
    print('Dev X Shape', dev_x.shape, 'Dev Y Shape', dev_y.shape)
    print('Test Y Shape', test_x.shape, 'Test Y Shape', test_y.shape)
    return train_x, train_y, dev_x, dev_y, test_x, test_y


def weight(shape, stddev=0.1, mean=0):
    initial = tf.truncated_normal(shape=shape, mean=mean, stddev=stddev)
    return tf.Variable(initial)


def bias(shape, value=0.1):
    initial = tf.constant(value=value, shape=shape)
    return tf.Variable(initial)


def lstm_cell(num_units, keep_prob=0.5):
    cell = tf.nn.rnn_cell.LSTMCell(num_units, reuse=tf.AUTO_REUSE)
    return tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=keep_prob)

def get_array_input(seq, max_len):
    char, pos, word_old, word_cur     = get_raw_input(seq)
    
    char2id, id2char, pos2id, _, word2id, _, _, id2tag = load_dict()
    
	# If can not find key, padding with index equals 0
    test_x_char 	= [char2id[ch] if ch in char2id else 0 for ch in char]
    test_x_pos 		= [pos2id[p] if p in pos2id else 0 for p in pos] 
    test_word_old   = [word2id[w] if w in word2id else 0 for w in word_old]
    test_word_cur 	= [word2id[w] if w in word2id else 0 for w in word_cur]
	
	
	# Padding to max_len
    [test_data.extend([0 for i in range(max_len - len(test_data))]) for test_data in [test_x_char, test_x_pos, test_word_old, test_word_cur] if len(test_data) < max_len]
    
	# List to ndarray
    return np.array([test_x_char]), np.array([test_x_pos]), np.array([test_word_old]), np.array([test_word_cur]), char2id, id2char, pos2id, word2id, id2tag

def do(seq):
    # Load data
    #train_data_char, dev_data_char, test_data_char, char2id, id2char,train_data_word_old, dev_data_word_old, test_data_word_old, train_data_word_cur, dev_data_word_cur, test_data_word_cur, word2id, id2word, train_data_pos, dev_data_pos, test_data_pos,pos2id, id2pos,train_data_tag, dev_data_tag, test_data_tag,tag2id, id2tag = load_data()
    test_x_char, test_x_pos, test_x_word_old, test_x_word_cur,char2id, id2char, pos2id, word2id, id2tag = get_array_input(seq, FLAGS.time_step) 
#	# Char
#    test_x_char  = test_data_char[0]
#    test_x_char = test_x_char.reshape((1, test_x_char.shape[0])) 
#    
#    # Pos
#    test_x_pos  = test_data_pos[0]
#    test_x_pos = test_x_pos.reshape((1, test_x_pos.shape[0])) 
#    
#    # Word(old)
#    test_x_word_old  = test_data_word_old[0]
#    test_x_word_old = test_x_word_old.reshape((1, test_x_word_old.shape[0])) 
#    
#    # Word(cur)
#    test_x_word_cur  = test_data_word_cur[0]
#    test_x_word_cur = test_x_word_cur.reshape((1, test_x_word_cur.shape[0])) 
    

    
	# Steps
    test_steps = math.ceil(test_x_char.shape[0] / FLAGS.test_batch_size)
    
    vocab_size = len(char2id) + 1
    print('Vocab Size', vocab_size)

    pos_size = len(pos2id) + 1
    print('Pos Size', pos_size)

    word_size = len(word2id) + 1
    print('Word Size', word_size)
    
    global_step = tf.Variable(-1, trainable=False, name='global_step')
    
    # Train and dev dataset
    test_dataset = tf.data.Dataset.from_tensor_slices((test_x_char, test_x_pos, test_x_word_old, test_x_word_cur))
    test_dataset = test_dataset.batch(FLAGS.test_batch_size)
    
    # A reinitializable iterator
    iterator = tf.data.Iterator.from_structure(test_dataset.output_types, test_dataset.output_shapes)
    
    test_initializer = iterator.make_initializer(test_dataset)
    
    # Input Layer
    with tf.variable_scope('inputs'):
        x_char, x_pos, x_word_old, x_word_cur= iterator.get_next()
    
    # Embedding Layer
    with tf.variable_scope('embedding'):
        #embedding = tf.Variable(tf.random_normal([vocab_size, FLAGS.embedding_size]), dtype=tf.float32)
        embedding_char = tf.Variable(tf.truncated_normal([vocab_size, FLAGS.char_embedding_size]), dtype=tf.float32)
        embedding_pos = tf.Variable(tf.truncated_normal([pos_size, FLAGS.pos_embedding_size]), dtype=tf.float32)
        embedding_word_old = tf.Variable(tf.truncated_normal([word_size, FLAGS.word_embedding_size]), dtype=tf.float32)
        embedding_word_cur = tf.Variable(tf.truncated_normal([word_size, FLAGS.word_embedding_size]), dtype=tf.float32)
    
    inputs_char = tf.nn.embedding_lookup(embedding_char, x_char)
    inputs_pos = tf.nn.embedding_lookup(embedding_pos, x_pos) 
    inputs_word_old = tf.nn.embedding_lookup(embedding_word_old, x_word_old) 
    inputs_word_cur = tf.nn.embedding_lookup(embedding_word_cur, x_word_cur) 
    
    # Merge dim
    merge_dim = 2

    inputs_ = tf.concat([inputs_char, inputs_pos, inputs_word_old, inputs_word_cur], axis=merge_dim)
    
    # Variables
    keep_prob = tf.placeholder(tf.float32, [])
    
    # RNN Layer
    # cell_fw = tf.nn.rnn_cell.MultiRNNCell([lstm_cell(FLAGS.num_units, keep_prob) for _ in range(FLAGS.num_layer)])
    # cell_bw = tf.nn.rnn_cell.MultiRNNCell([lstm_cell(FLAGS.num_units, keep_prob) for _ in range(FLAGS.num_layer)])
    cell_fw = [lstm_cell(FLAGS.num_units, keep_prob) for _ in range(FLAGS.num_layer)]
    cell_bw = [lstm_cell(FLAGS.num_units, keep_prob) for _ in range(FLAGS.num_layer)]
    # initial_state_fw = cell_fw.zero_state(tf.shape(x)[0], tf.float32)
    # initial_state_bw = cell_bw.zero_state(tf.shape(x)[0], tf.float32)
    inputs_ = tf.unstack(inputs_, FLAGS.time_step, axis=1)
    output, _, _ = tf.contrib.rnn.stack_bidirectional_rnn(cell_fw, cell_bw, inputs=inputs_, dtype=tf.float32)
    # output_fw, _ = tf.nn.dynamic_rnn(cell_fw, inputs=inputs, initial_state=initial_state_fw)
    # output_bw, _ = tf.nn.dynamic_rnn(cell_bw, inputs=inputs, initial_state=initial_state_bw)
    # print('Output Fw, Bw', output_fw, output_bw)
    # output_bw = tf.reverse(output_bw, axis=[1])
    # output = tf.concat([output_fw, output_bw], axis=2)
    output = tf.stack(output, axis=1)
    print('Output', output)
    output = tf.reshape(output, [-1, FLAGS.num_units * 2])
    print('Output Reshape', output)
    
    # Output Layer
    with tf.variable_scope('outputs'):
        w = weight([FLAGS.num_units * 2, FLAGS.category_num])
        b = bias([FLAGS.category_num])
        y = tf.matmul(output, w) + b
        y_predict = tf.cast(tf.argmax(y, axis=1), tf.int32)
        print('Output Y', y_predict)
    # Saver
    saver = tf.train.Saver()
    
    # Iterator
    sess = tf.Session(config = config)
    sess.run(tf.global_variables_initializer())
    
    # Load model
    ckpt = tf.train.get_checkpoint_state('ckpt')
    saver.restore(sess, ckpt.model_checkpoint_path)
    print('Restore from', ckpt.model_checkpoint_path)
    sess.run(test_initializer)
        
    result = []
    
    Y_pred = []
    for step in range(int(test_steps)):# test_steps equals 1
        x_rst, _, _, _, y_pred_rst= sess.run([x_char, x_pos, x_word_old, x_word_cur, y_predict], feed_dict={keep_prob: 1})
        # Get Y_pred
        Y_pred.extend( id2tag[y_pred_rst.tolist()].tolist() )
        
		# Filter padding    
        #y_pred_ = np.reshape(y_pred_rst, x_rst.shape)
        #x_rst_, y_pred_rst_ = list(filter(lambda x: x, x_rst[0])), list(filter(lambda x: x, y_pred_rst[0]))
        
        seq_len = len(seq)
        x_text = id2char[x_rst[0]].tolist()[:seq_len]
        y_pred_text = id2tag[y_pred_rst].tolist()[:seq_len]
        result = y_pred_text
	
    return result

def predict(seq):
	FLAGS.test_batch_size = 1
	FLAGS.dict_path='data/dict.pkl'
	FLAGS.num_layer=1
	FLAGS.num_units=128
	FLAGS.time_step=200
	FLAGS.char_embedding_size=50
	FLAGS.pos_embedding_size=16
	FLAGS.word_embedding_size=100
	FLAGS.category_num=10
	FLAGS.checkpoint_dir='ckpt/model.ckpt'
	result = do(seq)
	return result
if __name__ == '__main__':
    seq = 'xiaoming，从开始每个人都有理想的工作有的想当医生明星运动员老师等等却有很少孩子想当商人！'
    print(predict(seq))	
