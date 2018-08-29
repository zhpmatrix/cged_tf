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

from pyltp import Segmentor
from pyltp import Postagger


config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth = True

def word2char(word, pos):
	w = []
	p = []
	for i in range(len(word)):
			for j in range(len(word[i])):
					w.append(word[i][j])
					p.append(('B-' if j == 0 else 'I-')+pos[i])
	return w, p

def get_word(w):
	"""
		diff implementation with data.py
	"""
	w.insert(0,'<SOS>')
	w.append('<EOS>')
	w_old = [''.join([w[i], w[i+1]]) for i in range(len(w) - 2)]
	w_cur = [''.join([w[i], w[i+1]]) for i in range(1, len(w) - 1)]
	return w_old, w_cur

def get_raw_input(seq, seg_inst, pos_inst):
    # Filter seq
    seq = seq.replace(' ','')
    # Char
    char = list(seq)
    word = seg_inst.segment(seq)
    pos  = pos_inst.postag(word)
    # Pos
    w, pos_ = word2char(word, pos)
    # Word(old, cur)
    w_old, w_cur = get_word(w)
    label = ['X' for _ in range( len(char) ) ]
    return char, pos_, w_old, w_cur, label

def load_dict(dict_path):
	with open(dict_path, 'rb') as f:
		char2id     = pickle.load(f)
		id2char     = pickle.load(f)
		pos2id      = pickle.load(f)
		id2pos      = pickle.load(f)
		word2id     = pickle.load(f)
		id2word     = pickle.load(f)
		tag2id      = pickle.load(f)
		id2tag      = pickle.load(f)

	return char2id, id2char, pos2id, id2pos, word2id, id2word, tag2id, id2tag

def predict():
    
    model_dir   = './ckpt_'
    model_name  = 'model.ckpt-15.meta'
    dict_path   = 'data/dict.pkl'
    time_step   = 200
    seg_path    = 'ltp/cws.model'
    pos_path    = 'ltp/pos.model'
    
    # Load dict
    char2id, id2char, pos2id, id2pos, word2id, id2word, tag2id, id2tag = load_dict(dict_path)
    # Load cws and pos model
    seg_inst, pos_inst = Segmentor(), Postagger()
    seg_inst.load(seg_path), pos_inst.load(pos_path)
    
    # Load model    
    sess = tf.Session(config = config)
    sess.run(tf.global_variables_initializer())
    
    saver = tf.train.import_meta_graph(model_dir+'/'+model_name)
    saver.restore(sess, tf.train.latest_checkpoint(model_dir))
    
    graph = tf.get_default_graph()
    
    train_x_char = graph.get_tensor_by_name('train_x_char:0')
    train_x_pos = graph.get_tensor_by_name('train_x_pos:0')
    train_x_word_old = graph.get_tensor_by_name('train_x_word_old:0')
    train_x_word_cur = graph.get_tensor_by_name('train_x_word_cur:0')
    train_y = graph.get_tensor_by_name('train_y:0')
    keep_prob = graph.get_tensor_by_name('keep_prob:0')    
    y_predict = graph.get_tensor_by_name('y_predict:0')    
    train_initializer = graph.get_operation_by_name('train_initializer')
    
    #accuracy = graph.get_tensor_by_name('accuracy:0')
    #cross_entropy = graph.get_tensor_by_name('cross_entropy:0')
    
    
    while(1): 
        
        print('Enter sequence:')    
        seq = input()
        
        # Preprocess
        char, pos, word_old, word_cur, label  = get_raw_input(seq, seg_inst, pos_inst)
            
        # If can not find key, padding with index equals 0
        test_x_char 	= [char2id[ch] if ch in char2id else 0 for ch in char]
        test_x_pos 	= [pos2id[p] if p in pos2id else 0 for p in pos] 
        test_word_old   = [word2id[w] if w in word2id else 0 for w in word_old]
        test_word_cur 	= [word2id[w] if w in word2id else 0 for w in word_cur]
        test_label      = [tag2id[t] if t in tag2id else 0 for t in label]
        
        # Padding
        [test_data.extend([0 for i in range(time_step - len(test_data))]) for test_data in [test_x_char, test_x_pos, test_word_old, test_word_cur, test_label] if len(test_data) < time_step]
        
        # Clipping
        test_x_char_ = np.array( test_x_char).reshape((1, time_step))
        test_x_pos_ = np.array( test_x_pos).reshape((1, time_step))
        test_x_word_old_ = np.array( test_word_old).reshape((1, time_step))
        test_x_word_cur_ = np.array( test_word_cur).reshape((1, time_step))
        test_label_ = np.array( test_label).reshape((1, time_step))
                
        sess.run([train_initializer],feed_dict={train_x_char:test_x_char_,train_x_pos:test_x_pos_,train_x_word_old:test_x_word_old_,train_x_word_cur:test_x_word_cur_,train_y:test_label_ })
        
        rst  = sess.run([y_predict],feed_dict={keep_prob:1})
        result = id2tag[rst[0]].tolist() 
        print(result)	
    
    print('Done!')

if __name__ == '__main__':
    predict()
