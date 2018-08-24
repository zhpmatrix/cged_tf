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


FLAGS = None
tf.set_random_seed(-1)


config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth = True

def load_data():
    """
    Load data from pickle
    :return: Arrays
    """
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


def do():
    # Load data
    train_data_char, dev_data_char, test_data_char, char2id, id2char,train_data_word_old, dev_data_word_old, test_data_word_old, train_data_word_cur, dev_data_word_cur, test_data_word_cur, word2id, id2word, train_data_pos, dev_data_pos, test_data_pos,pos2id, id2pos,train_data_tag, dev_data_tag, test_data_tag,tag2id, id2tag = load_data()
    
    # Char
    #train_x_char = train_data_char
    #dev_x_char   = dev_data_char
    #test_x_char  = test_data_char
    #
    ## Pos
    #train_x_pos = train_data_pos
    #dev_x_pos   = dev_data_pos
    #test_x_pos  = test_data_pos
    #
    ## Word(old)
    #train_x_word_old = train_data_word_old
    #dev_x_word_old   = dev_data_word_old
    #test_x_word_old  = test_data_word_old
    # 
    ## Word(cur)
    #train_x_word_cur = train_data_word_cur
    #dev_x_word_cur   = dev_data_word_cur
    #test_x_word_cur  = test_data_word_cur
    #

    #train_y = train_data_tag
    #dev_y   = dev_data_tag
    #test_y  = test_data_tag
    
    train_number = 40000
    dev_number = 10000
	# Char
    train_x_char = train_data_char[:train_number]
    dev_x_char   = dev_data_char[:dev_number]
    test_x_char  = test_data_char
    
    # Pos
    train_x_pos = train_data_pos[:train_number]
    dev_x_pos   = dev_data_pos[:dev_number]
    test_x_pos  = test_data_pos
    
    # Word(old)
    train_x_word_old = train_data_word_old[:train_number]
    dev_x_word_old   = dev_data_word_old[:dev_number]
    test_x_word_old  = test_data_word_old
     
    # Word(cur)
    train_x_word_cur = train_data_word_cur[:train_number]
    dev_x_word_cur   = dev_data_word_cur[:dev_number]
    test_x_word_cur  = test_data_word_cur
    

    train_y = train_data_tag[:train_number]
    dev_y   = dev_data_tag[:dev_number]
    test_y  = test_data_tag
    
	# Steps
    train_steps = math.ceil(train_x_char.shape[0] / FLAGS.train_batch_size)
    dev_steps = math.ceil(dev_x_char.shape[0] / FLAGS.dev_batch_size)
    test_steps = math.ceil(test_x_char.shape[0] / FLAGS.test_batch_size)
    
    vocab_size = len(char2id) + 1
    print('Vocab Size', vocab_size)

    pos_size = len(pos2id) + 1
    print('Pos Size', pos_size)

    word_size = len(word2id) + 1
    print('Word Size', word_size)
    
    global_step = tf.Variable(-1, trainable=False, name='global_step')
    
    # Train and dev dataset
    train_dataset = tf.data.Dataset.from_tensor_slices((train_x_char, train_x_pos, train_x_word_old, train_x_word_cur, train_y))
    train_dataset = train_dataset.batch(FLAGS.train_batch_size)
    
    dev_dataset = tf.data.Dataset.from_tensor_slices((dev_x_char, dev_x_pos, dev_x_word_old, dev_x_word_cur, dev_y))
    dev_dataset = dev_dataset.batch(FLAGS.dev_batch_size)
    
    test_dataset = tf.data.Dataset.from_tensor_slices((test_x_char, test_x_pos, test_x_word_old, test_x_word_cur, test_y))
    test_dataset = test_dataset.batch(FLAGS.test_batch_size)
    
    # A reinitializable iterator
    iterator = tf.data.Iterator.from_structure(train_dataset.output_types, train_dataset.output_shapes)
    
    train_initializer = iterator.make_initializer(train_dataset)
    dev_initializer = iterator.make_initializer(dev_dataset)
    test_initializer = iterator.make_initializer(test_dataset)
    
    # Input Layer
    with tf.variable_scope('inputs'):
        x_char, x_pos, x_word_old, x_word_cur, y_label = iterator.get_next()
    
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
    
    tf.summary.histogram('y_predict', y_predict)
    
    y_label_reshape = tf.cast(tf.reshape(y_label, [-1]), tf.int32)
    print('Y Label Reshape', y_label_reshape)
    
    # Prediction
    correct_prediction = tf.equal(y_predict, y_label_reshape)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar('accuracy', accuracy)
    
    print('Prediction', correct_prediction, 'Accuracy', accuracy)
    
    # Loss
    cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_label_reshape,logits=tf.cast(y, tf.float32)))
    tf.summary.scalar('loss', cross_entropy)
    
    # Train
    train = tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(cross_entropy, global_step=global_step)
    
    # Saver
    saver = tf.train.Saver()
    
    # Iterator
    sess = tf.Session(config = config)
    sess.run(tf.global_variables_initializer())
    
    # Global step
    gstep = 0
    
    # Summaries
    summaries = tf.summary.merge_all()
    writer = tf.summary.FileWriter(FLAGS.summaries_dir,sess.graph)
    if FLAGS.train:
        
        #if tf.gfile.Exists(FLAGS.summaries_dir):
        #    tf.gfile.DeleteRecursively(FLAGS.summaries_dir)
        
        for epoch in range(FLAGS.epoch_num):
            tf.train.global_step(sess, global_step_tensor=global_step)
            # Train
            sess.run(train_initializer)
            for step in range(int(train_steps)):
                smrs, loss, acc, gstep, _ = sess.run([summaries, cross_entropy, accuracy, global_step, train],feed_dict={keep_prob: FLAGS.keep_prob})
                
                # Print log
                if step % FLAGS.steps_per_print == 0:
                    print('Global Step', gstep, 'Step', step, 'Train Loss', loss, 'Accuracy', acc)
                
                # Summaries for tensorboard
                if gstep % FLAGS.steps_per_summary == 0:
                    if not os.path.exists(FLAGS.summaries_dir):
                        os.mkdir(FLAGS.summaries_dir)
                    writer.add_summary(smrs, gstep)
                    print('Write summaries to', FLAGS.summaries_dir)
            
            if epoch % FLAGS.epochs_per_dev == 0:
                # Dev
                sess.run(dev_initializer)
                
                Y_pred = []
                Y_true = []
                
                for step in range(int(dev_steps)):
                    y_predict_results, acc = sess.run([y_predict, accuracy], feed_dict={keep_prob: 1})
                    Y_pred.extend( id2tag[y_predict_results.tolist()].tolist() )
                    #print('Dev Accuracy', sess.run(accuracy, feed_dict={keep_prob: 1}), 'Step', step)
                # Y_true 
                dev_y_ = dev_y.tolist()
                Y_true = id2tag[ list( chain( *dev_y_ ))].tolist()
                print(classification_report(Y_true, Y_pred))
            
            # Save model
            if epoch % FLAGS.epochs_per_save == 0:
                saver.save(sess, FLAGS.checkpoint_dir, global_step=gstep)
    
    else:
        # Load model
        ckpt = tf.train.get_checkpoint_state('ckpt')
        saver.restore(sess, ckpt.model_checkpoint_path)
        print('Restore from', ckpt.model_checkpoint_path)
        sess.run(test_initializer)
        

        Y_pred = []
        Y_true = []
        
        for step in range(int(test_steps)):
            x_results, x_pos_, x_word_old_, x_word_cur_, y_predict_results, acc = sess.run([x_char, x_pos, x_word_old, x_word_cur, y_predict, accuracy], feed_dict={keep_prob: 1})
            print('Test step', step, 'Accuracy', acc)
            #Y_test
            Y_pred.extend( id2tag[y_predict_results.tolist()].tolist() )
        
            
            y_predict_results = np.reshape(y_predict_results, x_results.shape)
            for i in range(len(x_results)):
               x_text = id2char[x_results[i]].tolist()
               y_predict_text = id2tag[y_predict_results[i]].tolist()
               print(x_text[:20], y_predict_text[:20])

        #Y_true 
        test_y_ = test_y.tolist()
        Y_true = id2tag[ list( chain( *test_y_ ))].tolist()
        
        #print(classification_report(Y_true, Y_pred))

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='BI LSTM')
	
    DATA_TAG = True

    if DATA_TAG == True:
        DATASET_DIR = 'data/TOEFL_NEWS_16_18/'
        MODEL_DIR = 'ckpt/TOEFL_NEWS_16_18/'
        SUMMARY_DIR = 'summaries/TOEFL_NEWS_16_18'
    else:
        DATASET_DIR = 'data/16_18/'
        MODEL_DIR = 'ckpt/16_18/'
        SUMMARY_DIR = 'summaries/16_18'

    parser.add_argument('--train_batch_size', help='train batch size', default=256)
    parser.add_argument('--dev_batch_size', help='dev batch size', default=64)
    parser.add_argument('--test_batch_size', help='test batch size', default=256)
    parser.add_argument('--dict_path', help='dict path', default=DATASET_DIR+'dict.pkl')
    parser.add_argument('--data_path', help='data path', default=DATASET_DIR+'data.pkl')
    parser.add_argument('--num_layer', help='num of layer', default=1, type=int)
    parser.add_argument('--num_units', help='num of units', default=128, type=int)
    parser.add_argument('--time_step', help='time steps', default=200, type=int)
    parser.add_argument('--char_embedding_size', help='char_embedding size', default=50, type=int)
    parser.add_argument('--pos_embedding_size', help='pos embedding size', default=16, type=int)
    parser.add_argument('--word_embedding_size', help='word embedding size', default=100, type=int)
    parser.add_argument('--category_num', help='category num', default=10, type=int)
    parser.add_argument('--learning_rate', help='learning rate', default=0.001, type=float)
    parser.add_argument('--epoch_num', help='num of epoch', default=1000, type=int)
    parser.add_argument('--epochs_per_test', help='epochs per test', default=100, type=int)
    parser.add_argument('--epochs_per_dev', help='epochs per dev', default=2, type=int)
    parser.add_argument('--epochs_per_save', help='epochs per save', default=2, type=int)
    parser.add_argument('--steps_per_print', help='steps per print', default=10, type=int)
    parser.add_argument('--steps_per_summary', help='steps per summary', default=5, type=int)
    parser.add_argument('--keep_prob', help='train keep prob dropout', default=0.5, type=float)
    parser.add_argument('--checkpoint_dir', help='checkpoint dir', default=MODEL_DIR+'model.ckpt', type=str)
    parser.add_argument('--summaries_dir', help='summaries dir', default=SUMMARY_DIR, type=str)
    parser.add_argument('--train', help='train', default=True, type=bool)
    
    FLAGS, args = parser.parse_known_args()
    
    do()
