from flask import Flask,render_template
from flask_wtf import Form
from flask_bootstrap import Bootstrap
from wtforms import StringField,SubmitField,TextAreaField
from wtforms.validators import Required,Length

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


from test_ import *

config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth = True

class NameForm(Form):
    name = TextAreaField('Input',validators=[Required()],render_kw={"placeholder":"此处输入文本"})
    submit = SubmitField('Submit')


app = Flask(__name__)
app.config['SECRET_KEY'] = '666'
bootstrap = Bootstrap(app)

print('Load global resources.')

model_dir   = './ckpt_'
model_name  = 'model.ckpt-15.meta'
dict_path   = 'data/dict.pkl'
time_step   = 200
seg_path    = 'ltp/cws.model'
pos_path    = 'ltp/pos.model'

# Load dict
print('Load dict.')
char2id, id2char, pos2id, id2pos, word2id, id2word, tag2id, id2tag = load_dict(dict_path)
print('Load dict done.')


# Load cws and pos model
print('Load cws and pos model.')
seg_inst, pos_inst = Segmentor(), Postagger()
seg_inst.load(seg_path), pos_inst.load(pos_path)
print('Load cws and pos model done.')


# Load model    
print('Start session.')
sess = tf.Session(config = config)
sess.run(tf.global_variables_initializer())
print('Start session done.')

print('Load graph.')
saver = tf.train.import_meta_graph(model_dir+'/'+model_name)
saver.restore(sess, tf.train.latest_checkpoint(model_dir))
print('Load graph done.')

print('Load tensor and operation.')

graph = tf.get_default_graph()
train_x_char = graph.get_tensor_by_name('train_x_char:0')
train_x_pos = graph.get_tensor_by_name('train_x_pos:0')
train_x_word_old = graph.get_tensor_by_name('train_x_word_old:0')
train_x_word_cur = graph.get_tensor_by_name('train_x_word_cur:0')
train_y = graph.get_tensor_by_name('train_y:0')
keep_prob = graph.get_tensor_by_name('keep_prob:0')    
y_predict = graph.get_tensor_by_name('y_predict:0')    
train_initializer = graph.get_operation_by_name('train_initializer')

print('Load tensor and operation done.')

print('Load global resources done.')


@app.route('/',methods=['GET','POST'])
def index():
    name = None
    nameForm = NameForm()
    print('Start to infer.')
    rst = []
    if nameForm.validate_on_submit():
        name = nameForm.name.data
        nameForm.name.data = ''
        
        
        print('Submit: ', name)
        
        char, pos, word_old, word_cur, label  = get_raw_input(name, seg_inst, pos_inst)
            
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
        print('Inference done.')

        # Merge data
        seq = list(name)
        tag = result
        for i in range(len(seq)):
            elem = {}
            elem['seq'] = seq[i]
            elem['tag'] = tag[i]
            rst.append(elem)
        
    return render_template('index.html',form=nameForm,data=rst)

if __name__ == '__main__':
    app.run('10.8.0.122', debug=False)

