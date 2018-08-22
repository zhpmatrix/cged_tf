import re
import pickle
import pandas as pd
import numpy as np
from itertools import chain
from os import makedirs
from os.path import exists, join

def get_dict(data):
    all_datas_set = pd.Series(list(chain(*data))).value_counts().index
    all_datas_ids = range(1, len(all_datas_set) + 1)
    # Dict to transform
    data2id = pd.Series(all_datas_ids, index=all_datas_set)
    id2data = pd.Series(all_datas_set, index=all_datas_ids)
    return data2id, id2data

def transform(data, dict_, max_length):
    ids = list(dict_[data])
    if len(ids) >= max_length:
        ids = ids[:max_length]
    ids.extend([0] * (max_length - len(ids)))
    return ids

def save_data(datas, path, filename):
    path = 'data/'
    if not exists(path):
        makedirs(path)
    print('Starting pickle to file...')
    with open(join(path,filename), 'wb') as f:
        for data in datas:
            pickle.dump(data, f)
    print('Pickle finished')

def get_word(char):
    char_list = char.tolist()
    char_list.insert(0, '<SOS>')
    char_list.append('<EOS>')
    bigram_list = [''.join([char_list[i], char_list[i+1]]) for i in range(len(char_list)-1)]
    return np.array(bigram_list)

def get_word_old(char):
    char_list = char.tolist()
    char_list.insert(0, '<SOS>')
    bigram_list = [''.join([char_list[i], char_list[i+1]]) for i in range(len(char_list)-1)]
    return np.array(bigram_list)

def get_word_cur(char):
    char_list = char.tolist()
    char_list.append('<EOS>')
    bigram_list = [''.join([char_list[i], char_list[i+1]]) for i in range(len(char_list)-1)]
    return np.array(bigram_list)


def get_dicts(data_path, save_path, filename):
    
    text = open(data_path, encoding='utf-8').readlines()

    # To numpy array
    char, word, pos, tag= [], [], [], []
    print('Start creating chars, pos and tags...')
    for i,sentence in enumerate(text):
        groups = re.findall('(.)&(.-.)&(.-.|.)', sentence)
        arrays = np.asarray(groups)
        char.append(arrays[:, 0])
        word.append(get_word(arrays[:, 0]))
        pos.append(arrays[:, 1])
        tag.append(arrays[:, 2])
    
    padding = '<UNK>'
    word2id, id2word = get_dict(word)
    char2id, id2char = get_dict(char)
    pos2id, id2pos = get_dict(pos)
    
	# Padding with <UNK>
    word2id[padding] = 0
    id2word[0] = padding
    char2id[padding] = 0
    id2char[0] = padding
    pos2id[padding]  = 0
    id2pos[0]  =  padding

    tags_set = ['O','B-R', 'I-R', 'B-M', 'I-M', 'B-S', 'I-S', 'B-W', 'I-W']
    #tags_ids = range(1, len(tags_set)+1)
    tags_set.insert(0, padding)
    tags_ids = range(len(tags_set))
    # Dict to transform
    tag2id = pd.Series(tags_ids, index=tags_set)
    id2tag = pd.Series(tags_set, index=tag2id)


    datas = [char2id, id2char, pos2id,  id2pos, word2id, id2word, tag2id, id2tag]

    save_data(datas, save_path, filename)

def merge_test(test_input_path, test_truth_path, testset_path):
    input_ = open(test_input_path, encoding='utf-8').readlines()
    input_dict = {}
    for i,sentence in enumerate(input_):
        sid = re.findall('sid=(.*?)\)\t', sentence)[0]
        doc = re.findall('\t(.*?)\n', sentence)[0]
        input_dict[sid] = doc
    
    truth = open(test_truth_path, encoding='utf-8').readlines()
    truth_dict = {}
    for i,sentence in enumerate(truth):
        row  = sentence.split(',') 
        sid = row[0]
        tag = []
        for i in range(1,len(row)):
            tag.append(row[i].strip())
        
        if sid not in truth_dict.keys():
            values = []
            values.append(tag)
            truth_dict[sid] = values
        else:
            truth_dict[sid].append(tag)

    with open(testset_path, 'a') as f:
        for key, value in input_dict.items():
            tags = truth_dict[key]
            if len(tags[0]) != 1:# Filter correct
                f.write('<DOC>\n')
                f.write('<TEXT id="'+key+'">\n')
                f.write(value+'\n')
                f.write('</TEXT>\n')
                for tag in tags:
                    start_off, end_off, type_ = tag
                    f.write('<ERROR'+' start_off="'+start_off+'" end_off="'+end_off+'" type="'+type_+'"></ERROR>\n')
                f.write('</DOC>\n')
    print('Done!')

def read_data(data_path, dicts, max_length):
    char2id, pos2id, word2id, tag2id = dicts
    
    text = open(data_path, encoding='utf-8').readlines()    
    # To numpy array
    char, word_old, word_cur, pos, tag = [], [], [], [], []

    print('Start creating chars, pos and tags...')
    for i,sentence in enumerate(text):
        groups = re.findall('(.)&(.-.)&(.-.|.)', sentence)
        arrays = np.asarray(groups)
        char.append(arrays[:, 0])
        word_old.append(get_word_old(arrays[:, 0]))
        word_cur.append(get_word_cur(arrays[:, 0]))
        pos.append(arrays[:, 1])
        tag.append(arrays[:, 2])
    print('Starting transform...')

    data_char = list(map(lambda x: transform(x, char2id, max_length), char))
    data_word_old = list(map(lambda x: transform(x, word2id, max_length), word_old))
    data_word_cur = list(map(lambda x: transform(x, word2id, max_length), word_cur))
    
    data_pos = list(map(lambda x: transform(x, pos2id, max_length), pos))
    data_tag = list(map(lambda x: transform(x, tag2id, max_length), tag))
    
    data_char = np.asarray(data_char)
    data_word_old = np.asarray(data_word_old)
    data_word_cur = np.asarray(data_word_cur)

    data_pos = np.asarray(data_pos)
    data_tag = np.asarray(data_tag)
    
    return data_char, data_word_old, data_word_cur, data_pos, data_tag

def save_datas(dict_path, max_length, train_data_path, dev_data_path, test_data_path):
    
    with open(dict_path, 'rb') as f:
        char2id = pickle.load(f)
        id2char = pickle.load(f)
        pos2id  = pickle.load(f)
        id2pos  = pickle.load(f)
        word2id = pickle.load(f)
        id2word = pickle.load(f)
        tag2id  = pickle.load(f)
        id2tag  = pickle.load(f)
    dicts = [char2id, pos2id, word2id, tag2id]

    train_data_char, train_data_word_old, train_data_word_cur, train_data_pos, train_data_tag = read_data(train_data_path, dicts, max_length) 
    dev_data_char, dev_data_word_old, dev_data_word_cur, dev_data_pos, dev_data_tag = read_data(dev_data_path, dicts, max_length) 
    test_data_char, test_data_word_old, test_data_word_cur, test_data_pos, test_data_tag = read_data(test_data_path, dicts, max_length) 

    
    datas = [   train_data_char,        dev_data_char,      test_data_char,
                train_data_word_old,    dev_data_word_old,  test_data_word_old,
                train_data_word_cur,    dev_data_word_cur,  test_data_word_cur,
                train_data_pos,         dev_data_pos,       test_data_pos,
                train_data_tag,         dev_data_tag,       test_data_tag
            ]

    save_data(datas, 'data/', 'data.pkl')

if __name__ == '__main__':
    
    #merge_test('data/raw/CGED16_HSK_Test_Input.txt', 'data/raw/CGED16_HSK_Test_Truth.txt','data/raw/CGED16_HSK_TestSet.xml')

    #get_dicts('data/input/merge_seq.txt','data/','dict.pkl')
    #exit()  
    
    data_dir = 'data/input/'
    max_length = 200
    save_datas('data/dict.pkl', max_length, data_dir+'train_seq.txt',
                                data_dir+'dev_seq.txt',
                                data_dir+'test_seq.txt')
