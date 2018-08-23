import copy
import random
from pyltp import Segmentor

def data_aug(data_path, save_path, seg_model = '../../../ltp/cws.model', stopwords_path='stopwords.txt'):

	
	text = open(data_path,'rb').readlines()
	
	seg = Segmentor()
	seg.load(seg_model)
	
	stopwords = open(stopwords_path).readlines()
	stopwords = [word.strip() for word in stopwords]
	
	input_dict = {}
	truth_dict = {}
	
	RW_num_ratio = 0.5
	TEXT_num = len(text)
	RW_END = int(TEXT_num * RW_num_ratio)

	# R,W augmentation
	for i, seq in enumerate(text[:RW_END]):
			print(i, seq.decode('utf-8'), len(text[:RW_END]))
			tags = []

			seq = seq.decode('utf-8')
			seq = seq.replace(' ','').strip()
			# Keep raw
			raw = copy.deepcopy(seq)
			word = seg.segment(seq)
			
			R_nums = 3
			
			word_list = [i for i in range(len(word))]
			if len(word_list) < R_nums:
					continue
			word_idxs = random.sample(word_list,R_nums)
			word_idxs = [idx for idx in word_idxs if word[idx] not in stopwords]
			

			# R
			pos_list = [i for i in range(len(raw))]
			pos_idxs = random.sample(pos_list,len(word_idxs))
			pos_idxs.sort()

			
			for j in range(len(pos_idxs)):
				if j == 0 and len(pos_idxs) > 1:
					seq = raw[:pos_idxs[j]]+word[word_idxs[j]]+raw[pos_idxs[j]:pos_idxs[j+1]]
				elif j == (len(pos_idxs) - 1):
					seq += word[word_idxs[j]]+raw[pos_idxs[j]:]
				else:
					seq += word[word_idxs[j]]+raw[pos_idxs[j]:pos_idxs[j+1]]
			
			
			R_tags = []
			for k in range(len(pos_idxs)): # index begin with 1, add 1
				if k == 0:
					R_tags.append([pos_idxs[k] + 1, pos_idxs[k]+len(word[word_idxs[k]]) - 1 + 1, 'R'])	
				else:
					append_len = sum([ len( word[ word_idxs[k] ] ) for q in range(0, k) ])
					R_tags.append([append_len + pos_idxs[k] + 1, append_len + pos_idxs[k]+len(word[word_idxs[k]]) - 1 + 1, 'R'])	
			
			if len(R_tags) > 0:
				tags.extend(R_tags)
			
			# W
			W_tags = []
			for k in range(len(R_tags)):
					if k != (len(R_tags) - 1):
						start, end = R_tags[k][1] - 1, R_tags[k+1][0] - 1
						if (start + 1) < (end - 1):
								start_off = random.randint(start+1, end - 1)
								end_off = random.randint(start+1, end - 1)
								if start_off != end_off:
										if start_off < end_off:
											start_off_, end_off_ = start_off, end_off
										else:
											start_off_, end_off_ = end_off, start_off
										
										w_range = [i for i in range(start_off_, end_off_)]
										random.shuffle(w_range)
										
										W_tags.append([start_off_ + 1, end_off_ + 1,'W'])				
										
										R_seq_list = list( seq )
										for q in range(len(w_range)):
													R_seq_list[start_off_] = R_seq_list[w_range[q]]
													start_off_ += 1
										seq = ''.join(R_seq_list)
										pass
			if len(W_tags) > 0:
				tags.extend(W_tags)
	
	
			input_dict[i] = seq
			truth_dict[i] = tags
	
	# M augmentation
	for i, seq in enumerate(text[RW_END:], start=RW_END):
			print(i, seq.decode('utf-8'), len(text[RW_END:]))
			tags = []

			seq = seq.decode('utf-8')
			seq = seq.replace(' ','').strip()
			
			# Keep raw
			raw = copy.deepcopy(seq)
			word = seg.segment(seq)
			
			M_nums = 3
			
			word_list = [i for i in range(len(word))]
			if len(word_list) < M_nums:
					continue
			word_idxs = random.sample(word_list,M_nums)
			word_idxs = [idx for idx in word_idxs if word[idx] not in stopwords]
			word_idxs.sort()

			M_tags = []
			for j in range(len( word_idxs )):
					if j == 0 and len(word_idxs) > 1:
						seq = ''.join(word[:word_idxs[j]])
						start_off , end_off  = len(seq), len(seq) + len(word[word_idxs[j]])
					elif j == (len(word_idxs) - 1):
						seq += ''.join(word[ word_idxs[j-1] + 1: word_idxs[j] ])
						start_off , end_off  = len(seq), len(seq) + len(word[word_idxs[j]])
						seq += ''.join(word[word_idxs[j]+1:])
					else:
						seq += ''.join(word[ word_idxs[j-1] + 1: word_idxs[j] ])
						start_off , end_off  = len(seq), len(seq) + len(word[word_idxs[j]])
					M_tags.append([start_off + 1, end_off, 'M'])
			
			if len(M_tags) > 0:
				tags.extend(M_tags)
	
			input_dict[i] = seq
			truth_dict[i] = tags
	
	exit()
	# Write to file	
	with open(save_path, 'a') as f:
		for key, value in input_dict.items():
			tags = truth_dict[key]
			f.write('<DOC>\n')
			f.write('<TEXT id="'+str(key)+'">\n')
			f.write(value+'\n')
			f.write('</TEXT>\n')
			for tag in tags:
				start_off, end_off, type_ = tag
				f.write('<ERROR'+' start_off="'+str(start_off)+'" end_off="'+str(end_off)+'" type="'+str(type_)+'"></ERROR>\n')
			f.write('</DOC>\n')
if __name__ == '__main__':
		data_aug('merge.zh', 'news.xml')
