from pyltp import Segmentor
from pyltp import Postagger

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

def get_raw_input(seq):
	seg = Segmentor()
	seg.load('ltp/cws.model')
	pos = Postagger()
	pos.load('ltp/pos.model')
	# Filter seq
	seq = seq.replace(' ','')
	# Char
	char = list(seq)
	word = seg.segment(seq)
	pos  = pos.postag(word)
	# Pos
	w, pos_ = word2char(word, pos)
	# Word(old, cur)
	w_old, w_cur = get_word(w)
	return char, pos_, w_old, w_cur

if __name__ == '__main__':
		seq = '年轻人觉得人生是自己满足的别人不能管'
		char, pos_, w_old, w_cur = get_raw_input(seq)
		print(char, pos_, w_old, w_cur)

