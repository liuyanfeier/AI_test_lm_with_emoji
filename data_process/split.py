import random
import sys

data_list = []

with open(sys.argv[1]) as f:
	for line in f:
		data_list.append(line)

#random.shuffle(data_list)

def split_list(path, start, end):
	with open(path, 'w') as f:
		for it in data_list[start: end]:
			f.write(it)

split_list('valid.txt', 0, len(data_list)//100*3)
split_list('test.txt', len(data_list)//100*3, len(data_list)//100*5)
split_list('train.txt', len(data_list)//100*5, len(data_list))
