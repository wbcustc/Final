import numpy as np

path_train_3k = './CS5785-final-data/train.txt'
path_test_3k = './CS5785-final-data/test.txt'
path_attr_list = './CS5785-final-data/attributes_list.txt'
path_attr_train = './CS5785-final-data/attributes_train.txt'
path_attr_test = './CS5785-final-data/attributes_train.txt'

def read_train_3k(path):
	return

def read_test_3k(path):
	return

def read_attributes_list(path):
	ret = []
	with open(path) as filein:
		for line in filein:
			line = line.strip()
			ret.append(line)
	return ret

def read_attributes(path):
	ret = []
	with open(path) as filein:
		for line in filein:
			feature = line.split(' ')[1].strip()
			vector = [int(e) for e in feature.split(',')]
			ret.append(vector)
	return np.array(ret)

def read_data():
	return

if __name__ == '__main__':
	ret = read_attributes_train('./CS5785-final-data/attributes_train.txt')
	print type(ret)
