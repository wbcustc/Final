import numpy as np

train_3k = '../CS5785-final-data/train.txt'
test_3k = '../CS5785-final-data/test.txt'
attr_list = '../CS5785-final-data/attributes_list.txt'
attr_train = '../CS5785-final-data/attributes_train.txt'
attr_test = '../CS5785-final-data/attributes_train.txt'
alexnet_train = '../CS5785-final-data/alexnet_feat_train.npy'
alexnet_test = '../CS5785-final-data/alexnet_feat_test.npy'
siftbow_train = '../CS5785-final-data/SIFTBoW_train.npy'
siftbow_test = '../CS5785-final-data/SIFTBoW_test.npy'

def read_train_3k(path):
    reader = open(path)
    piclabel_3k = []
    picname_3k = []
    labels_3k = []
    labels_num = {}
    for line in reader:
        line = line.strip()
        line = line.split(' ')
        jpg_name = line[0]
        label = line[1]
        picname_3k.append(jpg_name)
        if labels_num.has_key(label):
            piclabel_3k.append(labels_num[label])
        else:
            labels_num[label] = len(labels_3k)
            piclabel_3k.append(labels_num[label])
            labels_3k.append(label)
    piclabel_3k = np.array(piclabel_3k)
    return piclabel_3k, picname_3k, labels_3k

def read_test_3k(path):
    reader = open(path)
    jpg_file = []
    for line in reader:
        line=line.strip()
        jpg_file.append(line)
    return jpg_file

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

def read_npy(path):
	ret = np.load(path)
	return ret

<<<<<<< HEAD
if __name__ == '__main__':
	ret = read_attributes_train('./CS5785-final-data/attributes_train.txt')
	print type(ret)
=======
>>>>>>> origin/master
