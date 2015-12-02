import numpy as np

def read_train_3k(path = '../CS5785-final-data/train.txt'):
        reader = open(path,'rb')
        piclabel = list()
        picname = list()
        labels = list()
        labels_num = dict()
        for line in reader:
                line = line.strip()
                line = line.split(' ')
                jpg_name = line[0]
                label = line[1]
                picname.append(jpg_name)
                if labels_num.has_key(label):
                        piclabel.append(labels_num[label])
                else:
                        labels_num[label] = len(labels)
                        piclabel.append(labels_num[label])
                        labels.append(label)
	piclabel = np.array(piclabel)
        return piclabel, picname, labels

def read_test_3k(path):
	return

def read_attributes_train(path):
	return

def read_attributes_test(path):
	return

