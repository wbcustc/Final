import numpy as np
<<<<<<< HEAD

path_train_3k = './CS5785-final-data/train.txt'
path_test_3k = './CS5785-final-data/test.txt'
path_attr_list = './CS5785-final-data/attributes_list.txt'
path_attr_train = './CS5785-final-data/attributes_train.txt'
path_attr_test = './CS5785-final-data/attributes_train.txt'
=======
>>>>>>> 3a0c4263466b98336c63093be693b29a58e2eaaa

def read_train_3k(path = '../CS5785-final-data/train.txt'):
        reader = open(path,'rb')
        piclabel_3k = list()
        picname_3k = list()
        labels_3k = list()
        labels_num = dict()
        for line in reader:
                line = line.strip()
                line = line.split(' ')
                jpg_name = line[0]
                label = line[1]
                picname_3k.append(jpg_name)
                if labels_num.has_key(label):
                        piclabel_3k.append(labels_num[label])
                else:
                        labels_num[label] = len(labels)
                        piclabel_3k.append(labels_num[label])
                        labels_3k.append(label)
	piclabel_3k = np.array(piclabel_3k)
        return piclabel_3k, picname_3k, labels_3k

def read_test_3k(path):
    reader=open('/Users/jingjingzhang/Desktop/Data_Minning_Final/CS5785-final-data/test.txt','rb')
    jpg_file=[]

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

def read_data():
	return

<<<<<<< HEAD
if __name__ == '__main__':
	ret = read_attributes_train('./CS5785-final-data/attributes_train.txt')
	print type(ret)
=======
def read_attributes_test(path):
	return
>>>>>>> 3a0c4263466b98336c63093be693b29a58e2eaaa
