import numpy as np
import read_data as rd
from sklearn.cross_validation import KFold
from sklearn.svm import SVC
from sklearn.svm import LinearSVC

train_label,picname_3k,labelname_3k = rd.read_train_3k(rd.train_3k)
test_name = rd.read_test_3k(rd.test_3k)
attr_name = rd.read_attributes_list(rd.attr_list)
train_data = rd.read_attributes(rd.attr_train)
# test_data = rd.read_attributes(rd.attr_test)
train_data_alex = rd.read_npy(rd.alexnet_train)
test_data_alex = rd.read_npy(rd.alexnet_test)
train_data_siftbow = rd.read_npy(rd.siftbow_train)
# test_data_siftbow = rd.read_npy(rd.siftbow_test)


def cross_validate(train_data,n_folds = 5):
	kf = KFold(len(train_data), n_folds = n_folds)
	ret = 0.0
	for train_index, test_index in kf:
		X_train, X_test = train_data[train_index], train_data[test_index]
	   	y_train, y_test = train_label[train_index], train_label[test_index]
	   	temp = svm(X_train,y_train,X_test,y_test)
	   	print temp
	   	ret += temp
	return ret/n_folds

def svm(X_train,y_train,X_test,y_test):
	# clf = SVC(C=1.0,kernel='sigmoid')
	clf = LinearSVC(dual=False)
	score = clf.fit(X_train,y_train).score(X_test,y_test)
	return score

def svm_predict(X,y,test):
	clf = LinearSVC(dual=False)
	test_label = clf.fit(X,y).predict(test)
	return test_label

def generate_csv(filename,test_label):
	with open(filename, 'w') as f:
		temp = 'ID,Category'
		f.writelines(temp+'\n')
		for i in xrange(len(test_label)):
			temp = test_name[i]+','+labelname_3k[test_label[i]] + '\n'
			f.writelines(temp)

def main():
	# print cross_validate(train_data)
	# print cross_validate(train_data_alex)
	# print cross_validate(train_data_siftbow)
	test_label = svm_predict(train_data_alex, train_label, test_data_alex)
	generate_csv('svm_result.csv',test_label)


if __name__ == '__main__':
	main()