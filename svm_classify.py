import numpy as np
import read_data as rd
from sklearn.cross_validation import KFold
from sklearn.svm import SVC

train_label,picname_3k,labelname_3k = rd.read_train_3k(rd.train_3k)
test_name = rd.read_test_3k(rd.test_3k)
attr_name = rd.read_attributes_list(rd.attr_list)
train_data = rd.read_attributes(rd.attr_train)
test_data = rd.read_attributes(rd.attr_test)

def cross_validate():
	kf = KFold(len(train_data), n_folds=5)
	for train_index, test_index in kf:
		# print("TRAIN:", train_index, "TEST:", test_index)
		X_train, X_test = train_data[train_index], train_data[test_index]
	   	y_train, y_test = train_label[train_index], train_label[test_index]
	   	svm(X_train,y_train,X_test,y_test)

def svm(X_train,y_train,X_test,y_test):
	clf = SVC(C=1.0,kernel = 'rbf',decision_function_shape = 'ovr')
	clf.fit(X_train,y_train)

	return


def main():
	cross_validate()

if __name__ == '__main__':
	main()