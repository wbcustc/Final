from sklearn import naive_bayes
from sklearn import cross_validation
import read_data
import numpy as np
import svm_classify
from sklearn.svm import SVC
from sklearn.svm import LinearSVC

def gaussianNB(train_data, train_label, test_data):
        gnb = naive_bayes.GaussianNB()
        prediction = gnb.fit(train_data, train_label).predict(test_data)
        return prediction

def multinomialNB(train_data, train_label, test_data):
        mnb = naive_bayes.MultinomialNB()
        prediction = mnb.fit(train_data, train_label).predict(test_data)
        return prediction

def bernoulliNB(train_data, train_label, test_data):
        bnb = naive_bayes.BernoulliNB()
        prediction = bnb.fit(train_data, train_label).predict(test_data)
        return prediction

def crossvalid_svm(data, label):
	X_train, X_test, y_train, y_test = cross_validation.train_test_split(data, label, test_size = 0.2, random_state = 0)
	clf = LinearSVC(dual=False)
	score = clf.fit(X_train,y_train).predict(X_test)
	return score, y_test

def crossvalid_gaussianNB(data, label):
	X_train, X_test, y_train, y_test = cross_validation.train_test_split(data, label, test_size = 0.2, random_state = 0)
	gnb = naive_bayes.GaussianNB()
        #accuracy = gnb.fit(X_train, y_train).score(X_test, y_test)
        accuracy = gnb.fit(X_train, y_train).predict(X_test)
	return accuracy, y_test

def crossvalid_multinomialNB(data, label):
	X_train, X_test, y_train, y_test = cross_validation.train_test_split(data, label, test_size = 0.2, random_state = 0)
        mnb = naive_bayes.MultinomialNB()
        #accuracy = mnb.fit(X_train, y_train).score(X_test, y_test)
        accuracy = mnb.fit(X_train, y_train).predict(X_test)
	return accuracy, y_test

def crossvalid_bernoulliNB(data, label):
	X_train, X_test, y_train, y_test = cross_validation.train_test_split(data, label, test_size = 0.2, random_state = 0)
        bnb = naive_bayes.BernoulliNB(alpha = 0.2)
        #accuracy = bnb.fit(X_train, y_train).score(X_test, y_test)
        accuracy = bnb.fit(X_train, y_train).predict(X_test)
	return accuracy, y_test

def crossvalid(data, label):
	X_train, X_test, y_train, y_test = cross_validation.train_test_split(data, label, test_size = 0.2, random_state = 0)
	return X_train, X_test, y_train, y_test

def mode(data):
	result = np.zeros((600, 200))
	for i in range(0, len(data)):
		result[i][data[i]] = 1
	return result

if __name__ == "__main__":
	piclabel_3k, picname_3k, labels_3k = read_data.read_train_3k(read_data.train_3k)
	attributes_train = read_data.read_attributes(read_data.attr_train)
	alexnet_train = read_data.read_npy(read_data.alexnet_train)
	alexmin = np.amin(alexnet_train)
	alexnet_train_ = alexnet_train - alexmin;
	#X_train, X_test, y_train, y_test = crossvalid(data, label)
	#siftbow_train = read_data.read_npy(read_data.siftbow_train)
	acc_attri_gNB, ytest = crossvalid_gaussianNB(attributes_train, piclabel_3k)
	#print acc_attri_gNB
	acc_attri_mNB, ytest  = crossvalid_multinomialNB(attributes_train, piclabel_3k)
	#print acc_attri_mNB
	acc_attri_bNB, ytest = crossvalid_bernoulliNB(attributes_train, piclabel_3k)
	#print acc_attri_bNB
	acc_alexnet_gNB, ytest = crossvalid_gaussianNB(alexnet_train, piclabel_3k)
        #print acc_alexnet_gNB
        acc_alexnet_mNB, ytest = crossvalid_multinomialNB(alexnet_train_, piclabel_3k)
        #print acc_alexnet_mNB
        acc_alexnet_bNB, ytest = crossvalid_bernoulliNB(alexnet_train, piclabel_3k)
        #acc_attri_svm, ytest = crossvalid_svm(attributes_train, piclabel_3k)
	#acc_alexnet_svm, ytest = crossvalid_svm(alexnet_train, piclabel_3k)
	#print acc_alexnet_bNB
	#piclabel_3k = ytest
	acc = mode(acc_attri_gNB) + mode(acc_attri_mNB) + mode(acc_attri_bNB) + mode(acc_alexnet_gNB) + mode(acc_alexnet_mNB) + mode(acc_alexnet_bNB)# + mode(acc_alexnet_svm)# + mode(acc_alexnet_svm)
	prediction = np.argmax(acc, axis = 1)
	n = 0	
	for i in range(0, len(ytest)):
		if prediction[i] == ytest[i]:
			n = n + 1
	print n/600.00
	'''
	acc = acc_attri_gNB + acc_attri_mNB + acc_attri_bNB + acc_alexnet_gNB + acc_alexnet_mNB + acc_alexnet_bNB
	num = np.zeros(7)
	for i in range(0, len(piclabel_3k)):
		n = 0
		if acc_attri_gNB[i] == piclabel_3k[i]:
			n = n + 1
		if acc_attri_mNB[i] == piclabel_3k[i]:
                        n = n + 1
		if acc_attri_bNB[i] == piclabel_3k[i]:
                        n = n + 1
		if acc_alexnet_gNB[i] == piclabel_3k[i]:
                        n = n + 1
		if acc_alexnet_mNB[i] == piclabel_3k[i]:
                        n = n + 1
		if acc_alexnet_bNB[i] == piclabel_3k[i]:
                        n = n + 1 
		num[n] = num[n] + 1
	print num
	#acc_siftbow_gNB = crossvalid_gaussianNB(siftbow_train, piclabel_3k)
        #print acc_siftbow_gNB
        #acc_siftbow_mNB = crossvalid_multinomialNB(siftbow_train, piclabel_3k)
        #print acc_siftbow_mNB
        #acc_siftbow_bNB = crossvalid_bernoulliNB(siftbow_train, piclabel_3k)
        #print acc_siftbow_bNB
	'''
