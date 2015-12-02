from sklearn import naive_bayes
from sklearn import cross_validation
import read_data
import numpy as np

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

def crossvalid_gaussianNB(data, label):
	X_train, X_test, y_train, y_test = cross_validation.train_test_split(data, label, test_size = 0.2, random_state = 0)
	gnb = naive_bayes.GaussianNB()
        accuracy = gnb.fit(X_train, y_train).score(X_test, y_test)
        return accuracy

def crossvalid_multinomialNB(data, label):
	X_train, X_test, y_train, y_test = cross_validation.train_test_split(data, label, test_size = 0.2, random_state = 0)
        mnb = naive_bayes.MultinomialNB()
        accuracy = mnb.fit(X_train, y_train).score(X_test, y_test)
        return accuracy

def crossvalid_bernoulliNB(data, label):
	X_train, X_test, y_train, y_test = cross_validation.train_test_split(data, label, test_size = 0.2, random_state = 0)
        bnb = naive_bayes.BernoulliNB()
        accuracy = bnb.fit(X_train, y_train).score(X_test, y_test)
        return accuracy

if __name__ == "__main__":
	piclabel_3k, picname_3k, labels_3k = read_data.read_train_3k(read_data.train_3k)
	attributes_train = read_data.read_attributes(read_data.attr_train)
	alexnet_train = read_data.read_npy(read_data.alexnet_train)
	alexmin = np.amin(alexnet_train)
	alexnet_train_ = alexnet_train - alexmin;
	siftbow_train = read_data.read_npy(read_data.siftbow_train)
	acc_attri_gNB = crossvalid_gaussianNB(attributes_train, piclabel_3k)
	print acc_attri_gNB
	acc_attri_mNB = crossvalid_multinomialNB(attributes_train, piclabel_3k)
	print acc_attri_mNB
	acc_attri_bNB = crossvalid_bernoulliNB(attributes_train, piclabel_3k)
        print acc_attri_bNB
	acc_alexnet_gNB = crossvalid_gaussianNB(alexnet_train, piclabel_3k)
        print acc_alexnet_gNB
        acc_alexnet_mNB = crossvalid_multinomialNB(alexnet_train_, piclabel_3k)
        print acc_alexnet_mNB
        acc_alexnet_bNB = crossvalid_bernoulliNB(alexnet_train, piclabel_3k)
        print acc_alexnet_bNB
	acc_siftbow_gNB = crossvalid_gaussianNB(siftbow_train, piclabel_3k)
        print acc_siftbow_gNB
        acc_siftbow_mNB = crossvalid_multinomialNB(siftbow_train, piclabel_3k)
        print acc_siftbow_mNB
        acc_siftbow_bNB = crossvalid_bernoulliNB(siftbow_train, piclabel_3k)
        print acc_siftbow_bNB
