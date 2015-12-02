from sklearn import naive_bayes
from sklearn import cross_validation
import read_data
import numpy as np
import csv

def gaussianNB(train_data, train_label, test_data):
        gnb = naive_bayes.GaussianNB()
        prediction = gnb.fit(train_data, train_label).predict(test_data)
        return prediction

def multinomialNB(train_data, train_label, test_data):
        mnb = naive_bayes.MultinomialNB()
        prediction = mnb.fit(train_data, train_label).predict(test_data)
        return prediction

def bernoulliNB(train_data, train_label, test_data):
        bnb = naive_bayes.BernoulliNB(binarize = 2.5)
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
        bnb = naive_bayes.BernoulliNB(binarize = 2.5)
        accuracy = bnb.fit(X_train, y_train).score(X_test, y_test)
        return accuracy

if __name__ == "__main__":
	piclabel_3k, picname_3k, labels_3k = read_data.read_train_3k(read_data.train_3k)
	alexnet_train = read_data.read_npy(read_data.alexnet_train)
	alexnet_test = read_data.read_npy(read_data.alexnet_test)
	prediction = bernoulliNB(alexnet_train, piclabel_3k, alexnet_test)
	print prediction
	test_imgname = read_data.read_test_3k(read_data.test_3k)
	csvfile = file('bernoulliNB_alexnet.csv', 'wb')
	writer = csv.writer(csvfile)
	writer.writerow(["ID", "Category"])
	for i in range(0, len(prediction)):
		writer.writerow([test_imgname[i],labels_3k[prediction[i]]])
	csvfile.close()
