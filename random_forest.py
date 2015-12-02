# -*- coding: utf-8 -*-
from sklearn.ensemble import RandomForestClassifier
from sklearn import cross_validation
import read_data
import numpy as np


def random_forest(depth,num_trees,train_data,train_label,test_data):
    rf=RandomForestClassifier(n_estimators=50)
    prediction=rf.fit(train_data, train_label).predict(test_data)

    return prediction


def crossvalidate_rf(data,label):
    X_train,X_test, y_train, y_test=cross_validation.train_test_split(data,label,test_size=0.2,random_state=0)
    rf=RandomForestClassifier(n_estimators=10)
    accuracy=rf.fit(X_train, y_train).score(X_test,y_test)
    return accuracy


if __name__=="__main__":
    piclabel_3k,picname_3k,labels_3k=read_data.read_train_3k(read_data.train_3k)
    attributes_train=read_data.read_attributes(read_data.attr_train)
    alexnet_train=read_data.read_npy(read_data.alexnet_train)
    siftbow_train=read_data.read_npy(read_data.siftbow_train)

    att_rf=crossvalidate_rf(attributes_train,piclabel_3k)
    alex_rf=crossvalidate_rf(alexnet_train,piclabel_3k)
    sift_rf=crossvalidate_rf(siftbow_train,piclabel_3k)

    print att_rf,alex_rf,sift_rf
