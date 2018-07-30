#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
import numpy
sys.path.append("../tools/")
from email_preprocess import preprocess
from mlxtend.plotting import plot_decision_regions
import matplotlib.pyplot as plt



### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()

#########################################################
### your code goes here ###
from sklearn import svm
clf = svm.SVC(C = 10000.0)

features_train = features_train[:len(features_train)/100] 
labels_train = labels_train[:len(labels_train)/100] 

t0 = time()
clf.fit(features_train,labels_train)
print "training time:", round(time()-t0, 3), "s"

t0 = time()
pred = clf.predict(features_test)
print "predecting time:", round(time()-t0, 3), "s"

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(labels_test,pred)
print(accuracy)

answer10 = pred[10]
answer26 = pred[26]
answer50 = pred[50]
print(answer10,answer26,answer50)
count = 0
for i in pred:
    if i==1:
        count = count + 1
print count
plot_decision_regions(features_train,numpy.array(labels_train),clf = 'clf',legend = 2)

#########################################################


