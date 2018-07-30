#!/usr/bin/python

import sys
import pickle
import numpy
from time import time
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi','salary','from_this_person_to_poi','from_poi_to_this_person'] # You will need to use more features
from sklearn.feature_extraction.text import TfidfVectorizer
tf = TfidfVectorizer()                         
#print tf.get_feature_names()
### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Task 2: Remove outliers
def outlierCleaner(predictions, features, labels):
    """
        Clean away the 10% of points that have the largest
        residual errors (difference between the prediction
        and the actual net worth).

        Return a list of tuples named cleaned_data where 
        each tuple is of the form (age, net_worth, error).
    """
    
    cleaned_data = []
    for k in range(0,len(features)-1):
        cleaned_data.append((features[k],labels[k],predictions[k]-labels[k]))
    cleaned_data.sort(key=lambda x:x[2])
    x = len(cleaned_data)
    y = x - x/10
    del cleaned_data[int(y-1):x]
    return cleaned_data
### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
from sklearn.grid_search import GridSearchCV

from sklearn.svm import SVC


### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!

from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.3, random_state=42)

print "Fitting the classifier to the training set"
t0 = time()
param_grid = {'C': [1,10],'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1]}
# for sklearn version 0.16 or prior, the class_weight parameter value is 'auto'
clf = GridSearchCV(SVC(kernel='rbf', class_weight='balanced'), param_grid)
clf = clf.fit(features_train,labels_train)
print "done in %0.3fs" % (time() - t0)
print "Best estimator found by grid search:"
print clf.best_estimator_
pred = clf.predict(features_test)
pred1 = clf.predict(features_train)

#check accuracy
from sklearn.metrics import accuracy_score,precision_score,recall_score
score = accuracy_score(labels_test,pred)

p_score = precision_score(labels_test,pred,average = 'macro')

r_score = recall_score(labels_test,pred,average = 'macro')

print 'score =',score
print 'p_score =',p_score
print 'r_score =',r_score

cleaned_data = outlierCleaner( pred1, features_train, labels_train )
### refit your cleaned data!
if len(cleaned_data) > 0:
    features, labels, errors = zip(*cleaned_data) 
    features  = numpy.reshape( numpy.array(features), (len(features), 3))
    labels = numpy.reshape( numpy.array(labels), (len(labels), 1))
    from sklearn.cross_validation import train_test_split
    features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.3, random_state=42)
    
#refitting
from sklearn.grid_search import GridSearchCV

from sklearn.svm import SVC
print "Fitting the classifier to the cleaned training set"
clf = SVC(C=1, cache_size=200, class_weight='balanced', coef0=0.0,
  decision_function_shape='ovr', degree=3, gamma=0.0005, kernel='rbf',
  max_iter=-1, probability=False, random_state=None, shrinking=True,
  tol=0.001, verbose=False)
clf.fit(features_train,labels_train)
pred = clf.predict(features_test)


#check accuracy
from sklearn.metrics import accuracy_score,precision_score,recall_score
score = accuracy_score(labels_test,pred)

p_score = precision_score(labels_test,pred,average = 'macro')

r_score = recall_score(labels_test,pred,average = 'macro')

print 'new score =',score
print 'new p_score =',p_score
print 'new r_score =',r_score 
### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)

