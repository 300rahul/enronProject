#!/usr/bin/python

import pickle
import sys
import matplotlib.pyplot
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit


### read in data dictionary, convert to numpy array
data_dict = pickle.load( open("../final_project/final_project_dataset.pkl", "r") )
features = ["salary", "bonus"]
data = featureFormat(data_dict, features)


### your code below
salary = []
bonus = []
for i in data:
    salary.append(i[0])
    bonus.append(i[1])
matplotlib.pyplot.scatter(salary,bonus,color = 'r')
matplotlib.pyplot.show()
max_salary = max(salary)
#name of the person correspond to the outlier
for i in data_dict:
    if data_dict[i]['salary'] == max_salary:
        print i
     
#the list of the total features
for i in data_dict:
    print data_dict[i],'\n'
