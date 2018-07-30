#!/usr/bin/python

""" 
    Starter code for exploring the Enron dataset (emails + finances);
    loads up the dataset (pickled dict of dicts).

    The dataset has the form:
    enron_data["LASTNAME FIRSTNAME MIDDLEINITIAL"] = { features_dict }

    {features_dict} is a dictionary of features associated with that person.
    You should explore features_dict as part of the mini-project,
    but here's an example to get you started:

    enron_data["SKILLING JEFFREY K"]["bonus"] = 5600000
    
"""

import pickle
enron_data = pickle.load(open("../final_project/final_project_dataset.pkl", "r"))
print "no. of persons in the data set = ",len(enron_data)

print 'no. of features for each person = ',len(enron_data[enron_data.keys()[0]]) 

count = 0
for person_name in enron_data:
    if enron_data[person_name]["poi"] == 1:
        count=count+1
print "no. of POIs in the dataset = ",count


count = 0
name_data = open('../final_project/poi_names.txt','r')

'''while name_data.readlines():
    count = count+1'''
    
for line in name_data:
    count = count+1
name_data.close()   
 
print "number of totel POIs = ",count
'''for key in enron_data.keys(): #this print the list or persons in the dataset
    print key'''
    
print "toal stock value of james prentice = ",enron_data["PRENTICE JAMES"]["total_stock_value"]
#print enron_data



print "number of messages from Wesley Colwell  to POI = ",enron_data["COLWELL WESLEY"]["from_this_person_to_poi"]

print "value of stock options exercised by Jeffrey K Skilling = ",enron_data["SKILLING JEFFREY K"]["exercised_stock_options"]

count1 = 0
count2 = 0
for name in enron_data.keys():

    if enron_data[name]["salary"] != 'NaN':
        count1 = count1+1
    if enron_data[name]["email_address"] != 'NaN':
        count2 = count2+1
print count1,count2 

#lesson 6 quize 29
count3 = 0
for name in enron_data.keys():

    if enron_data[name]["total_payments"] == 'NaN':
        count3 = count3+1
percentage = count3*100/len(enron_data)
print count3,percentage


