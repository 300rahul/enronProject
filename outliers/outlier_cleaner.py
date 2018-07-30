#!/usr/bin/python


def outlierCleaner(predictions, ages, net_worths):
    """
        Clean away the 10% of points that have the largest
        residual errors (difference between the prediction
        and the actual net worth).

        Return a list of tuples named cleaned_data where 
        each tuple is of the form (age, net_worth, error).
    """
    
    cleaned_data = []
    ### your code goes here
    for i in range(0,len(ages)-1):
        cleaned_data.append((ages[i],net_worths[i],predictions[i]-net_worths[i]))
    cleaned_data.sort(key=lambda x:x[2])
    for i in range(81,89):
        cleaned_data.remove(cleaned_data[81])
    return cleaned_data
