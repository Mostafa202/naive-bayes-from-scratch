import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

dataset=pd.read_csv('naive.csv')

x=dataset.iloc[:,2:-1].values
y=dataset.iloc[:,-1].values

from sklearn.preprocessing import *
lb=LabelEncoder()

y=lb.fit_transform(y)

from sklearn.model_selection import *

train_x,test_x,train_y,test_y=train_test_split(x,y,test_size=.5,random_state=0)


def group_under_class(x,y):
    dict={}
    for i in range(len(x)):
        if y[i] not in dict:
            dict[y[i]]=[]
        dict[y[i]].append(x[i])
    return dict


def mean(numbers):
    return sum(numbers)/len(numbers)

def std(numbers):
    m=mean(numbers)
    variance=sum([(x-m)**2 for x in numbers])/(len(numbers)-1)
    return np.sqrt(variance)

def mean_and_std(mydata):
    res=[(mean(attribute),std(attribute)) for attribute in zip(*mydata)]
    return res

def mean_and_std_forclass(x,y):
    info={}
    res=group_under_class(x,y)
    for classvalue,instances in res.items():
        info[classvalue]=mean_and_std(instances)
    return info

def gaussian_prob(x,mean,std):
    expo = math.exp(-(math.pow(x - mean, 2) / (2 * math.pow(std, 2)))) 
    return (1 / (math.sqrt(2 * math.pi) * std)) * expo 

def calculate_prob_class(info,test):
    probabilities={}
    for classvalue,summary_instances in info.items():
        probabilities[classvalue]=1
        for val in range(len(summary_instances)):
            mean,std=summary_instances[val]
            ch=test[val]
            probabilities[classvalue]*=gaussian_prob(ch,mean,std)
    return probabilities

test=mean_and_std_forclass(train_x,train_y)
c=calculate_prob_class(test,test_x[0])


def predict(info,test):
    best_label,best_prob=None,-1
    probs=calculate_prob_class(info,test)
    for classval,prob in probs.items():
        if classval is None or prob>best_prob:
            best_prob=prob
            best_label=classval
    return best_label

def get_predictions(test,x,y):
    predictions=[]
    info=mean_and_std_forclass(x,y)
    for i in range(len(test)):
        predictions.append(predict(info,test[i]))
    return predictions

def calc_accuracy(test_y,predictions):
    return (sum(np.array(predictions)==test_y)/len(test_y))*100
    
preds=get_predictions(test_x,train_x,train_y)

acc=calc_accuracy(test_y,preds)
print(acc)
            
    
    
    

