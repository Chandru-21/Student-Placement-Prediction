# -*- coding: utf-8 -*-
"""
Created on Sat May 16 17:05:06 2020

@author: Chandramouli
"""

import pandas as pd

dataset= pd.read_csv('C://Users//Chandramouli//.spyder-py3//ML DATASETS//Placement_Data_Full_Class.csv')
dataset.info()
x=dataset.loc[:,['gender','ssc_p','hsc_p','degree_p','degree_t','workex','etest_p','specialisation','mba_p']]
y=dataset.loc[:,['status']]
y=pd.get_dummies(y)
y.info()
y=y.drop(['status_Not Placed'],1) #1-placed 0-not placed
x.info()
x1=x.loc[:,['gender','degree_t','workex','specialisation']]
x2=pd.get_dummies(x1)
x2.info()
x2=x2.drop(['gender_M','degree_t_Others','workex_No','specialisation_Mkt&HR'],1)
x=x.drop(['gender','degree_t','workex','specialisation'],1)
x2=pd.concat([x,x2],axis=1)#without scaling
#with scaling
x3=x2.loc[:,['ssc_p','hsc_p','degree_p','etest_p','mba_p']]
from sklearn.preprocessing import StandardScaler
scale=StandardScaler()
x3=scale.fit_transform(x3)
x3=pd.DataFrame(x3)
x3.columns=['ssc_p','hsc_p','degree_p','etest_p','mba_p']
x4=x2.drop(['ssc_p','hsc_p','degree_p','etest_p','mba_p'],1)
x5=pd.concat([x4,x3],axis=1)#with scaling

#Who is getting more placements girls or boys?
dataset['salary'].fillna(0,inplace = True)
import seaborn as sns
import matplotlib.pyplot as plt
def plot(data,x,y):
    plt.Figure(figsize =(10,10))
    sns.boxplot(x = dataset[x],y= dataset[y])
    g = sns.FacetGrid(dataset,row = y)
    g = g.map(plt.hist,x)
  
plot(dataset,"salary","gender")

sns.countplot(dataset['status'],hue=dataset['gender'])#hue for color encoding
#Students with higher pg percentage had higher probability of getting placed
plot(dataset,"mba_p","status")


#cbse vs sb who gets highest salary package?
plot(dataset,"salary","hsc_b")
#The Range of salary is high for central board students with the median of 2.5 Lakhs per annum
#The Median salary for other board students is 2.4 Lakhs per annum

#The Range of salary is high for boys compared to their counterpart
#logistic Regression using cross validation
from sklearn.model_selection import StratifiedKFold
skf=StratifiedKFold(n_splits=5)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
accuracy_logistic=[]
count=0
for train_index,test_index in skf.split(x5,y):
    count=count+1
    if(count<4):
    

        x_train,x_test =x5.iloc[train_index],x5.iloc[test_index]
        y_train,y_test=y.iloc[train_index],y.iloc[test_index]
        classifier=LogisticRegression()
        classifier.fit(x_train,y_train)
        y_pred=classifier.predict(x_test)
        print(accuracy_score(y_test,y_pred))
        accuracy_logistic.append(accuracy_score(y_test,y_pred))
from sklearn.metrics import confusion_matrix
cm_logistic=confusion_matrix(y_test,y_pred)
accuracy_score(y_test,y_pred)*100




#normal method using logistic regression

from sklearn.model_selection import train_test_split

x_train1,x_test1,y_train1,y_test1=train_test_split(x5,y,test_size=0.25,random_state=0)

from sklearn.linear_model import LogisticRegression
classifier=LogisticRegression()
classifier.fit(x_train1,y_train1)

y_pred1=classifier.predict(x_test1)
from sklearn.metrics import confusion_matrix
cm_logistic=confusion_matrix(y_test1,y_pred1)
from sklearn.metrics import accuracy_score
accuracy_score(y_test1,y_pred1)*100


#k-nearest using cross validation

from sklearn.model_selection import StratifiedKFold
skf=StratifiedKFold(n_splits=10)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
accuracy_Knearest=[]
count=0
for train_index,test_index in skf.split(x5,y):
    count=count+1
    if(count<9):
    
        x_train,x_test =x5.iloc[train_index],x5.iloc[test_index]
        y_train,y_test=y.iloc[train_index],y.iloc[test_index]
        classifier=KNeighborsClassifier(n_neighbors=5,metric='minkowski',p=2)
        classifier.fit(x_train,y_train)
        y_pred=classifier.predict(x_test)
        print(accuracy_score(y_test,y_pred))
        accuracy_Knearest.append(accuracy_score(y_test,y_pred))
from sklearn.metrics import confusion_matrix
cm_Knearest=confusion_matrix(y_test,y_pred)
from sklearn.metrics import accuracy_score
accuracy_score(y_test,y_pred)*100

#normal method using K nearest algorithm

from sklearn.model_selection import train_test_split

x_train1,x_test1,y_train1,y_test1=train_test_split(x5,y,test_size=0.25,random_state=0)

from sklearn.neighbors import KNeighborsClassifier
classifier=KNeighborsClassifier(n_neighbors=5,metric='minkowski',p=2)
classifier.fit(x_train1,y_train1)

y_pred1=classifier.predict(x_test1)
from sklearn.metrics import confusion_matrix
cm_logistic=confusion_matrix(y_test1,y_pred1)
from sklearn.metrics import accuracy_score
accuracy_score(y_test1,y_pred1)*100
##Naive Bayes using cross validation
#feature scaling is not needed
from sklearn.model_selection import StratifiedKFold
skf=StratifiedKFold(n_splits=6)
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
accuracy_NB=[]
count=0
for train_index,test_index in skf.split(x2,y):
    count=count+1
    if(count<2):
    
    
        x_train,x_test =x2.iloc[train_index],x2.iloc[test_index]
        y_train,y_test=y.iloc[train_index],y.iloc[test_index]
        classifier=GaussianNB()
        classifier.fit(x_train,y_train)
        y_pred=classifier.predict(x_test)
        print(accuracy_score(y_test,y_pred))
        accuracy_NB.append(accuracy_score(y_test,y_pred))
from sklearn.metrics import confusion_matrix
cm_NB=confusion_matrix(y_test,y_pred)
from sklearn.metrics import accuracy_score
accuracy_score(y_test,y_pred)*100

#normal method using Naive baye's

from sklearn.model_selection import train_test_split

x_train1,x_test1,y_train1,y_test1=train_test_split(x2,y,test_size=0.25,random_state=0)
from sklearn.naive_bayes import GaussianNB
classifier=GaussianNB()
classifier.fit(x_train1,y_train1)
y_pred1=classifier.predict(x_test1)
from sklearn.metrics import confusion_matrix
cm_NB=confusion_matrix(y_test1,y_pred1)
from sklearn.metrics import accuracy_score
accuracy_score(y_test1,y_pred1)*100

###Decision tree
from sklearn.model_selection import StratifiedKFold
skf=StratifiedKFold(n_splits=10)
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
accuracy_DT=[]
count=0
for train_index,test_index in skf.split(x5,y):
    count=count+1
    if(count<3):
    
     
        x_train,x_test =x5.iloc[train_index],x5.iloc[test_index]
        y_train,y_test=y.iloc[train_index],y.iloc[test_index]
        classifier=DecisionTreeClassifier(criterion='entropy')
        classifier.fit(x_train,y_train)
        y_pred=classifier.predict(x_test)
        print(accuracy_score(y_test,y_pred))
        accuracy_DT.append(accuracy_score(y_test,y_pred))
from sklearn.metrics import confusion_matrix
cm_DT=confusion_matrix(y_test,y_pred)
from sklearn.metrics import accuracy_score
accuracy_score(y_test,y_pred)*100

#normal method using DT
from sklearn.model_selection import train_test_split

x_train1,x_test1,y_train1,y_test1=train_test_split(x5,y,test_size=0.25,random_state=0)

from sklearn.tree import DecisionTreeClassifier
classifier=DecisionTreeClassifier(criterion='entropy')
classifier.fit(x_train1,y_train1)

y_pred1=classifier.predict(x_test1)
from sklearn.metrics import confusion_matrix
cm_DT=confusion_matrix(y_test1,y_pred1)
from sklearn.metrics import accuracy_score
accuracy_score(y_test1,y_pred1)*100

#random forest
from sklearn.model_selection import StratifiedKFold
skf=StratifiedKFold(n_splits=15)
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
accuracy_RF=[]
count=0
for train_index,test_index in skf.split(x5,y):
    count=count+1
    if(count<4):
     
        x_train,x_test =x5.iloc[train_index],x5.iloc[test_index]
        y_train,y_test=y.iloc[train_index],y.iloc[test_index]
        classifier=RandomForestClassifier(n_estimators=10,criterion='entropy')
        classifier.fit(x_train,y_train)
        y_pred=classifier.predict(x_test)
        print(accuracy_score(y_test,y_pred))
        accuracy_RF.append(accuracy_score(y_test,y_pred))
from sklearn.metrics import confusion_matrix
cm_RF=confusion_matrix(y_test,y_pred)
from sklearn.metrics import accuracy_score
accuracy_score(y_test,y_pred)*100

#Random Forest with Hyperparameter Tuning
from sklearn.model_selection import GridSearchCV
parameter=[{'n_estimators':[10,20,30,40],'criterion':['gini','entropy'],'bootstrap':[True,False],'max_depth':[2,3,4,5,6]}]
grid_search=GridSearchCV(classifier,param_grid=parameter,scoring='accuracy',cv=skf)
grid_search=grid_search.fit(x_train,y_train)
grid_search.best_params_
grid_search.best_score_
grid_search.cv_results_
grid_search.best_index_

#hyper parameter result 
classifier=RandomForestClassifier(n_estimators=30,criterion='entropy',max_depth=5,bootstrap=True)

#using it in Random Forest algorithm

from sklearn.model_selection import StratifiedKFold
skf=StratifiedKFold(n_splits=15)
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
accuracy_RF1=[]
count=0
for train_index,test_index in skf.split(x5,y):
    count=count+1
    if(count<2):
    
    
  
     
        x_train,x_test =x5.iloc[train_index],x5.iloc[test_index]
        y_train,y_test=y.iloc[train_index],y.iloc[test_index]
        classifier=RandomForestClassifier(n_estimators=30,criterion='entropy',max_depth=5,bootstrap=True)
        classifier.fit(x_train,y_train)
        y_pred=classifier.predict(x_test)
        print(accuracy_score(y_test,y_pred))
        accuracy_RF1.append(accuracy_score(y_test,y_pred))
from sklearn.metrics import confusion_matrix
cm_RF1=confusion_matrix(y_test,y_pred)
from sklearn.metrics import accuracy_score
accuracy_score(y_test,y_pred)*100
y_pred=pd.DataFrame(y_pred)
