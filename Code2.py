# -*- coding: utf-8 -*-
"""Copy of BDMH_PROJECT_FINAL.ipynb

**HEART DISEASE PREDICTION**
"""

import pandas as pd
from sklearn.tree import DecisionTreeClassifier as cl
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow import keras
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

from google.colab import drive
drive.mount('/content/drive')

accuracy_dict = {}
precision_dict = {}
recall_dict = {}
fscore_dict = {}

data1=pd.read_csv("/content/drive/My Drive/heart.csv")
data2=pd.read_csv("/content/drive/My Drive/heart.csv")
target = data1['target']
data1 = data1.drop(columns=['target','sex','age'],axis=1)
data2 = data2.drop(columns=['target'],axis=1)
print(data1)
print(data2)

"""**MULTI-LAYER PERCEPTRON MODEL**"""

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(512, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(256, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(512, activation=tf.nn.softmax))

model.compile(optimizer = tf.optimizers.Adam(),
              loss = 'sparse_categorical_crossentropy',
              metrics=['accuracy'])
target = np.array(target)
data1 = np.array(data1)
data2 = np.array(data2)
train_x,test_x,train_y,test_y = train_test_split(data1,target,test_size=0.2)
x_train,x_test,y_train,y_test = train_test_split(data2,target,test_size=0.2)
model.fit(train_x,train_y,epochs=50)
pred1 = model.predict_classes(test_x)

model1 = tf.keras.models.Sequential()
model1.add(tf.keras.layers.Flatten())
model1.add(tf.keras.layers.Dense(512, activation=tf.nn.relu))
model1.add(tf.keras.layers.Dense(256, activation=tf.nn.relu))
model1.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model1.add(tf.keras.layers.Dense(512, activation=tf.nn.softmax))

model1.compile(optimizer = tf.optimizers.Adam(),
              loss = 'sparse_categorical_crossentropy',
              metrics=['accuracy'])
model1.fit(x_train,y_train,epochs=50)
pred2 = model1.predict_classes(x_test)

print("*************Neural Network*************************")
print("******************************************************")
print("Accuracy without using two features namely \"sex\" , \"age\" : ")

print(accuracy_score(test_y,pred1))
print("*******************************************************")
print("Accuracy with using two features namely \"sex\" , \"age\" : ")
print(accuracy_score(y_test,pred2))

accuracy_dict[11] = []
fscore_dict[11] = []
accuracy_dict[13] = []
fscore_dict[13] = []

accuracy_dict[11].append(accuracy_score(test_y, pred1))
fscore_dict[11].append(f1_score(test_y, pred1, average='macro'))
print("*******************************************************")
print("F1 score without using two features namely \"sex\" , \"age\" : ")
print("F1 score",f1_score(test_y, pred1, average='macro'))

accuracy_dict[13].append(accuracy_score(y_test, pred2))
fscore_dict[13].append(f1_score(y_test, pred2, average='macro'))
print("*******************************************************")
print("F1 score with using two features namely \"sex\" , \"age\" : ")
print("F1 score",f1_score(y_test, pred2, average='macro'))

"""**SVM MODEL**"""

def notmalise__in_0to_1Data(fullcol):
# -----------------------------------------------------------------------------------------
    min_is=min(fullcol)
    max_is=max(fullcol)
    fullcol1=fullcol.apply(lambda x:((x-min_is)/(max_is-min_is)))
    
#     ---------------------------------------------
    
    return fullcol1
from sklearn.svm import SVC 
from sklearn.model_selection import GridSearchCV 
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score,train_test_split
from sklearn import preprocessing
from sklearn import metrics
import pandas as pd
import numpy as np
df=pd.read_csv("/content/drive/My Drive/heart.csv", header = 0)
df1=pd.read_csv("/content/drive/My Drive/heart.csv", header = 0)
X= df.drop(columns=['target','age','sex'],axis=0)
X1=df1.drop(columns=['target'],axis=0)
y=df['target']
y1=df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
X_train_normalise=X_train.apply(lambda col: notmalise__in_0to_1Data(col), axis = 0)
X_test_normalise=X_test.apply(lambda col: notmalise__in_0to_1Data(col), axis = 0)
svm1 =SVC(C=1000,gamma=0.01,kernel="rbf")
svm1.fit(X_train_normalise, y_train)
svm_score_is = cross_val_score(estimator=svm1, X=X_train, y=y_train, cv=5)
svm_predict= svm1.predict(X_test_normalise)
print("Accuracy of svm (Without age and sex) is ",svm_score_is.mean())
accuracy_dict[11].append(svm_score_is.mean())
fscore_dict[11].append(f1_score(y_test, svm_predict, average='macro'))
print("*******************************************************")
print("F1 score without using two features namely \"sex\" , \"age\" : ")
print("F1 score",f1_score(y_test, svm_predict, average='macro'))
X_train, X_test, y_train, y_test = train_test_split(X1, y1, test_size=0.2)
X_train_normalise=X_train.apply(lambda col: notmalise__in_0to_1Data(col), axis = 0)
X_test_normalise=X_test.apply(lambda col: notmalise__in_0to_1Data(col), axis = 0)
svm1 =SVC(C=1000,gamma=0.01,kernel="rbf")
svm1.fit(X_train_normalise, y_train)
svm_score_is = cross_val_score(estimator=svm1, X=X_train, y=y_train, cv=5)

svm_predict= svm1.predict(X_test_normalise)
print("Accuracy of svm (With age and sex)  is ",svm_score_is.mean())
accuracy_dict[13].append(svm_score_is.mean())
fscore_dict[13].append(f1_score(y_test, svm_predict, average='macro'))
print("*******************************************************")
print("F1 score with using two features namely \"sex\" , \"age\" : ")
print("F1 score",f1_score(y_test, svm_predict, average='macro'))

"""**LOGISTIC REGRESSION**"""

from sklearn.linear_model import LogisticRegression

data = pd.read_csv('/content/drive/My Drive/heart.csv')
data2 = pd.read_csv('/content/drive/My Drive/heart.csv')
d=data.isnull().sum(axis = 1)
x=data['target']
a = data['age']
s = data['sex']
data = data.drop(columns=['target','age','sex'],axis=0)
data2=data2.drop(columns=['target'],axis=0)
classifier = LogisticRegression()
x_train,x_test,y_train,y_test=train_test_split(data,x,test_size=0.2)
classifier.fit(x_train, y_train)
y=classifier.predict(x_test)
score_is = cross_val_score(estimator=classifier, X=x_train, y=y_train, cv=5)
print("----------------------------------- Logistic Regression MODEL -----------------------------------")
print("***********************Without age and sex column************************")
print("ACCURACY SCORE:-",score_is.mean())
accuracy_dict[11].append(score_is.mean())
fscore_dict[11].append(f1_score(y_test, y, average='macro'))
print("*******************************************************")
print("F1 score without using two features namely \"sex\" , \"age\" : ")
print("F1 score",f1_score(y_test, y, average='macro'))
print("\n")
x_train1,x_test1,y_train1,y_test1=train_test_split(data2,x,test_size=0.2)
classifier.fit(x_train1,y_train1)
y1=classifier.predict(x_test1)
score_is = cross_val_score(estimator=classifier, X=x_train1, y=y_train1, cv=5)
print("**********************With age and sex column****************************")
print("ACCURACY SCORE:-",score_is.mean())
accuracy_dict[13].append(score_is.mean())
fscore_dict[13].append(f1_score(y_test1, y1, average='macro'))
print("*******************************************************")
print("F1 score with using two features namely \"sex\" , \"age\" : ")
print("F1 score",f1_score(y_test1, y1, average='macro'))

"""**RANDOM FOREST**"""

from sklearn.ensemble import RandomForestClassifier
data = pd.read_csv('/content/drive/My Drive/heart.csv')
data2 = pd.read_csv('/content/drive/My Drive/heart.csv')
d=data.isnull().sum(axis = 1)
x=data['target']
a = data['age']
s = data['sex']
data = data.drop(columns=['target','age','sex'],axis=0)
data2=data2.drop(columns=['target'],axis=0)
classifier = RandomForestClassifier(max_depth=2, random_state=0)
x_train,x_test,y_train,y_test=train_test_split(data,x,test_size=0.2)
classifier.fit(x_train, y_train)
y=classifier.predict(x_test)
score_is = cross_val_score(estimator=classifier, X=x_train, y=y_train, cv=5)
print("----------------------------------- Random-Forest Model -----------------------------------")
print("***********************Without age and sex column************************")
print("ACCURACY SCORE:-",score_is.mean())
accuracy_dict[11].append(score_is.mean())
fscore_dict[11].append(f1_score(y_test, y, average='macro'))
print("F1 score",f1_score(y_test, y, average='macro'))
print("\n")

x_train1,x_test1,y_train1,y_test1=train_test_split(data2,x,test_size=0.2)
classifier.fit(x_train1,y_train1)
y1=classifier.predict(x_test1)
score_is = cross_val_score(estimator=classifier, X=x_train1, y=y_train1, cv=5)
print("**********************With age and sex column****************************")
print("ACCURACY SCORE:-",score_is.mean())
accuracy_dict[13].append(score_is.mean())
fscore_dict[13].append(f1_score(y_test1, y1, average='macro'))
print("F1 score",f1_score(y_test1, y1, average='macro'))

"""**NAIVE-BAYES MODEL**"""

from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
data = pd.read_csv("/content/drive/My Drive/heart.csv")
data2 = pd.read_csv("/content/drive/My Drive/heart.csv")
d=data.isnull().sum(axis = 1)
x=data['target']
a = data['age']
s = data['sex']
data = data.drop(columns=['target','age','sex'],axis=0)
data2=data2.drop(columns=['target'],axis=0)
x_train,x_test,y_train,y_test=train_test_split(data,x,test_size=0.2)
nb = GaussianNB()
nb.fit(x_train,y_train)
pred = nb.predict(x_test)
score_is = cross_val_score(estimator=nb, X=x_train, y=y_train, cv=5)
print("----------------------------------- Naive-Bayes MODEL -----------------------------------")
print("***********************Without age and sex column************************")
print("Accuracy",score_is.mean())
accuracy_dict[11].append(score_is.mean())
fscore_dict[11].append(f1_score(y_test, pred, average='macro'))
print("F1 score",f1_score(y_test, pred, average='macro'))
print("\n")
x_train,x_test,y_train,y_test=train_test_split(data2,x,test_size=0.2)
nb = GaussianNB()
nb.fit(x_train,y_train)
pred = nb.predict(x_test)
score_is = cross_val_score(estimator=nb, X=x_train, y=y_train, cv=5)
print("----------------------------------- Naive-Bayes MODEL -----------------------------------")
print("***********************With age and sex column************************")
print("Accuracy",score_is.mean())
accuracy_dict[13].append(score_is.mean())
fscore_dict[13].append(f1_score(y_test, pred, average='macro'))
print("F1 score",f1_score(y_test, pred, average='macro'))

def find_max(arr):
  ma= arr[0]
  i_m=0
  for i in range(len(arr)):
    if(arr[i]>ma):
      ma=arr[i]
      i_m=i
  return i_m

"""**KNN MODEL**"""

import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
data = pd.read_csv('/content/drive/My Drive/heart.csv')
data2 = pd.read_csv('/content/drive/My Drive/heart.csv')
d=data.isnull().sum(axis = 1)
x=data['target']
a = data['age']
s = data['sex']
neighbour=list(range(1,10))
data = data.drop(columns=['target','age','sex'],axis=0)
data2=data2.drop(columns=['target'],axis=0)
acc1=[]
acc2=[]
f1_1=[]
f1_2=[]
x_train,x_test,y_train,y_test=train_test_split(data,x,test_size=0.2)
for i in neighbour:
  classifier = KNeighborsClassifier(n_neighbors=i)

  classifier.fit(x_train, y_train)
  y=classifier.predict(x_test)
  acc1.append(accuracy_score(y_test,y))
  f1_1.append(f1_score(y_test, y, average='macro'))
i_max =find_max(acc1)
print("----------------------------------- KNN MODEL -----------------------------------")
print("***********************Without age and sex column************************")
print("ACCURACY SCORE:-",accuracy_score(y_test,y))
accuracy_dict[11].append(acc1[i_max])
fscore_dict[11].append(f1_1[i_max])
print("F1 score",f1_1[i_max])
print("\n")

x_train1,x_test1,y_train1,y_test1=train_test_split(data2,x,test_size=0.2)
for i in neighbour:
  classifier = KNeighborsClassifier(n_neighbors=i)

  classifier.fit(x_train1, y_train1)
  y1=classifier.predict(x_test1)
  acc2.append(accuracy_score(y_test1,y1))
  f1_2.append(f1_score(y_test1, y1, average='macro'))
i_max2 =find_max(acc2)

print("**********************With age and sex column****************************")
# print("ACCURACY SCORE:-",acc2[i_max1])
accuracy_dict[13].append(acc2[i_max2])
fscore_dict[13].append(f1_2[i_max2])
print("F1 score",f1_2[i_max2])
print("\n")
plt.plot(neighbour,acc1,label="With Age and Sex Column")
plt.plot(neighbour,acc2,label="Without Age and Sex Column")
plt.xlabel("Neighbours")
plt.ylabel("Accuracy Obtained")
plt.legend()
plt.show()

"""**DECISION-TREE CLASSIFIER**"""

import pandas as pd
from sklearn.tree import DecisionTreeClassifier as cl
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
data = pd.read_csv('/content/drive/My Drive/heart.csv')
data2 = pd.read_csv('/content/drive/My Drive/heart.csv')
d=data.isnull().sum(axis = 1)
x=data['target']
a = data['age']
s = data['sex']
data = data.drop(columns=['target','age','sex'],axis=0)

def count(x):
    count=0
    for i in x:
        if i==0:
            count=count+1
        else:
            continue
    return count
data2=data2.drop(columns=['target'],axis=0)
print("Target is 0(i.e. not having heart disease) for",count(x),"patients")
print("Target is 1(i.e. having heart disease) for" ,len(x)-count(x),"patients")
acc1=[]
acc2=[]
f1_1=[]
f1_2=[]
x_train,x_test,y_train,y_test=train_test_split(data,x,test_size=0.2)
for i in range(1,10):
  clf = cl(random_state=0,criterion='entropy',max_depth=i)
  clf.fit(x_train,y_train)
  y=clf.predict(x_test)
  acc1.append(accuracy_score(y_test,y))
  f1_1.append(f1_score(y_test, y, average='macro'))
i_max =find_max(acc1)

print("***********Decision-tree***************")
print("***********************Without age and sex column************************")
print("Accuracy",accuracy_score(y_test,y))
accuracy_dict[11].append(acc1[i_max])
fscore_dict[11].append(f1_1[i_max])
print("F1 score",f1_1[i_max])
print("\n")

x_train1,x_test1,y_train1,y_test1=train_test_split(data2,x,test_size=0.2)
for i in range(1,10):
  clf1 = cl(random_state=0,criterion='entropy',max_depth=i)
  clf1.fit(x_train1,y_train1)
  y1=clf1.predict(x_test1)
  acc2.append(accuracy_score(y_test1,y1))
  f1_2.append(f1_score(y_test1, y1, average='macro'))
i_max1 =find_max(acc2)

print("***********************With age and sex column************************")
print(accuracy_score(y_test1,y1))
accuracy_dict[13].append(acc2[i_max1])
fscore_dict[13].append(f1_2[i_max1])
print("F1 score",f1_2[i_max1])
print("\n")
max_depth_range=list(range(1,10))
plt.plot(max_depth_range,acc1,label="With Age and Sex Column")
plt.plot(max_depth_range,acc2,label="Without Age and Sex Column")
plt.xlabel("Depths")
plt.ylabel("Accuracy Obtained")
plt.legend()
plt.show()

"""**HYBRID APPROACH USING LASSO(LINEAR MODEL) AND RANDOM-FOREST**"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_regression
from sklearn.linear_model import Lasso
regr = RandomForestClassifier()
ridge = Lasso(alpha=1.0)
data3 = pd.read_csv("/content/drive/My Drive/heart.csv")
target = data3['target']
data3 = data3.drop(columns=['target','age','sex'],axis=0)
ridge.fit(data3,target)
cof=ridge.coef_
c=0
to_be_d = []
for i in data3.columns:
  if cof[c]<0:
    to_be_d.append(i)
  c=c+1
data3 = data3.drop(columns=to_be_d,axis=0)
x_tr,x_te,y_tr,y_te = train_test_split(data3,target,test_size=0.2)
print(cof)
regr.fit(x_tr, y_tr)
p = regr.predict(x_te)
score_is = cross_val_score(estimator=regr, X=x_tr, y=y_tr, cv=5)

print("**************************************************")
print("Without age and sex column")
print("Accuracy",score_is.mean())
accuracy_dict[11].append(score_is.mean())
fscore_dict[11].append(f1_score(p, y_te, average='macro'))
print("F1 score",f1_score(p, y_te, average='macro'))
print("\n")

from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_regression
from sklearn.linear_model import Lasso
regr = RandomForestClassifier()
ridge = Lasso(alpha=1.0)
data3 = pd.read_csv("/content/drive/My Drive/heart.csv")
target = data3['target']
data3 = data3.drop(columns=['target'],axis=0)
ridge.fit(data3,target)
cof=ridge.coef_
c=0
to_be_d = []
for i in data3.columns:
  if cof[c]<0:
    to_be_d.append(i)
  c=c+1
data3 = data3.drop(columns=to_be_d,axis=0)
x_tr,x_te,y_tr,y_te = train_test_split(data3,target,test_size=0.2)
print(cof)
regr.fit(x_tr, y_tr)
p = regr.predict(x_te)
score_is = cross_val_score(estimator=regr, X=x_tr, y=y_tr, cv=5)

print("**************************************************")
print("With age and sex column")
print("Accuracy",score_is.mean())
accuracy_dict[13].append(score_is.mean())
fscore_dict[13].append(f1_score(p, y_te, average='macro'))
print("F1 score",f1_score(p, y_te, average='macro'))
print("\n")

"""Language Model"""

data = pd.read_csv("/content/drive/My Drive/heart.csv")
y = list(data["target"])
data = data.drop(['target'], axis=1)
X_train, X_test, Y_train, Y_test = train_test_split(data, y, test_size = 0.3)
X_train = X_train.reset_index(drop=True)
X_train["target"] = list(Y_train)
unigram_dict = {}
bigram_dict = {}
X_train_features = list(X_train.columns)
for feature in X_train_features:
  if(feature!="target"):
    # print(feature)
    bigram_dict[feature] = {}
    unigram_dict[feature] = {}
    for i in X_train.index:
      temp_bigram = str(X_train[feature][i])+" "+str(X_train["target"][i])
      temp_unigram = X_train[feature][i]
      if temp_bigram not in bigram_dict[feature]:
        bigram_dict[feature][temp_bigram] = 2
      else:
        bigram_dict[feature][temp_bigram] += 1

      if temp_unigram not in unigram_dict[feature]:
        unigram_dict[feature][temp_unigram] = 2
      else:
        unigram_dict[feature][temp_unigram] += 1


predicted_class_labels = []
for i in X_test.index:
  temp_prob_1 = 0
  temp_prob_0 = 0
  for feature in X_train_features:
    if(feature!="target"):
      temp_bigram_1 = str(X_test[feature][i])+" "+str(1)
      temp_bigram_0 = str(X_test[feature][i])+" "+str(0)
      temp_unigram = X_test[feature][i]
      if(temp_bigram_1 in bigram_dict[feature]):
        temp_prob_1 += bigram_dict[feature][temp_bigram_1]/(unigram_dict[feature][temp_unigram] + len(unigram_dict[feature]))
      elif(temp_unigram in unigram_dict[feature]):
        temp_prob_1 += 1/(unigram_dict[feature][temp_unigram] + len(unigram_dict[feature]))
      else:
        temp_prob_1 += 1/len(unigram_dict[feature])

      if(temp_bigram_0 in bigram_dict[feature]):
        temp_prob_0 += bigram_dict[feature][temp_bigram_0]/(unigram_dict[feature][temp_unigram] + len(unigram_dict[feature]))
      elif(temp_unigram in unigram_dict[feature]):
        temp_prob_0 += 1/(unigram_dict[feature][temp_unigram] + len(unigram_dict[feature]))
      else:
        temp_prob_0 += 1/len(unigram_dict[feature])

  if(temp_prob_1 >= temp_prob_0):
    predicted_class_labels.append(1)
  else:
    predicted_class_labels.append(0)

print("**************************************************")
print("With age and sex column")
print("Accuracy",accuracy_score(Y_test, predicted_class_labels))
accuracy_dict[13].append(accuracy_score(Y_test, predicted_class_labels))
fscore_dict[13].append(f1_score(Y_test, predicted_class_labels, average='macro'))
print("F1 score",f1_score(Y_test, predicted_class_labels, average='macro'))
print("\n")

data = pd.read_csv("/content/drive/My Drive/heart.csv")
y = list(data["target"])
data = data.drop(['target', 'age', 'sex'], axis=1)
X_train, X_test, Y_train, Y_test = train_test_split(data, y, test_size = 0.3)
X_train = X_train.reset_index(drop=True)
X_train["target"] = list(Y_train)
unigram_dict = {}
bigram_dict = {}
X_train_features = list(X_train.columns)
for feature in X_train_features:
  if(feature!="target"):
    # print(feature)
    bigram_dict[feature] = {}
    unigram_dict[feature] = {}
    for i in X_train.index:
      temp_bigram = str(X_train[feature][i])+" "+str(X_train["target"][i])
      temp_unigram = X_train[feature][i]
      if temp_bigram not in bigram_dict[feature]:
        bigram_dict[feature][temp_bigram] = 2
      else:
        bigram_dict[feature][temp_bigram] += 1

      if temp_unigram not in unigram_dict[feature]:
        unigram_dict[feature][temp_unigram] = 2
      else:
        unigram_dict[feature][temp_unigram] += 1


predicted_class_labels = []
for i in X_test.index:
  temp_prob_1 = 0
  temp_prob_0 = 0
  for feature in X_train_features:
    if(feature!="target"):
      temp_bigram_1 = str(X_test[feature][i])+" "+str(1)
      temp_bigram_0 = str(X_test[feature][i])+" "+str(0)
      temp_unigram = X_test[feature][i]
      if(temp_bigram_1 in bigram_dict[feature]):
        temp_prob_1 += bigram_dict[feature][temp_bigram_1]/(unigram_dict[feature][temp_unigram] + len(unigram_dict[feature]))
      elif(temp_unigram in unigram_dict[feature]):
        temp_prob_1 += 1/(unigram_dict[feature][temp_unigram] + len(unigram_dict[feature]))
      else:
        temp_prob_1 += 1/len(unigram_dict[feature])

      if(temp_bigram_0 in bigram_dict[feature]):
        temp_prob_0 += bigram_dict[feature][temp_bigram_0]/(unigram_dict[feature][temp_unigram] + len(unigram_dict[feature]))
      elif(temp_unigram in unigram_dict[feature]):
        temp_prob_0 += 1/(unigram_dict[feature][temp_unigram] + len(unigram_dict[feature]))
      else:
        temp_prob_0 += 1/len(unigram_dict[feature])

  if(temp_prob_1 >= temp_prob_0):
    predicted_class_labels.append(1)
  else:
    predicted_class_labels.append(0)

print("**************************************************")
print("Without age and sex column")
print("Accuracy",accuracy_score(Y_test, predicted_class_labels))
accuracy_dict[11].append(accuracy_score(Y_test, predicted_class_labels))
fscore_dict[11].append(f1_score(Y_test, predicted_class_labels, average='macro'))
print("F1 score",f1_score(Y_test, predicted_class_labels, average='macro'))
print("\n")

print(len(accuracy_dict[11]))
print(len(fscore_dict[11]))
print(len(accuracy_dict[13]))
print(len(fscore_dict[13]))

import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np
import matplotlib.pyplot as plt

list_of_models = ['multilayer', 'svm', 'logistic regression', 'random forest', 'naive bayes', 'knn', 'decision tree', 'HRFLM', 'language']
y_pos = np.arange(len(list_of_models))
print(len(y_pos))
print(len(accuracy_dict[11]))

plt.bar(y_pos, accuracy_dict[11], align='center', alpha=0.5)
plt.xticks(y_pos, list_of_models, rotation='vertical')
plt.ylabel('Accuracy Values')
plt.title('Accuracy plot of different models without "age" and "sex" column')

plt.show()

import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np
import matplotlib.pyplot as plt


y_pos = np.arange(len(list_of_models))

plt.bar(y_pos, accuracy_dict[13], align='center', alpha=0.5)
plt.xticks(y_pos, list_of_models, rotation='vertical')
plt.ylabel('Accuracy Values')
plt.title('Accuracy plot of different models with "age" and "sex" column')

plt.show()

import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np
import matplotlib.pyplot as plt


y_pos = np.arange(len(list_of_models))

plt.bar(y_pos, fscore_dict[11], align='center', alpha=0.5)
plt.xticks(y_pos, list_of_models, rotation='vertical')
plt.ylabel('F-score Values')
plt.title('F-score plot of different models without "age" and "sex" column')

plt.show()

import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np
import matplotlib.pyplot as plt


y_pos = np.arange(len(list_of_models))

plt.bar(y_pos, fscore_dict[13], align='center', alpha=0.5)
plt.xticks(y_pos, list_of_models, rotation='vertical')
plt.ylabel('F-score Values')
plt.title('F-score plot of different models with "age" and "sex" column')

plt.show()
