#Set the working directory #LOGISTIC REGRESSION(linear classifier) #CLASSIFICATION TEMPLATE 

#preprocessing 

import pandas as nd 

import numpy as np 

import matplotlib.pyplot as plt 

Dataset = pd.read_csv() 

X= dataset.iloc[:, [2,3]].values 

y=datatset.iloc[:, 4].values 

from sklearn.cross_validation import train_test_split 

X_train, y_train ,X_test, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0) 

from sklearn.preprocessing import StandardScaler 

sc_X=StandardScaler() 

X_train=sc_X.fit_transform(X_train) 

X_test=sc_X.transform(X_test)//Not scaled as its categorical binary data 

 

#fitting on train_set 

from sklearn.linear_model import LogisticRegression 

classifier = LogisticRegression(random_state = 0) 

classifier.fit(X_train, y_train) 

 

#prediction on test_set 

y_pred = classifier.predict(X_test) 

 

#confusion matrix 

from sklearn.matrix import confusion_matrix 

cm = confusion_matrix(y_test, y_pred) 

 

#visualising the train_set results 

from matplotlib.colors impor ListedColormap 

X_set, y_set = X_train, y_train 

X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min()-1, stop = X_set[:, 0].max()+1, step =0.01), 

np.arange(start = X_set[:, 1].min()-1, stop =X_set[:, 1].max()+1, step =0.01)) 

plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(),X2.ravel()]).T).reshape(X1.shape), 

alpha = .075, cmap = ListedColorMap(('red', 'green'))) 

plt.xlim(X1.min(), X1.max()) 

plt.ylim(X2.min(), X2.max()) 

for i,j in enumerate(np.unique(y_set)): 

plt.scatter(X_set[y_set==j,0], X_set[y_set == j, 1], 

c = ListedColorMap(('red', 'green'))(i), label =j) 

plt.title('Logistic Regression on Training set') 

plt.xlabel('Age') 

plt.ylabel('Estimated Salary') 

plt.legend() 

plt.show() 

 

#visualising the test_set results 

from matplotlib.colors impor ListedColormap 

X_set, y_set = X_test, y_test 

X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min()-1, stop = X_set[:, 0].max()+1, step =0.01), 

np.arange(start = X_set[:, 1].min()-1, stop =X_set[:, 1].max()+1, step =0.01)) 

plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(),X2.ravel()]).T).reshape(X1.shape), 

alpha = .075, cmap = ListedColorMap(('red', 'green'))) 

plt.xlim(X1.min(), X1.max()) 

plt.ylim(X2.min(), X2.max()) 

for i,j in enumerate(np.unique(y_set)): 

plt.scatter(X_set[y_set==j,0], X_set[y_set == j, 1], 

c = ListedColorMap(('red', 'green'))(i), label =j) 

plt.title('Logistic Regression on Test set') 

plt.xlabel('Age') 

plt.ylabel('Estimated Salary') 

plt.legend() 

plt.show() 