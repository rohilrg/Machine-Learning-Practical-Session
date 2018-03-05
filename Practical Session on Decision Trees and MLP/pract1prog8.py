from sklearn import neighbors
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import random

#1
nb_neighb = 3
clf = neighbors.KNeighborsClassifier(nb_neighb)

a= ['Feature 1', 'Feature 2', 'Target']
data= pd.read_csv('C:/Users/Rohil/Documents/GitHub/Basic-Machine-Learning-Understanding/my_data_gen.csv', names= a)
data.head()

features_data= data.iloc[:,0:2]
target_data= np.ravel(data.iloc[:,2:3])
target_data1= data.iloc[:,2:3]
clf.fit(features_data,target_data)
clf.score(features_data,target_data)
#2
X_train,X_test,Y_train,Y_test = train_test_split(features_data,target_data, test_size=0.3,random_state=random.seed())
clf.fit(X_train,Y_train)
Y_pred=clf.predict(X_test)
print("Target Variable Predicted",Y_pred )
print ("Target Variable Original", Y_test)
for i in range(1,12):
    if Y_pred[i]==Y_test[i]:
        print('The correctly predicted examples are', i)
    else:
        print('The incorrectly predicted examples are', i)
    
plt.scatter(Y_test,Y_pred)
plt.xlabel('Target Variable Original')
plt.ylabel('Target Variable Predicted')
plt.show()
#3
from sklearn.model_selection import LeaveOneOut
loo = LeaveOneOut()
avg_score =[]
for train_index, test_index in loo.split(X):

	X_trian, X_test = X[train_index], X[test_index]
	Y_train, Y_test = Y[train_index], Y[test_index]
	clf.fit(X_trian, Y_train)
	avg_score.append(clf.score(X_test,Y_test))

print("\nThe average score using LeaveOneOut Cross-validation : " + 
	str(sum(avg_score)/len(avg_score)))

print("The scores over 25 iterations are : \n" + str(avg_score))