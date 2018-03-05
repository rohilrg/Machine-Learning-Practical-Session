import sys
from sklearn.datasets import load_iris
from sklearn.model_selection import KFold
from sklearn import neighbors
import matplotlib.pyplot as plt
irisData = load_iris()
X = irisData.data
Y = irisData.target
x = 0
y = 1
X = irisData.data
Y = irisData.target
kf = KFold(n_splits=10, shuffle=True)
scores = []
for k in range(1,30):
	score = 0
	clf = neighbors.KNeighborsClassifier(k)
	for learn,test in kf.split(X):
		X_train = X[learn]
		Y_train = Y[learn]
		clf.fit(X_train, Y_train)
		X_test = X[test]
		Y_test = Y[test]
		score = score + clf.score(X_test, Y_test)
	scores.append(score)
print(scores)
print("best k:", scores.index(max(scores))+1)