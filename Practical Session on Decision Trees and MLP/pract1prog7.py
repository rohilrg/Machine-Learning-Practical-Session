#The MLP is made on digits dataset while tuning different parameters given below.

from sklearn.neural_network import MLPClassifier
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import random
digits = load_digits()
digits.data[0]
digits.images[0]
digits.data[0].reshape(8,8)
digits.target[0]

TaX= digits.data
TX= digits.target
clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(20,10, 10,8,9),random_state=1)
X_train,X_test,Y_train,Y_test = train_test_split(TaX,TX, test_size=0.3,random_state=random.seed())
print(X_train.shape) 
print(X_test.shape)

clf.fit(X_train, Y_train)

Y_pred =clf.predict(X_test)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, Y_pred)
print(cm)
print(clf.score(X_test,Y_test))

clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(10,10,10, 12),random_state=1, learning_rate= 'adaptive')
clf.fit(X_train, Y_train)
Y_pred =clf.predict(X_test)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, Y_pred)
print(cm)
print(clf.score(X_test,Y_test))