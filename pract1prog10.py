from sklearn.neural_network import MLPClassifier
clf2 = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(20,20,20),random_state=1)
clf2.fit(features_data, target_data)

clf2.score(features_data,target_data)

clf2.fit(X_train,Y_train)
Y_pred=clf2.predict(X_test)
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
	clf2.fit(X_trian, Y_train)
	avg_score.append(clf2.score(X_test,Y_test))

print("\nThe average score using LeaveOneOut Cross-validation : " + 
	str(sum(avg_score)/len(avg_score)))

print("The scores over 25 iterations are : \n" + str(avg_score))