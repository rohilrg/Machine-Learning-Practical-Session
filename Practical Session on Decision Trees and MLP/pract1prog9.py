from sklearn import tree
clf1= tree.DecisionTreeClassifier(criterion= 'entropy', max_leaf_nodes=20)
clf1.fit(features_data,target_data)
clf1.score(features_data,target_data)

clf1.fit(X_train,Y_train)
Y_pred=clf1.predict(X_test)
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

from sklearn.tree import export_graphviz
loo = LeaveOneOut()
avg_score =[]
i = 1
for train_index, test_index in loo.split(X):

	
	X_trian, X_test = X[train_index], X[test_index]
	Y_train, Y_test = Y[train_index], Y[test_index]
	clf1.fit(X_trian, Y_train)
	avg_score.append(clf1.score(X_test,Y_test))
	tree.export_graphviz(clf1, out_file= 'part_6_3(' +str(i)+')_dt.dot')
	i = i + 1

print("\nThe average score using LeaveOneOut Cross-validation : " + 
	str(sum(avg_score)/len(avg_score)))
print("The scores over 25 iterations are : \n" + str(avg_score))