clf = tree.DecisionTreeClassifier(criterion='gini') 
clf = clf.fit(iris.data, iris.target) 
tree.export_graphviz(clf, out_file='C:/Users/Rohil/Desktop/gini-iris.dot')


clf = tree.DecisionTreeClassifier(criterion='entropy') 
clf = clf.fit(iris.data, iris.target) 
tree.export_graphviz(clf, out_file='C:/Users/Rohil/Desktop/entropy-iris.dot')
