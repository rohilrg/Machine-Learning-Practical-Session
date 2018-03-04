from sklearn.datasets import load_iris
from sklearn import tree
iris = load_iris()
clf = tree.DecisionTreeClassifier(max_leaf_nodes= 3) 
clf = clf.fit(iris.data, iris.target) 

tree.export_graphviz(clf, out_file='C:/Users/Rohil/Desktop/tree1.dot')