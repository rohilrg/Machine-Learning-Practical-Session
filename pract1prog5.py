from sklearn.model_selection import train_test_split
import random
from sklearn import datasets
from sklearn import tree
def main():
    #generate random dataset X, Y
    X, Y = datasets.make_classification(n_samples=100000,n_features=20,n_informative=15,n_classes=3)
    #split data into train and test data
    X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.3, random_state=random.seed())
    #iterate i 20times
    for i in range(1,20):
        #set a score value
        x = 0
        y = 0
        #call our classification function
        clf = tree.DecisionTreeClassifier(max_leaf_nodes=500*i)
        #fit X-train and Y-train to decision tress classifier
        clf.fit(X_train, Y_train)
        #after calculating the score for both the train and test
        #We noticed the phonenon called: Overfitting
        y = y + clf.score(X_train, Y_train)
        x = x + clf.score(X_test, Y_test)
        print("Test: %6.4f" %x,"Train: %6.4f" %y)
       
        
if __name__ == '__main__':
    main()
