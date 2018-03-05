import sys
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
irisData = load_iris()
X = irisData.data
Y = irisData.target
x = 0
y = 1
# ^ fill-in with imports and data loading...
colors = ["red", "green", "blue"]
for i in range(3):
	plt.scatter(X[Y==i][:, x], X[Y==i][:,y], c=colors[i], label=irisData.target_names[i])
plt.legend()
plt.xlabel(irisData.feature_names[x])
plt.ylabel(irisData.feature_names[y])
plt.title("Iris Data - size of the sepals only")
if len(sys.argv) > 1:
	plt.savefig(sys.argv[1])
else:
	plt.show()