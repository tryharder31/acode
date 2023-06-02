from sklearn.datasets import make_classification, load_iris
from sklearn.ensemble import RandomForestClassifier
import numpy as np

# now load the iris dataset  inty X,y
iris = load_iris()
X = iris.data
y = iris.target

clf = RandomForestClassifier()
clf.fit(X, y)

def get_top_k_features(clf, K=500):
    importances = clf.feature_importances_
    indices = np.argsort(importances)[::-1]

    return indices[:K]

print(top_K_features_indices)