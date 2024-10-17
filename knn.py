import numpy as np
import pandas as pd
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


def calculate_distance(point1, point2):
    return np.sqrt(np.sum((point1 - point2)**2))

class MyKNN:
    def __init__(self, k=3):
        self.k = k
        self.X_train = None
        self.y_train = None

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test):
        predictions = [self.make_prediction(x) for x in X_test]
        return predictions

    def make_prediction(self, x):

        distances = [calculate_distance(x, x_train) for x_train in self.X_train]
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        most_common_label = Counter(k_nearest_labels).most_common(1)[0][0]
        return most_common_label


cmap = ListedColormap(['#FF0000','#00FF00','#0000FF'])

file_path = 'data.csv'

df = pd.read_csv(file_path)

label_encoder = LabelEncoder()
df_encoded = df.apply(label_encoder.fit_transform)

labels = df_encoded['Genus'].values
features = df_encoded.drop(['Genus', 'Family', 'Species'], axis=1).values

X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.1, train_size=0.3, random_state=1234)

# plt.figure()
# plt.scatter(features[:, 2], features[:, 3], c=labels, cmap=cmap, edgecolor='k', s=20)
# plt.show()

clf = MyKNN(k=5)
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)

# print(predictions)
acc = np.sum(predictions == y_test) / len(y_test)
print(acc)

test_accuracies = []

for k in range(1, 51):
    knn_model = MyKNN(k=k)
    knn_model.fit(X_train, y_train)
    predictions = knn_model.predict(X_test)
    accuracy = np.sum(predictions == y_test) / len(y_test)
    test_accuracies.append(accuracy)

plt.plot(range(1, 51), test_accuracies, marker='o')
plt.title('Test Accuracy vs. k')
plt.xlabel('k')
plt.ylabel('Test Accuracy')
plt.grid(True)
plt.show()

"""Part3: Using Scikit-Learn functions"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

# Load the dataset
file_path = 'Q1Data.csv'
df = pd.read_csv(file_path)

# Split features and labels
X = df.drop(['Genus', 'Family', 'Species'], axis=1)
y = df['Genus']

# Decision Trees
depths = range(1, 16)
dt_test_accuracies = []

for depth in depths:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)
    dt_classifier = DecisionTreeClassifier(max_depth=depth)
    dt_classifier.fit(X_train, y_train)
    dt_test_accuracy = dt_classifier.score(X_test, y_test)
    dt_test_accuracies.append(dt_test_accuracy)

# k Nearest Neighbors
ks = range(1, 1001)
knn_test_accuracies = []

for k in ks:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)
    knn_classifier = KNeighborsClassifier(n_neighbors=k)
    knn_classifier.fit(X_train, y_train)
    knn_test_accuracy = knn_classifier.score(X_test, y_test)
    knn_test_accuracies.append(knn_test_accuracy)

# Plotting
plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.plot(depths, dt_test_accuracies, marker='o')
plt.title('Decision Tree Test Accuracy vs. Depth')
plt.xlabel('Depth')
plt.ylabel('Test Accuracy')
plt.grid(True)

plt.subplot(2, 2, 2)
plt.plot(ks, knn_test_accuracies, marker='o')
plt.title('kNN Test Accuracy vs. k')
plt.xlabel('k')
plt.ylabel('Test Accuracy')
plt.grid(True)

plt.tight_layout()
plt.show()

ks = range(1, 1001)

plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.plot(depths, dt_test_accuracies, marker='o')
plt.title('Decision Tree Test Accuracy vs. Depth')
plt.xlabel('Depth')
plt.ylabel('Test Accuracy')
plt.grid(True)

plt.subplot(2, 2, 2)
plt.plot(ks, knn_test_accuracies, marker='o')
plt.title('kNN Test Accuracy vs. k')
plt.xlabel('k')
plt.ylabel('Test Accuracy')
plt.grid(True)

plt.subplot(2, 2, 3)
plt.plot(depths, acc_dt, marker='o')
plt.title('Decision Tree - Own Test Accuracy vs. Depth')
plt.xlabel('Depth')
plt.ylabel('Test Accuracy')
plt.grid(True)

plt.subplot(2, 2, 4)
plt.plot(list(range(1,51)), test_accuracies, marker='o')
plt.title('kNN - Own Test Accuracy vs. k')
plt.xlabel('k')
plt.ylabel('Test Accuracy')
plt.grid(True)

plt.tight_layout()
plt.show()

