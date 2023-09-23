# Importing necessary libraries and modules
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# Loading the Iris dataset
iris = load_iris()

# Separating features and labels from the dataset
features = iris.data  # Features (attributes) of the dataset
labels = iris.target  # Corresponding labels (target values)

# Splitting the dataset into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.2)

# Initializing and training a KNN classifier with 7 neighbors
model = KNeighborsClassifier(n_neighbors=7)
model.fit(x_train, y_train)

# Predicting labels for the test set
y_pred = model.predict(x_test)

# Calculating and printing the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print(accuracy * 100)
