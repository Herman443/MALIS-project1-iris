import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Use sepal length and sepal width for visualization
feature_indices = [0, 1]  # Indices of sepal length and sepal width
X = X[:, feature_indices]
feature_names = np.array(iris.feature_names)[feature_indices]

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Train a linear regression model for each class
models = {}
for class_label in np.unique(y):
    # Create binary labels for the current class
    y_train_binary = (y_train == class_label).astype(int)

    # Train a linear regression model
    model = LinearRegression()
    model.fit(X_train, y_train_binary)

    # Store the model
    models[class_label] = model

# Create a mesh grid for plotting
x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
xx, yy = np.meshgrid(
    np.linspace(x_min, x_max, 300),
    np.linspace(y_min, y_max, 300),
)

# Prepare grid for prediction
grid_points = np.c_[xx.ravel(), yy.ravel()]

# Predict probabilities for each class
probabilities = np.zeros((grid_points.shape[0], len(models)))
for class_label, model in models.items():
    probabilities[:, class_label] = model.predict(grid_points)

# Assign class labels based on the highest probability
predicted_labels = np.argmax(probabilities, axis=1)
predicted_labels = predicted_labels.reshape(xx.shape)

# Make predictions for the test set
test_probabilities = np.zeros((X_test.shape[0], len(models)))
for class_label, model in models.items():
    test_probabilities[:, class_label] = model.predict(X_test)
test_predictions = np.argmax(test_probabilities, axis=1)

# Calculate and print accuracy
accuracy = accuracy_score(y_test, test_predictions)
print(f"Accuracy on Test Set: {accuracy * 100:.2f}%")

# Generate confusion matrix
conf_matrix = confusion_matrix(y_test, test_predictions)

# Plot the confusion matrix
plt.figure(figsize=(6, 5))
sns.heatmap(
    conf_matrix,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=iris.target_names,
    yticklabels=iris.target_names,
)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()

# Plot the decision boundaries
plt.figure(figsize=(10, 8))
plt.contourf(xx, yy, predicted_labels, alpha=0.3, cmap=plt.cm.rainbow)

# Plot training set
plt.scatter(
    X_train[:, 0],
    X_train[:, 1],
    c=y_train,
    edgecolor="k",
    cmap=plt.cm.rainbow,
    marker="o",
    label="Training Set",
)

# Plot test set
plt.scatter(
    X_test[:, 0],
    X_test[:, 1],
    c=y_test,
    edgecolor="k",
    cmap=plt.cm.rainbow,
    marker="x",
    label="Test Set",
)

plt.title("Decision Boundaries for Linear Regression (Sepal Length & Sepal Width)")
plt.xlabel(feature_names[0])
plt.ylabel(feature_names[1])
plt.legend()
plt.colorbar(label="Class")
plt.show()
