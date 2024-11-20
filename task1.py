import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
iris = load_iris()
X = iris.data
y = iris.target

# Select two classes (0 and 1 = "Setosa" and "Versicolor") and their corresponding features
X_binary = X[y != 2][
    :, :2
]  # Use first two features (sepal length and width) for visualization
y_binary = y[y != 2]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X_binary, y_binary, test_size=0.3, random_state=42
)

# Train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions for the test set
y_pred = model.predict(X_test)
y_classified = (y_pred >= 0.5).astype(int)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_classified)
print(f"Accuracy: {accuracy}")

# Generate confusion matrix
conf_matrix = confusion_matrix(y_test, y_classified)

# Plot the confusion matrix
plt.figure(figsize=(6, 5))
sns.heatmap(
    conf_matrix,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Setosa", "Versicolor"],
    yticklabels=["Setosa", "Versicolor"],
)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()

# Create a mesh grid for plotting decision boundaries
x_min, x_max = X_train[:, 0].min() - 0.5, X_train[:, 0].max() + 0.5
y_min, y_max = X_train[:, 1].min() - 0.5, X_train[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))

# Predict probabilities for the grid points
grid_points = np.c_[xx.ravel(), yy.ravel()]
grid_predictions = model.predict(grid_points)
grid_classifications = (grid_predictions >= 0.5).astype(int).reshape(xx.shape)

# Plot the decision boundaries
plt.figure(figsize=(10, 6))
plt.contourf(xx, yy, grid_classifications, alpha=0.6, cmap=plt.cm.Paired)
plt.scatter(
    X_train[:, 0],
    X_train[:, 1],
    c=y_train,
    edgecolor="k",
    cmap=plt.cm.Paired,
    marker="o",
    label="Training Set",
)
plt.scatter(
    X_test[:, 0],
    X_test[:, 1],
    c=y_test,
    edgecolor="k",
    cmap=plt.cm.Paired,
    marker="x",
    label="Test Set",
)
plt.title("Decision Boundaries for Binary Classification (Linear Regression)")
plt.xlabel("Sepal Length")
plt.ylabel("Sepal Width")
plt.legend()
plt.show()
