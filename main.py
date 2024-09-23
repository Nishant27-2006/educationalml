
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error, accuracy_score

# Load the dataset
data = pd.read_csv('Students_gamification_grades.csv')

# Data Preprocessing
data_cleaned = data.drop(['Student_ID'], axis=1).fillna(0)

# Split data into features and target
X = data_cleaned.drop(['Final_Exam'], axis=1)
y = data_cleaned['Final_Exam']

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 1. Random Forest Model
rf_model = RandomForestRegressor(random_state=42)
rf_model.fit(X_train, y_train)
rf_predictions = rf_model.predict(X_test)
rf_mse = mean_squared_error(y_test, rf_predictions)
print(f"Random Forest MSE: {rf_mse}")

# 2. K-Means Clustering
kmeans_model = KMeans(n_clusters=3, random_state=42)
kmeans_model.fit(X)
kmeans_labels = kmeans_model.labels_
print(f"K-Means Inertia: {kmeans_model.inertia_}")

# 3. Logistic Regression
y_binary = (y > y.median()).astype(int)
X_train_bin, X_test_bin, y_train_bin, y_test_bin = train_test_split(X, y_binary, test_size=0.2, random_state=42)

log_reg_model = LogisticRegression(max_iter=1000, random_state=42)
log_reg_model.fit(X_train_bin, y_train_bin)
log_reg_predictions = log_reg_model.predict(X_test_bin)
log_reg_accuracy = accuracy_score(y_test_bin, log_reg_predictions)
print(f"Logistic Regression Accuracy: {log_reg_accuracy}")

# Visualization
fig, ax = plt.subplots(1, 3, figsize=(15, 5))

# Random Forest - scatter plot of true vs predicted values
ax[0].scatter(y_test, rf_predictions, color='skyblue', edgecolor='black')
ax[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
ax[0].set_title('Random Forest: True vs Predicted')
ax[0].set_xlabel('True Values')
ax[0].set_ylabel('Predicted Values')

# K-Means - bar chart of cluster sizes
unique_labels, counts = np.unique(kmeans_labels, return_counts=True)
ax[1].bar(unique_labels, counts, color='lightgreen', edgecolor='black')
ax[1].set_title('K-Means: Cluster Sizes')
ax[1].set_xlabel('Cluster')
ax[1].set_ylabel('Number of Students')

# Logistic Regression - confusion matrix-like plot
log_reg_confusion = np.array([[np.sum((y_test_bin == 1) & (log_reg_predictions == 1)),
                               np.sum((y_test_bin == 1) & (log_reg_predictions == 0))],
                              [np.sum((y_test_bin == 0) & (log_reg_predictions == 1)),
                               np.sum((y_test_bin == 0) & (log_reg_predictions == 0))]])

im = ax[2].imshow(log_reg_confusion, cmap='Blues')
ax[2].set_title('Logistic Regression: Confusion Matrix')
ax[2].set_xticks([0, 1])
ax[2].set_yticks([0, 1])
ax[2].set_xticklabels(['Predicted 1', 'Predicted 0'])
ax[2].set_yticklabels(['Actual 1', 'Actual 0'])

for i in range(2):
    for j in range(2):
        ax[2].text(j, i, log_reg_confusion[i, j], ha='center', va='center', color='black')

plt.tight_layout()
plt.show()
