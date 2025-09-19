#Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report

# Load dataset
url = 'https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv'
df = pd.read_csv(url)
df.head()

# Basic EDA
print(df.shape)
print(df.info())
df.describe()

# Correlation heatmap
plt.figure(figsize=(10,6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.show()

# Distribution plots
df.hist(figsize=(10, 8))
plt.tight_layout()
plt.show()

X = df.drop('Outcome', axis=1)
y = df['Outcome']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(X_train.shape, X_test.shape)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# KNN Model
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train)
y_pred_knn = knn.predict(X_test_scaled)

print("KNN Results:")
print("Accuracy:", accuracy_score(y_test, y_pred_knn))
print(confusion_matrix(y_test, y_pred_knn))
print(classification_report(y_test, y_pred_knn))

# SVM Model
svm = SVC(kernel="linear", random_state=42)
svm.fit(X_train_scaled, y_train)
y_pred_svm = svm.predict(X_test_scaled)

print("SVM Results:")
print("Accuracy:", accuracy_score(y_test, y_pred_svm))
print(confusion_matrix(y_test, y_pred_svm))
print(classification_report(y_test, y_pred_svm))

# Confusion Matrix for KNN
plt.figure(figsize=(6, 4))
sns.heatmap(confusion_matrix(y_test, y_pred_knn), annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix - KNN')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Interpretation of Confusion Matrix:
# The confusion matrix shows the counts of true positive, true negative, false positive, and false negative predictions.
# - True Negatives (Top-Left): Correctly predicted as negative (no diabetes).
# - False Positives (Top-Right): Incorrectly predicted as positive (diabetes) when actual is negative. (Type I error)
# - False Negatives (Bottom-Left): Incorrectly predicted as negative (no diabetes) when actual is positive. (Type II error)
# - True Positives (Bottom-Right): Correctly predicted as positive (diabetes).

# Confusion Matrix for SVM
plt.figure(figsize=(6, 4))
sns.heatmap(confusion_matrix(y_test, y_pred_svm), annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix - SVM')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# ROC Curve
from sklearn.metrics import roc_curve

y_prob_knn = knn.predict_proba(X_test_scaled)[:, 1]
y_prob_svm = svm.decision_function(X_test_scaled)

fpr_knn, tpr_knn, _ = roc_curve(y_test, y_prob_knn)
fpr_svm, tpr_svm, _ = roc_curve(y_test, y_prob_svm)

plt.figure(figsize=(8, 6))
plt.plot(fpr_knn, tpr_knn, label=f'KNN (AUC = {roc_auc_score(y_test, y_prob_knn):.2f})')
plt.plot(fpr_svm, tpr_svm, label=f'SVM (AUC = {roc_auc_score(y_test, y_prob_svm):.2f})')
plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()

# Interpretation of ROC Curve and AUC:
# The ROC (Receiver Operating Characteristic) curve plots the True Positive Rate (Sensitivity) against the False Positive Rate (1 - Specificity) at various threshold settings.
# AUC (Area Under the Curve) represents the degree or measure of separability. It tells how much the model is capable of distinguishing between classes. Higher AUC means the model is better at predicting 0s as 0s and 1s as 1s.
# An AUC of 0.5 suggests a random guess model, while an AUC of 1.0 represents a perfect model.

# Metric Comparison Table
metrics = {
    'Model': ['KNN', 'SVM'],
    'Accuracy': [accuracy_score(y_test, y_pred_knn), accuracy_score(y_test, y_pred_svm)],
    'Precision': [precision_score(y_test, y_pred_knn), precision_score(y_test, y_pred_svm)],
    'Recall': [recall_score(y_test, y_pred_knn), recall_score(y_test, y_pred_svm)],
    'F1-Score': [f1_score(y_test, y_pred_knn), f1_score(y_test, y_pred_svm)],
    'ROC-AUC': [roc_auc_score(y_test, y_prob_knn), roc_auc_score(y_test, y_prob_svm)]
}

metrics_df = pd.DataFrame(metrics)
print("\nMetric Comparison Table:")
display(metrics_df)

# Interpretation of Metric Comparison Table:
# - Accuracy: Overall correctness of the model.
# - Precision: Ability of the model to correctly identify positive cases (out of all predicted positives). High precision means fewer false positives.
# - Recall: Ability of the model to find all positive cases (out of all actual positives). High recall means fewer false negatives.
# - F1-Score: The harmonic mean of Precision and Recall, providing a balance between the two.
# - ROC-AUC: Area under the ROC curve, a measure of the model's ability to distinguish between positive and negative classes. Higher is better.

# Load the Breast Cancer dataset
from sklearn.datasets import load_breast_cancer

breast_cancer = load_breast_cancer()
X_bc = pd.DataFrame(breast_cancer.data, columns=breast_cancer.feature_names)
y_bc = pd.Series(breast_cancer.target, name='target')

print("Breast Cancer Dataset:")
display(X_bc.head())
display(y_bc.head())

# Basic EDA for Breast Cancer dataset
print("Breast Cancer Dataset Shape:", X_bc.shape)
print("\nBreast Cancer Dataset Info:")
X_bc.info()
print("\nBreast Cancer Dataset Description:")
display(X_bc.describe())

# Check for missing values
print("\nMissing values in Breast Cancer dataset:")
print(X_bc.isnull().sum())

# Correlation heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(X_bc.corr(), annot=False, cmap='coolwarm')
plt.title('Correlation Heatmap - Breast Cancer Dataset')
plt.show()

# Distribution plots
X_bc.hist(figsize=(15, 12))
plt.tight_layout()
plt.show()

# Data Preprocessing & Train/Test Split for Breast Cancer dataset
X_train_bc, X_test_bc, y_train_bc, y_test_bc = train_test_split(X_bc, y_bc, test_size=0.2, random_state=42, stratify=y_bc)
print(X_train_bc.shape, X_test_bc.shape)

# Scale features for Breast Cancer dataset
scaler_bc = StandardScaler()
X_train_bc_scaled = scaler_bc.fit_transform(X_train_bc)
X_test_bc_scaled = scaler_bc.transform(X_test_bc)

# KNN Model for Breast Cancer dataset
knn_bc = KNeighborsClassifier(n_neighbors=5)
knn_bc.fit(X_train_bc_scaled, y_train_bc)
y_pred_knn_bc = knn_bc.predict(X_test_bc_scaled)

print("KNN Results (Breast Cancer):")
print("Accuracy:", accuracy_score(y_test_bc, y_pred_knn_bc))
print(confusion_matrix(y_test_bc, y_pred_knn_bc))
print(classification_report(y_test_bc, y_pred_knn_bc))

# SVM Model for Breast Cancer dataset
svm_bc = SVC(kernel="linear", random_state=42)
svm_bc.fit(X_train_bc_scaled, y_train_bc)
y_pred_svm_bc = svm_bc.predict(X_test_bc_scaled)

print("SVM Results (Breast Cancer):")
print("Accuracy:", accuracy_score(y_test_bc, y_pred_svm_bc))
print(confusion_matrix(y_test_bc, y_pred_svm_bc))
print(classification_report(y_test_bc, y_pred_svm_bc))

# Confusion Matrix for KNN (Breast Cancer)
plt.figure(figsize=(6, 4))
sns.heatmap(confusion_matrix(y_test_bc, y_pred_knn_bc), annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix - KNN (Breast Cancer)')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Interpretation of Confusion Matrix:
# The confusion matrix shows the counts of true positive, true negative, false positive, and false negative predictions.
# - True Negatives (Top-Left): Correctly predicted as negative (no breast cancer).
# - False Positives (Top-Right): Incorrectly predicted as positive (breast cancer) when actual is negative. (Type I error)
# - False Negatives (Bottom-Left): Incorrectly predicted as negative (no breast cancer) when actual is positive. (Type II error)
# - True Positives (Bottom-Right): Correctly predicted as positive (breast cancer).

# Confusion Matrix for SVM (Breast Cancer)
plt.figure(figsize=(6, 4))
sns.heatmap(confusion_matrix(y_test_bc, y_pred_svm_bc), annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix - SVM (Breast Cancer)')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# ROC Curve (Breast Cancer)
from sklearn.metrics import roc_curve

# For SVM, use decision_function as it provides the distance to the hyperplane,
# which is needed for ROC curve when probability=False in SVC.
# For KNN, use predict_proba to get probabilities.
y_prob_svm_bc = svm_bc.decision_function(X_test_bc_scaled)

fpr_knn_bc, tpr_knn_bc, _ = roc_curve(y_test_bc, knn_bc.predict_proba(X_test_bc_scaled)[:, 1])
fpr_svm_bc, tpr_svm_bc, _ = roc_curve(y_test_bc, y_prob_svm_bc)

plt.figure(figsize=(8, 6))
plt.plot(fpr_knn_bc, tpr_knn_bc, label=f'KNN (AUC = {roc_auc_score(y_test_bc, knn_bc.predict_proba(X_test_bc_scaled)[:, 1]):.2f})')
plt.plot(fpr_svm_bc, tpr_svm_bc, label=f'SVM (AUC = {roc_auc_score(y_test_bc, y_prob_svm_bc):.2f})')
plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Breast Cancer Dataset')
plt.legend()
plt.show()

# Interpretation of ROC Curve and AUC:
# The ROC (Receiver Operating Characteristic) curve plots the True Positive Rate (Sensitivity) against the False Positive Rate (1 - Specificity) at various threshold settings.
# AUC (Area Under the Curve) represents the degree or measure of separability. It tells how much the model is capable of distinguishing between classes. Higher AUC means the model is better at predicting 0s as 0s and 1s as 1s.
# An AUC of 0.5 suggests a random guess model, while an AUC of 1.0 represents a perfect model.

# Metric Comparison Table (Breast Cancer)
metrics_bc = {
    'Model': ['KNN', 'SVM'],
    'Accuracy': [accuracy_score(y_test_bc, y_pred_knn_bc), accuracy_score(y_test_bc, y_pred_svm_bc)],
    'Precision': [precision_score(y_test_bc, y_pred_knn_bc), precision_score(y_test_bc, y_pred_svm_bc)],
    'Recall': [recall_score(y_test_bc, y_pred_knn_bc), recall_score(y_test_bc, y_pred_svm_bc)],
    'F1-Score': [f1_score(y_test_bc, y_pred_knn_bc), f1_score(y_test_bc, y_pred_svm_bc)],
    'ROC-AUC': [roc_auc_score(y_test_bc, knn_bc.predict_proba(X_test_bc_scaled)[:, 1]), roc_auc_score(y_test_bc, y_prob_svm_bc)]
}

metrics_df_bc = pd.DataFrame(metrics_bc)
print("\nMetric Comparison Table (Breast Cancer):")
display(metrics_df_bc)

# Interpretation of Metric Comparison Table:
# - Accuracy: Overall correctness of the model.
# - Precision: Ability of the model to correctly identify positive cases (out of all predicted positives). High precision means fewer false positives.
# - Recall: Ability of the model to find all positive cases (out of all actual positives). High recall means fewer false negatives.
# - F1-Score: The harmonic mean of Precision and Recall, providing a balance between the two.
# - ROC-AUC: Area under the ROC curve, a measure of the model's ability to distinguish between positive and negative classes. Higher is better.
