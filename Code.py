# Import necessary libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE

# Load dataset
df = pd.read_csv("creditcard.csv")

# Check the shape and overview
print(f"Dataset shape: {df.shape}")
print(df.head())

# Check for null values
print(f"Null values:\n{df.isnull().sum()}")

# Feature scaling (normalization) for 'Amount'
scaler = StandardScaler()
df['Amount'] = scaler.fit_transform(df[['Amount']])

# Drop unnecessary columns
df = df.drop(columns=['Time'])

# Check for class distribution (imbalanced dataset)
sns.countplot(x='Class', data=df)
plt.title("Fraud vs Non-Fraud Transactions")
plt.show()

# Separate features and target variable
X = df.drop(columns=['Class'])
y = df['Class']

# Split data into train and test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Apply SMOTE to balance the dataset
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

# Initialize Logistic Regression model
log_reg = LogisticRegression()

# Train the model on the balanced data
log_reg.fit(X_train_res, y_train_res)

# Make predictions on the test set
y_pred = log_reg.predict(X_test)

# Evaluate the model's accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()

# Classification report
print("Classification Report:\n", classification_report(y_test, y_pred))

# Optional: Show distribution of predicted classes
sns.countplot(x=y_pred)
plt.title("Predicted Fraud vs Non-Fraud Transactions")
plt.show()
