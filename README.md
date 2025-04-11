# Supervised Learning on credit risk assessment.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('/content/credit_risk_dataset.csv.zip')

print(df.head())

print(df.info())
!mkdir -p eda
print(df.describe())
print(df.isnull().sum())

#  Class Distribution
plt.figure(figsize=(6,4))
sns.countplot(x='loan_status', data=df, palette='Set2')
plt.title('Loan Status Distribution')
plt.xlabel('Loan Status')
plt.ylabel('Count')
plt.show()

# Correlation Heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(df.select_dtypes(include=np.number).corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap')
plt.show()

# Boxplot of Loan Amount by Loan Status
plt.figure(figsize=(8, 5))
sns.boxplot(x='loan_status', y='loan_amnt', data=df, palette='Set3')
plt.title('Loan Amount by Loan Status')
plt.show()

print(df.columns)

# Countplot
plt.figure(figsize=(7, 4))
sns.countplot(x='person_home_ownership', hue='loan_status', data=df)
plt.title('Home Ownership vs Loan Status')
plt.show()

# Pie Chart for Home Ownership
labels = df['person_home_ownership'].value_counts().index
sizes = df['person_home_ownership'].value_counts().values

plt.figure(figsize=(6,6))
plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=sns.color_palette('pastel'))
plt.title('Distribution of Home Ownership')
plt.show()

# model training
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay

# Data Preprocessing
df_copy = df.copy()

le = LabelEncoder()
for col in df_copy.select_dtypes(include='object').columns:
    df_copy[col] = le.fit_transform(df_copy[col])

# Train-Test Split
X = df_copy.drop('loan_status', axis=1)
y = df_copy['loan_status']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train Model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Make Predictions
y_pred = model.predict(X_test)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)

# Display Confusion Matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
disp.plot(cmap='Blues')
plt.title('Confusion Matrix')
plt.show()

# Classification Report
print("Classification Report:\n")
print(classification_report(y_test, y_pred, target_names=le.classes_))
