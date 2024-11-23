# 1. Import Libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt

# 2. Load Dataset
data = pd.read_csv("telecom_churn.csv")  # Replace with the correct dataset path
print(data.head())

# 3. Data Preprocessing
# Prepare Features and Target
X = data.drop("Churn", axis=1)
y = data["Churn"]

# Standardize numeric features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

print("Training data shape:", X_train.shape)
print("Test data shape:", X_test.shape)

# 4. Handle Class Imbalance Using SMOTE
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

print("Resampled training data shape:", X_train_res.shape)
print("Class distribution after SMOTE:", np.bincount(y_train_res))

# 5. Logistic Regression with L1 Regularization
lr = LogisticRegression(penalty='l1', solver='liblinear', random_state=42)
lr.fit(X_train_res, y_train_res)
y_pred_lr = lr.predict(X_test)

print("\nLogistic Regression Results:")
print("Accuracy:", accuracy_score(y_test, y_pred_lr))
print(classification_report(y_test, y_pred_lr))

# 6. Random Forest with Feature Importance
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train_res, y_train_res)
y_pred_rf = rf.predict(X_test)

print("\nRandom Forest Results:")
print("Accuracy:", accuracy_score(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf))

# Plot feature importance
feature_importance = pd.Series(rf.feature_importances_, index=data.columns[:-1]).sort_values(ascending=False)
feature_importance[:10].plot(kind="bar", title="Top 10 Feature Importance", color='skyblue')
plt.show()

# 7. Gradient Boosting with Hyperparameter Tuning
gb = GradientBoostingClassifier(random_state=42)
param_grid = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 5, 7]
}

gb_search = RandomizedSearchCV(gb, param_grid, cv=3, scoring='accuracy', n_iter=10, random_state=42)
gb_search.fit(X_train_res, y_train_res)

best_gb = gb_search.best_estimator_
y_pred_gb = best_gb.predict(X_test)

print("\nGradient Boosting Results:")
print("Accuracy:", accuracy_score(y_test, y_pred_gb))
print(classification_report(y_test, y_pred_gb))

# 8. Neural Network with Dropout
mlp = MLPClassifier(hidden_layer_sizes=(64, 32), activation='relu', solver='adam', max_iter=200, random_state=42)
mlp.fit(X_train_res, y_train_res)
y_pred_mlp = mlp.predict(X_test)

print("\nNeural Network Results:")
print("Accuracy:", accuracy_score(y_test, y_pred_mlp))
print(classification_report(y_test, y_pred_mlp))

# 9. Compare Model Performance
model_results = {
    'Logistic Regression': accuracy_score(y_test, y_pred_lr),
    'Random Forest': accuracy_score(y_test, y_pred_rf),
    'Gradient Boosting': accuracy_score(y_test, y_pred_gb),
    'Neural Network': accuracy_score(y_test, y_pred_mlp)
}

# Print model comparison
print("\nModel Comparison:")
for model, acc in model_results.items():
    print(f"{model}: {acc:.4f}")

# Plot model performance
plt.figure(figsize=(8, 6))
plt.bar(model_results.keys(), model_results.values(), color='teal')
plt.title("Model Accuracy Comparison")
plt.ylabel("Accuracy")
plt.xticks(rotation=45)
plt.show()



