# Titanic Survival Prediction Project
# Author: Asrar Ahmed (BCA, Data Science)
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt

data = pd.read_csv("Titanic-Dataset.csv")

features = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
X = data[features]
y = data["Survived"]
X = X.copy()
for col in ["Sex", "Embarked"]:
    X[col] = LabelEncoder().fit_transform(X[col].astype(str))
imputer = SimpleImputer(strategy="median")
X = pd.DataFrame(imputer.fit_transform(X), columns=features)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("✅ Model Evaluation Results:")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
importances = model.feature_importances_
plt.figure(figsize=(8, 5))
plt.barh(features, importances, color="teal")
plt.xlabel("Importance Score")
plt.ylabel("Feature")
plt.title("Feature Importance in Titanic Survival Prediction")
plt.show()
