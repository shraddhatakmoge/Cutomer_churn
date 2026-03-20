import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Load dataset
df = pd.read_csv("Telco-Customer-Churn.csv")

# Clean TotalCharges
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

# Drop customerID
df.drop("customerID", axis=1, inplace=True)

# Encode target
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

# Features and target
X = df.drop("Churn", axis=1)
y = df["Churn"]

# One-hot encoding
X = pd.get_dummies(X, drop_first=True)

# Save model columns as normal Python list
model_columns = X.columns.tolist()

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Final model
rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=10,
    random_state=42,
    class_weight="balanced"
)

rf.fit(X_train, y_train)

# Save model
with open("random_forest_model.pkl", "wb") as f:
    pickle.dump(rf, f)

# Save columns
with open("model_columns.pkl", "wb") as f:
    pickle.dump(model_columns, f)

print("Model and columns saved successfully.")