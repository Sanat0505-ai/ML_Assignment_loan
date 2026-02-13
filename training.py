import pandas as pd
import numpy as np
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# ---------------- LOAD DATA ----------------
data = pd.read_csv("train.csv")

# Clean column names (remove spaces)
data.columns = data.columns.str.strip()

# Remove ID column safely
data.drop(columns=["Loan_ID"], errors="ignore", inplace=True)

# ---------------- TARGET CONVERSION ----------------
# Y/N → 1/0
data["Loan_Status"] = data["Loan_Status"].map({"Y": 1, "N": 0})

# ---------------- HANDLE MISSING VALUES ----------------

# Numeric columns → fill with median
num_cols = data.select_dtypes(include=np.number).columns
for col in num_cols:
    data[col].fillna(data[col].median(), inplace=True)

# Categorical columns → fill with mode
cat_cols = data.select_dtypes(include="object").columns
for col in cat_cols:
    data[col].fillna(data[col].mode()[0], inplace=True)

# ---------------- ENCODING ----------------
data = pd.get_dummies(data)

# Create model folder if not exists
os.makedirs("model", exist_ok=True)

# Save column names
joblib.dump(data.columns, "model/columns.pkl")

# ---------------- SPLIT FEATURES ----------------
X = data.drop("Loan_Status", axis=1)
y = data["Loan_Status"]

# ---------------- SCALING ----------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

joblib.dump(scaler, "model/scaler.pkl")

# ---------------- TRAIN TEST SPLIT ----------------
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# ---------------- MODELS ----------------
models = {
    "logistic": LogisticRegression(max_iter=1000),
    "tree": DecisionTreeClassifier(),
    "knn": KNeighborsClassifier(),
    "bayes": GaussianNB(),
    "forest": RandomForestClassifier(),
    "xgb": XGBClassifier(use_label_encoder=False, eval_metric="logloss")
}

# ---------------- TRAIN + SAVE ----------------
for name, model in models.items():
    model.fit(X_train, y_train)
    joblib.dump(model, f"model/{name}.pkl")
    print(f"{name} model trained & saved successfully")
