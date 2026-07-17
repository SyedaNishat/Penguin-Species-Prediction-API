import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# -----------------------------
# Load the dataset
# -----------------------------
df = pd.read_csv("data/penguins.csv")

# -----------------------------
# Select required columns
# -----------------------------
df = df[
    [
        "bill_length_mm",
        "bill_depth_mm",
        "flipper_length_mm",
        "body_mass_g",
        "species",
    ]
]

# -----------------------------
# Remove rows with missing values
# -----------------------------
df = df.dropna()

# -----------------------------
# Split features and target
# -----------------------------
X = df[
    [
        "bill_length_mm",
        "bill_depth_mm",
        "flipper_length_mm",
        "body_mass_g",
    ]
]

y = df["species"]

# -----------------------------
# Train-Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
)

# -----------------------------
# Train Random Forest Model
# -----------------------------
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# -----------------------------
# Model Evaluation
# -----------------------------
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print(f"Test Accuracy: {accuracy:.4f}")

# -----------------------------
# Create model directory
# -----------------------------
os.makedirs("model", exist_ok=True)

# -----------------------------
# Save model
# -----------------------------
model_path = "model/penguin_model.pkl"

joblib.dump(model, model_path)

print(f"Model saved successfully at: {model_path}")