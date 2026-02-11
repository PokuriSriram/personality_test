import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
import pickle as pkl

# Load dataset
df = pd.read_csv("data-final.csv", sep="\t")

# Select only 10 EXT-related questions
selected_cols = [
    "EXT1", "EXT2", "EXT3", "EXT4", "EXT5",
    "EXT6", "EXT7", "EXT8", "EXT9", "EXT10"
]

# Keep only required columns
df = df[selected_cols].copy()

# Drop missing values
df = df.dropna()

# Features & label
X = df[selected_cols]
y = (df[selected_cols].mean(axis=1) > 3).astype(int)  # 1 = Extrovert, 0 = Introvert

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    random_state=42
)

model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Save model + feature order
with open("personality_model.pkl", "wb") as file:
    pkl.dump((model, selected_cols), file)

print("Model saved as personality_model.pkl")
