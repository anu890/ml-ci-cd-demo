import pandas as pd
from sklearn.linear_model import LogisticRegression
import pickle
import os

os.makedirs("Model", exist_ok=True)
os.makedirs("Results", exist_ok=True)

data = pd.read_csv("Data/dummy_data.csv")
X = data[['feature1', 'feature2']]
y = data['target']

model = LogisticRegression()
model.fit(X, y)

with open("Model/dummy_model.pkl", "wb") as f:
    pickle.dump(model, f)

accuracy = model.score(X, y)
with open("Results/metrics.txt", "w") as f:
    f.write(f"Accuracy: {accuracy:.4f}\n")
print(f"Training complete. Accuracy: {accuracy:.4f}")
