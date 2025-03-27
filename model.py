# main.py
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
from joblib import dump

# Generate dummy data for training
X, y = make_classification(n_samples=100, n_features=10, random_state=42)

# Train a Logistic Regression model
model = LogisticRegression()
model.fit(X, y)

# Save the model to a file
dump(model, 'scikit_model.joblib')
print("Model saved as 'scikit_model.joblib'")