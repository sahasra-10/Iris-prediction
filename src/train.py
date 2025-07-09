import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load iris data
df = pd.read_csv('data/iris.data', header=None)
df.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']

# Features and target
X = df.drop('species', axis=1)
y = df['species']

# Train model
model = RandomForestClassifier()
model.fit(X, y)

# Save model
joblib.dump(model, 'model/iris_model.pkl')
print(" Iris model trained and saved.")
