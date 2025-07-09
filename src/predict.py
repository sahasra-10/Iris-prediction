import pandas as pd
import joblib

# Load trained model
model = joblib.load('model/iris_model.pkl')

# Example input (sepal length, sepal width, petal length, petal width)
sample = pd.DataFrame({
    'sepal_length': [5.1],
    'sepal_width': [3.5],
    'petal_length': [1.4],
    'petal_width': [0.2]
})

# Predict species
prediction = model.predict(sample)[0]
print(f" Predicted Species: {prediction}")
