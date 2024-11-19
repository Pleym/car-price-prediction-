import math

import tensorflow as tf
from datasets import Dataset, load_dataset
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error , r2_score
import numpy as np

# Load dataset
ds = load_dataset("VarunKumarGupta2003/Car-Price-Dataset")
ds = ds.remove_columns(["Model"])

# Remove the title row (assuming it's the first row)
ds_pandas = ds["train"].to_pandas()
ds_pandas = ds_pandas.iloc[1:].reset_index(drop=True)

# Convert back to a Hugging Face Dataset
X = ds_pandas.drop(columns=['Price'])
y = ds_pandas['Price'].astype(float)

#Hot spot encoding
categorical_columns = X.select_dtypes(include=['object']).columns.tolist()
encoder = OneHotEncoder(sparse_output=False)
one_hot_encoded = encoder.fit_transform(X[categorical_columns])
one_hot_df = pd.DataFrame(one_hot_encoded,columns=encoder.get_feature_names_out(categorical_columns))
X_encoded = pd.concat([X.drop(columns=categorical_columns).reset_index(drop=True),one_hot_df],axis=1)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

# Ensure float type for the features
X_train = X_train.values.astype(float)
X_test = X_test.values.astype(float)


# Normalize
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Define the model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(532, activation='relu', input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1)  # Output layer
])

# Compile the model
model.compile(loss='mean_absolute_error',
              optimizer=tf.keras.optimizers.Adam(0.001))

# Train the model
model.fit(X_train, y_train, epochs=100, batch_size=128, validation_split=0.2, verbose=1)

# Make predictions
predictions = model(X_train[:1]).numpy()
print(predictions)

# Probability
proba = tf.nn.softmax(predictions).numpy()
print(proba)

# Evaluate the model
print(model.evaluate(X_test, y_test))
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test,y_pred)
r2 = r2_score(y_test,y_pred)
mse = mean_squared_error(y_test,y_pred)
rmse = math.sqrt(mse)

print(f"RMSE: {rmse}")
print(f"Mean Absolute Error :{mae}")
print(f"R^2 Score: {r2}")


# Clean up resources
tf.keras.backend.clear_session()

