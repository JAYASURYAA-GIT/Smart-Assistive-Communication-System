import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Load dataset
data = pd.read_csv("data/landmark_data.csv", header=None)

# Split features and labels
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# Encode labels (A,B,C → 0,1,2)
encoder = LabelEncoder()
y = encoder.fit_transform(y)

# Split training & testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Build simple neural network
model = Sequential([
    Dense(128, activation='relu', input_shape=(63,)),
    Dense(64, activation='relu'),
    Dense(len(np.unique(y)), activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train model
model.fit(X_train, y_train, epochs=20, validation_data=(X_test, y_test))

# Save model
model.save("models/landmark_model.h5")

print("Classes:", encoder.classes_)
print("Model trained and saved successfully.")