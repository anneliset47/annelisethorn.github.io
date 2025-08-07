import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
import tensorflow as tf

# Load ClinVar dataset
df = pd.read_csv('/Users/annelisethorn/Documents/GitHub/annelisethorn.github.io/Datasets/Uncleaned/clinvar_repeat_pathogenic_variants.csv')

# Add a binary label column: 1 if "pathogenic" is in the ClinicalSignificance
df['label'] = df['ClinicalSignificance'].apply(lambda x: 1 if 'pathogenic' in str(x).lower() else 0)

# Encode the Gene names as integers (can later replace with one-hot if needed)
gene_encoder = LabelEncoder()
df['Gene_encoded'] = gene_encoder.fit_transform(df['Gene'].astype(str))

# Features and labels
X = df[['Gene_encoded']]
y = df['label']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# Convert to NumPy arrays
X_train_np = X_train.to_numpy()
X_test_np = X_test.to_numpy()
y_train_np = y_train.to_numpy()
y_test_np = y_test.to_numpy()

# Define a simple neural network
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(X_train_np.shape[1],)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')  # Binary classification
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train_np, y_train_np, epochs=20, batch_size=16, validation_split=0.2, verbose=1)

# Evaluate
loss, accuracy = model.evaluate(X_test_np, y_test_np)
print(f"\nTest Accuracy: {accuracy:.4f}")

# Predict and report
y_pred_probs = model.predict(X_test_np).flatten()
y_pred = (y_pred_probs > 0.5).astype(int)
print("\nClassification Report:")
print(classification_report(y_test_np, y_pred))
