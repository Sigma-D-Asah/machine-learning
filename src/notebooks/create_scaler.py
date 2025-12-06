"""
Script to create and save MinMaxScaler based on ACTUAL training data.
This ensures API uses the EXACT same normalization as during training.
"""
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split

# Load the actual training data
script_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(script_dir, 'predictive_maintenance_new.csv')

print("Loading data from:", data_path)
dfraw = pd.read_csv(data_path)
print(f"✓ Data loaded: {len(dfraw)} rows")

# Preprocess exactly as in training
df = dfraw.drop(['id', 'product_id'], axis=1)

# Label encode categorical columns (type, target, failure_type)
cat = df.select_dtypes(include=['object']).columns.tolist()
print(f"Categorical columns to encode: {cat}")

for col in cat:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    if col == 'type':
        print(f"  Type encoding: {dict(zip(le.classes_, le.transform(le.classes_)))}")

# Split features and targets
X = df.drop(['target', 'failure_type'], axis=1)
yb = df[['target']]
yc = df[['failure_type']]

print(f"\nFeatures (X) columns: {X.columns.tolist()}")
print(f"Features shape: {X.shape}")

# Split data (same as training)
X_train, X_test, yb_train, yb_test, yc_train, yc_test = train_test_split(
    X, yb, yc,
    test_size=0.3,
    random_state=42
)

# Create and fit scaler on training data (exactly as in training)
mm = MinMaxScaler(feature_range=(0, 1))
num_cols = X_train.select_dtypes(include=['int64', 'float64']).columns
print(f"\nNumeric columns to normalize: {num_cols.tolist()}")

# Fit scaler on training data
mm.fit(X_train[num_cols])

# Save scaler
models_dir = os.path.join(os.path.dirname(script_dir), 'src', 'models')
scaler_path = os.path.join(models_dir, 'scaler.pkl')

joblib.dump(mm, scaler_path)
print(f"\n✓ Scaler saved to: {scaler_path}")
print(f"\nScaler parameters:")
print(f"  Feature names: {num_cols.tolist()}")
print(f"  Min values: {mm.data_min_}")
print(f"  Max values: {mm.data_max_}")
print(f"  Scale: {mm.scale_}")

# Verify with test sample (L47257)
test_sample = pd.DataFrame({
    'air_temperature': [298.8],
    'process_temperature': [308.9],
    'rotational_speed': [1455],
    'torque': [41.3],
    'tool_wear': [208],
    'type': [0]  # L = 0
})

normalized = mm.transform(test_sample)
print(f"\n✓ Test normalization for L47257:")
print(f"  Original: {test_sample.values[0]}")
print(f"  Normalized: {normalized[0]}")
