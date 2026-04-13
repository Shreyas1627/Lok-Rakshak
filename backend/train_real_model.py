# train_real_model.py
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

print("🚀 Generating synthetic crowd-physics dataset...")

# --- 1. DEFINE CROWD PHYSICS RULES ---
# Features: [density, vector_variance, compression_score, acoustic_score, sdk_score]
num_samples_per_class = 1000

# CLASS 0: GREEN (Normal, calm flow)
green_data = pd.DataFrame({
    'density': np.random.randint(0, 40, num_samples_per_class),
    'vector_variance': np.random.uniform(0.0, 0.3, num_samples_per_class), # Calm, unified movement
    'compression_score': np.random.uniform(0.0, 0.2, num_samples_per_class),
    'acoustic_score': np.random.choice([0], num_samples_per_class), # No panic words
    'sdk_score': np.random.randint(0, 20, num_samples_per_class),
    'label': 0
})

# CLASS 1: YELLOW (Crowded, but moving. Minor bottlenecks)
yellow_data = pd.DataFrame({
    'density': np.random.randint(40, 90, num_samples_per_class),
    'vector_variance': np.random.uniform(0.2, 0.6, num_samples_per_class), # Slight friction
    'compression_score': np.random.uniform(0.2, 0.5, num_samples_per_class),
    'acoustic_score': np.random.choice([0, 1], num_samples_per_class, p=[0.8, 0.2]), # Occasional loud noise
    'sdk_score': np.random.randint(20, 50, num_samples_per_class),
    'label': 1
})

# CLASS 2: RED (Dangerous density, flow is breaking down)
red_data = pd.DataFrame({
    'density': np.random.randint(90, 150, num_samples_per_class),
    'vector_variance': np.random.uniform(0.5, 1.2, num_samples_per_class), # High friction, people stopping
    'compression_score': np.random.uniform(0.5, 0.8, num_samples_per_class),
    'acoustic_score': np.random.choice([1, 2], num_samples_per_class, p=[0.5, 0.5]), # Shouting detected
    'sdk_score': np.random.randint(50, 80, num_samples_per_class),
    'label': 2
})

# CLASS 3: CRITICAL (Active crush/stampede or active panic rumor)
# Notice how acoustic_score=3 (screaming/panic words) immediately spikes the risk.
critical_data = pd.DataFrame({
    'density': np.random.randint(130, 250, num_samples_per_class),
    'vector_variance': np.random.uniform(1.0, 3.0, num_samples_per_class), # Chaotic movement in all directions
    'compression_score': np.random.uniform(0.7, 1.0, num_samples_per_class), # Total bottleneck
    'acoustic_score': np.random.choice([2, 3], num_samples_per_class, p=[0.3, 0.7]), # Mass panic/rumor detected
    'sdk_score': np.random.randint(75, 100, num_samples_per_class),
    'label': 3
})

# --- 2. PREPARE THE DATA ---
# Combine all classes into one dataset
df = pd.concat([green_data, yellow_data, red_data, critical_data], ignore_index=True)

X = df[['density', 'vector_variance', 'compression_score', 'acoustic_score', 'sdk_score']]
y = df['label']

# Split into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- 3. TRAIN THE RANDOM FOREST ---
print(f"🧠 Training Random Forest on {len(X_train)} samples...")
# n_estimators=100 is standard for a robust model without being too slow
rf_model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
rf_model.fit(X_train, y_train)

# --- 4. EVALUATE & VERIFY ---
y_pred = rf_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\n✅ Model Trained Successfully! Accuracy on Test Data: {accuracy * 100:.2f}%\n")

print("📊 Classification Report:")
print(classification_report(y_test, y_pred, target_names=["GREEN", "YELLOW", "RED", "CRITICAL"]))

# Show the judges what the AI cares about!
importances = rf_model.feature_importances_
print("\n🔍 Feature Importances (Show this to the Judges!):")
for feature, imp in zip(X.columns, importances):
    print(f" - {feature}: {imp * 100:.1f}%")

# --- 5. EXPORT THE MODEL ---
model_path = 'app/ml/random_forest.pkl'
with open(model_path, 'wb') as f:
    pickle.dump(rf_model, f)
print(f"\n💾 Real model saved to '{model_path}'! Ready for inference.")