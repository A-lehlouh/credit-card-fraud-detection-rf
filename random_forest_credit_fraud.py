# ============================================================
# Project 4: Random Forest - Credit Card Fraud Detection
# Group 8 | Pattern Recognition | Al-Ahliyya Amman University
# Course: A0374502 | Dr. Ayman Mohamed
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, classification_report,
                              confusion_matrix, ConfusionMatrixDisplay)

# ── 1. Load Dataset ───────────────────────────────────────────
print("=" * 50)
print("Step 1: Loading Dataset")
print("=" * 50)

df = pd.read_csv('pr4_CreditcardDataset.csv')
print(f"Dataset shape: {df.shape}")
print(f"Columns: {list(df.columns)}")
print(f"\nClass Distribution:")
print(df['Class'].value_counts())
print(f"\nAny missing values? {df.isnull().sum().sum()}")

# ── 2. Data Preprocessing ────────────────────────────────────
print("\n" + "=" * 50)
print("Step 2: Preprocessing")
print("=" * 50)

# Drop missing values (if any)
df = df.dropna()

# Separate features and label
X = df.drop('Class', axis=1)
y = df['Class'].astype(int)

# Normalize the features (StandardScaler: mean=0, std=1)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print("Features normalized using StandardScaler.")

# Train/Test Split: 80% train, 20% test
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y)

print(f"Training samples: {len(X_train)}")
print(f"Testing  samples: {len(X_test)}")

# ── 3. Train Random Forest ───────────────────────────────────
print("\n" + "=" * 50)
print("Step 3: Training Random Forest")
print("=" * 50)

rf = RandomForestClassifier(
    n_estimators=100,   # 100 decision trees
    max_depth=5,        # max depth of each tree
    random_state=42
)
rf.fit(X_train, y_train)
print("Model training complete!")

# ── 4. Evaluate ──────────────────────────────────────────────
print("\n" + "=" * 50)
print("Step 4: Evaluation")
print("=" * 50)

y_pred = rf.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print(f"\nAccuracy: {acc * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred,
      target_names=["Normal (0)", "Fraud (1)"]))

# ── 5. Confusion Matrix ──────────────────────────────────────
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(5, 4))
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                               display_labels=['Normal (0)', 'Fraud (1)'])
disp.plot(ax=ax, colorbar=False, cmap='Greens')
ax.set_title('Confusion Matrix', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=150)
plt.show()
print("Confusion matrix saved as 'confusion_matrix.png'")

# ── 6. Feature Importance ────────────────────────────────────
importances = pd.Series(rf.feature_importances_, index=X.columns)
top10 = importances.nlargest(10)

fig, ax = plt.subplots(figsize=(7, 4))
top10.sort_values().plot(kind='barh', ax=ax, color='#2e7d32', edgecolor='white')
ax.set_title('Top 10 Feature Importances', fontsize=13, fontweight='bold')
ax.set_xlabel('Importance Score')
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=150)
plt.show()
print("Feature importance chart saved as 'feature_importance.png'")

print("\n" + "=" * 50)
print("Done! Project 4 - Group 8")
print("=" * 50)
