import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, precision_recall_fscore_support
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline

# Step 1: Load the data
data = pd.read_csv('creditcard.csv')

# Step 2: Data Preprocessing
# Separate features and target
X = data.drop('Class', axis=1)
y = data['Class']

# Normalize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 3: Handle Class Imbalance using SMOTE for oversampling the minority class
smote = SMOTE(sampling_strategy=0.5, random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_scaled, y)

# Alternative approach: Random undersampling the majority class
# rus = RandomUnderSampler(sampling_strategy=0.5, random_state=42)
# X_resampled, y_resampled = rus.fit_resample(X_scaled, y)

# Step 4: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.3, random_state=42)

# Step 5: Train the model using Logistic Regression
# model = LogisticRegression()
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Step 6: Make predictions and evaluate the model
y_pred = model.predict(X_test)

# Step 7: Evaluate the model using Precision, Recall, and F1-Score
precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1-Score: {f1:.2f}")

# Detailed classification report
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Optional: Feature Importance for Random Forest
if isinstance(model, RandomForestClassifier):
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    print("\nFeature ranking:")
    for f in range(X.shape[1]):
        print(f"{f + 1}. feature {indices[f]} ({importances[indices[f]]:.6f})")
