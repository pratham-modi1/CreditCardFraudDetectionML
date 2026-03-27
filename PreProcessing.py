# preprocessing.py
#do kfold at end

# -------------------- 1. Override print (optional formatting) --------------------
import builtins
def print(*args, **kwargs):
    kwargs.setdefault("end", "\n\n")
    return builtins.print(*args, **kwargs)


# -------------------- 2. Import libraries --------------------
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------- 3. Load dataset --------------------
df = pd.read_csv("creditcard.csv")

# Basic checks (optional)
# print(df.shape)
# print(df.head())
# print(df.info())
# print(df.describe())
# print(df["Class"].value_counts(normalize=True))


# -------------------- 4. Remove duplicates --------------------
# Duplicate rows can bias model
df = df.drop_duplicates()


# -------------------- 5. Feature Engineering --------------------
# Convert Time → Hour (more meaningful)
df["Hour"] = (df["Time"] / 3600) % 24

# Drop original Time column
df.drop(columns=["Time"], inplace=True)


# -------------------- 6. Handle skewed feature --------------------
# Log transform Amount to reduce skewness
df['Amount'] = np.log1p(df['Amount'])


# -------------------- 7. Split data (VERY IMPORTANT before scaling) --------------------
X = df.drop("Class", axis=1)  #all columns except class as input
y = df["Class"] #output is just the reqd class

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,        # maintain fraud ratio
    random_state=42
)
# random_state ensures reproducibility (same train-test split every run)
# Without it → different split & results each time
# Any fixed number works (42 is just a common choice)
# With stratify=y:
# Train and test sets maintain same class distribution
# Example:
# Train → Fraud: 0.17%, Normal: 99.83%
# Test  → Fraud: 0.17%, Normal: 99.83%
# → Ensures balanced and reliable evaluation

# -------------------- 8. Scaling (fit only on train → avoid data leakage) --------------------
scaler = StandardScaler()

# Scale Amount
X_train['Amount'] = scaler.fit_transform(X_train[['Amount']])
X_test['Amount'] = scaler.transform(X_test[['Amount']])

# Scale Hour
X_train['Hour'] = scaler.fit_transform(X_train[['Hour']])
X_test['Hour'] = scaler.transform(X_test[['Hour']])

# -------------------- DONE --------------------
# Now data is clean and ready for EDA / modeling

df = X_train.copy()
df["Class"] = y_train

# print(df['Class'].value_counts())
# print(df['Class'].value_counts(normalize=True)*100)

fraud = df[df['Class'] == 1]
legit = df[df['Class'] == 0]

# fig, axes = plt.subplots(1, 2, figsize=(14, 4))

# axes[0].hist(legit['Amount'], bins=50, color='blue', alpha=0.7)
# axes[0].set_title('Legit Transaction Amounts')
# axes[0].set_xlabel('Amount (log scaled)')

# axes[1].hist(fraud['Amount'], bins=50, color='red', alpha=0.7)
# axes[1].set_title('Fraud Transaction Amounts')
# axes[1].set_xlabel('Amount (log scaled)')

# plt.tight_layout()
# plt.show()
# print("Legit Amount Stats:")
# print(legit['Amount'].describe())
# print("\nFraud Amount Stats:")
# print(fraud['Amount'].describe())

# fig, axes = plt.subplots(1, 2, figsize=(14, 4))

# axes[0].hist(legit['Hour'], bins=24, color='blue', alpha=0.7)
# axes[0].set_title('Legit Transactions by Hour')
# axes[0].set_xlabel('Hour (scaled)')

# axes[1].hist(fraud['Hour'], bins=24, color='red', alpha=0.7)
# axes[1].set_title('Fraud Transactions by Hour')
# axes[1].set_xlabel('Hour (scaled)')

# plt.tight_layout()
# plt.show()
# plt.figure(figsize=(6, 4))
# sns.boxplot(x='Class', y='Hour', data=df)

# plt.title('Hour vs Class')
# plt.show()

# sns.boxplot(x='Class', y='Amount', data=df)

# plt.title('amt vs Class')
# plt.show()
#corr = df.corr()

# Correlation with target
# corr_with_class = corr['Class'].sort_values(ascending=False)

# print(corr_with_class)

#top3 = v11, v4 and v2, bottom3 = v12,14,17

# sns.boxplot(x='Class', y='V11', data=df)

# plt.title('v11 vs Class')
# plt.show()
#sns.boxplot(x='Class', y='V17', data=df)

# plt.title('v17 vs Class')
# plt.show()
#repeat for all others now

# sns.boxplot(x='Class', y='V4', data=df)
# plt.title('v4 vs Class')
# plt.show()

# -------------------- 9. Model Building --------------------

from sklearn.linear_model import LogisticRegression

# # Create model (handling imbalance)
model = LogisticRegression(class_weight='balanced', max_iter=1000)

# # Train model
model.fit(X_train, y_train)


# # -------------------- 10. Predictions --------------------

# # Predict on FULL test set
y_pred = model.predict(X_test)

# # Probabilities (important for future threshold tuning)
y_prob = model.predict_proba(X_test)[:, 1]


# # -------------------- 11. Evaluation --------------------

# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

# print("Accuracy:", accuracy_score(y_test, y_pred))

# print("Precision:", precision_score(y_test, y_pred))

# print("Recall:", recall_score(y_test, y_pred))

# print("F1 Score:", f1_score(y_test, y_pred))

# print("\nConfusion Matrix:")
# print(confusion_matrix(y_test, y_pred))

# print("\nClassification Report:")
# print(classification_report(y_test, y_pred))

# import numpy as np
# from sklearn.metrics import precision_score, recall_score, f1_score

# thresholds = np.arange(0.1, 0.9, 0.05)

# best_f1 = 0
# best_threshold = 0

# print("Threshold | Precision | Recall | F1")
# print("-------------------------------------")

# for t in thresholds:

#     y_pred_t = []
#     for prob in y_prob:
#         if prob >= t:
#             y_pred_t.append(1)
#         else:
#             y_pred_t.append(0)

#     p = precision_score(y_test, y_pred_t)
#     r = recall_score(y_test, y_pred_t)
#     f1 = f1_score(y_test, y_pred_t)

#     print(f"{t:.2f}      | {p:.3f}     | {r:.3f}  | {f1:.3f}")

#     if f1 > best_f1:
#         best_f1 = f1
#         best_threshold = t

# print("\nBest Threshold (F1-based):", best_threshold)

# -------------------- 12. Decision Tree Model --------------------

# from sklearn.tree import DecisionTreeClassifier

# # Create model (limit depth to prevent overfitting)
# dt_model = DecisionTreeClassifier(
#     max_depth=5,              # control complexity
#     class_weight='balanced',  # handle imbalance
#     random_state=42
# )

# # Train model
# dt_model.fit(X_train, y_train)


# # -------------------- 13. Predictions --------------------

# y_pred_dt = dt_model.predict(X_test)

# # Probabilities (for threshold tuning later)
# y_prob_dt = dt_model.predict_proba(X_test)[:, 1]


# # -------------------- 14. Evaluation --------------------

# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

# print("Decision Tree Results:\n")

# print("Accuracy:", accuracy_score(y_test, y_pred_dt))
# print("Precision:", precision_score(y_test, y_pred_dt))
# print("Recall:", recall_score(y_test, y_pred_dt))
# print("F1 Score:", f1_score(y_test, y_pred_dt))

# print("\nConfusion Matrix:")
# print(confusion_matrix(y_test, y_pred_dt))

# print("\nClassification Report:")
# print(classification_report(y_test, y_pred_dt))

# -------------------- 15. Random Forest Model --------------------

from sklearn.ensemble import RandomForestClassifier

# Create model
rf_model = RandomForestClassifier(
    n_estimators=100,        # number of trees
    max_depth=8,             # control overfitting
    class_weight='balanced', # handle imbalance
    random_state=42,
    n_jobs=-1                # use all CPU cores
)

# Train model
rf_model.fit(X_train, y_train)


# -------------------- 16. Predictions --------------------

y_pred_rf = rf_model.predict(X_test)

# Probabilities for threshold tuning
y_prob_rf = rf_model.predict_proba(X_test)[:, 1]


# -------------------- 17. Evaluation --------------------

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

# print("Random Forest Results:\n")

# print("Accuracy:", accuracy_score(y_test, y_pred_rf))
# print("Precision:", precision_score(y_test, y_pred_rf))
# print("Recall:", recall_score(y_test, y_pred_rf))
# print("F1 Score:", f1_score(y_test, y_pred_rf))

# print("\nConfusion Matrix:")
# print(confusion_matrix(y_test, y_pred_rf))

# print("\nClassification Report:")
# print(classification_report(y_test, y_pred_rf))

import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score

thresholds = np.arange(0.2, 0.7, 0.05)

print("Threshold | Precision | Recall | F1")
print("-------------------------------------")

best_choice = None

for t in thresholds:

    y_pred_t = []
    for prob in y_prob_rf:
        if prob >= t:
            y_pred_t.append(1)
        else:
            y_pred_t.append(0)

    p = precision_score(y_test, y_pred_t)
    r = recall_score(y_test, y_pred_t)
    f1 = f1_score(y_test, y_pred_t)

    print(f"{t:.2f}      | {p:.3f}     | {r:.3f}  | {f1:.3f}")

    # Store good balance (you can tweak this condition)
    if r >= 0.80 and p >= 0.60:
        best_choice = (t, p, r, f1)

if best_choice:
    print("\nGood Balanced Threshold Found:")
    print("Threshold:", best_choice[0])
    print("Precision:", best_choice[1])
    print("Recall:", best_choice[2])
    print("F1:", best_choice[3])