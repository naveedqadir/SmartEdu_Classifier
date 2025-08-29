"""Train multiple classifiers on the synthetic student dataset and save the best model.
"""
import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

DATA_PATH = 'data/students.csv'
os.makedirs('models', exist_ok=True)
os.makedirs('outputs', exist_ok=True)

print('Loading data...')
df = pd.read_csv(DATA_PATH)
print(df.head())

# Preprocessing
features = ['MST_Score', 'Quiz_Avg', 'Attendance', 'Assignment_Score']
X = df[features].copy()
y = df['Category'].copy()

imp = SimpleImputer(strategy='mean')
X_imp = pd.DataFrame(imp.fit_transform(X), columns=features)

scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X_imp), columns=features)

le = LabelEncoder()
y_enc = le.fit_transform(y)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_enc, test_size=0.25, random_state=42, stratify=y_enc)

models = {
    'LogisticRegression': LogisticRegression(max_iter=1000, multi_class='auto'),
    'DecisionTree': DecisionTreeClassifier(random_state=42),
    'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
    'KNN': KNeighborsClassifier(n_neighbors=5),
    'SVC': SVC(probability=True, random_state=42)
}

results = {}
for name, model in models.items():
    print(f'Training {name}...')
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(name, 'accuracy=', acc)
    results[name] = {'model': model, 'accuracy': acc}

# Cross-val for best model choice
print('\nCross-validation (5-fold) mean accuracies:')
for name, info in results.items():
    scores = cross_val_score(info['model'], X_scaled, y_enc, cv=5)
    print(name, scores.mean())
    results[name]['cv_mean'] = scores.mean()

# Choose best by cv_mean
best_name = max(results.keys(), key=lambda k: results[k]['cv_mean'])
best_model = results[best_name]['model']
print(f'Best model: {best_name}')

# Save best model and preprocessing objects
joblib.dump({'model': best_model, 'scaler': scaler, 'imputer': imp, 'label_encoder': le}, 'models/best_model.joblib')
print('Saved best model to models/best_model.joblib')

# Evaluation report for best model
preds = best_model.predict(X_test)
print('\nClassification report:')
print(classification_report(y_test, preds, target_names=le.classes_))
cm = confusion_matrix(y_test, preds)

plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=le.classes_, yticklabels=le.classes_, cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.savefig('outputs/confusion_matrix.png')
print('Saved confusion matrix to outputs/confusion_matrix.png')

# Feature distributions (quick EDA images)
plt.figure(figsize=(8,6))
for i, col in enumerate(features, 1):
    plt.subplot(2,2,i)
    sns.histplot(df[col].dropna(), kde=True)
    plt.title(col)
plt.tight_layout()
plt.savefig('outputs/feature_distributions.png')
print('Saved feature distributions to outputs/feature_distributions.png')

print('Done.')
