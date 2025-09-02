"""Train and compare multiple classifiers, visualize EDA and metrics, and save the best model.

This script:
- Loads the dataset from data/students.csv
- Preprocesses (impute + scale features, encode labels)
- Trains multiple models and reports metrics (accuracy, precision, recall, F1)
- Optionally performs light hyperparameter tuning for top models
- Saves: best model artifacts, confusion matrix, correlation heatmap, attendance vs performance plot,
  model comparison bars, ROC curves, and a CSV of per-model metrics.
"""
import os
import argparse
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder, label_binarize
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    auc,
)

# Optional XGBoost (if installed)
try:
    from xgboost import XGBClassifier  # type: ignore
    HAS_XGB = True
except Exception:
    HAS_XGB = False

DATA_PATH = 'data/students.csv'
os.makedirs('models', exist_ok=True)
os.makedirs('outputs', exist_ok=True)

parser = argparse.ArgumentParser(description='Train and evaluate student performance classifiers.')
parser.add_argument('--data', default=DATA_PATH, help='Path to CSV dataset')
parser.add_argument('--test-size', type=float, default=0.25, help='Test split size fraction')
parser.add_argument('--seed', type=int, default=42, help='Random seed')
parser.add_argument('--tune', action='store_true', help='Enable light hyperparameter tuning for top models')
args = parser.parse_args()

print('Loading data...')
df = pd.read_csv(args.data)
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
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_enc, test_size=args.test_size, random_state=args.seed, stratify=y_enc
)

models = {
    'LogisticRegression': LogisticRegression(max_iter=1000, multi_class='auto', random_state=args.seed),
    'DecisionTree': DecisionTreeClassifier(random_state=args.seed),
    'RandomForest': RandomForestClassifier(n_estimators=200, random_state=args.seed),
    'KNN': KNeighborsClassifier(n_neighbors=7),
    'SVC': SVC(probability=True, random_state=args.seed),
}
if HAS_XGB:
    models['XGBoost'] = XGBClassifier(
        n_estimators=300,
        learning_rate=0.1,
        max_depth=4,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        objective='multi:softprob',
        random_state=args.seed,
        eval_metric='mlogloss',
        n_jobs=-1,
    )

results = {}
for name, model in models.items():
    print(f'Training {name}...')
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds, average='macro', zero_division=0)
    rec = recall_score(y_test, preds, average='macro', zero_division=0)
    f1 = f1_score(y_test, preds, average='macro', zero_division=0)
    print(f'{name}: accuracy={acc:.4f}, f1_macro={f1:.4f}')
    results[name] = {'model': model, 'accuracy': acc, 'precision_macro': prec, 'recall_macro': rec, 'f1_macro': f1}

# Cross-val for best model choice
print('\nCross-validation (5-fold) mean accuracies:')
for name, info in results.items():
    scores = cross_val_score(info['model'], X_scaled, y_enc, cv=5)
    print(name, scores.mean())
    results[name]['cv_mean'] = scores.mean()

# Optional light hyperparameter tuning for top-2 by CV
def get_param_grid(model_name):
    if model_name == 'RandomForest':
        return {
            'n_estimators': [200, 400],
            'max_depth': [None, 6, 10],
            'min_samples_split': [2, 5],
        }
    if model_name == 'SVC':
        return {
            'C': [0.5, 1.0, 2.0],
            'gamma': ['scale', 'auto'],
            'kernel': ['rbf', 'linear'],
        }
    if model_name == 'KNN':
        return {'n_neighbors': [5, 7, 9], 'weights': ['uniform', 'distance']}
    if model_name == 'LogisticRegression':
        return {'C': [0.5, 1.0, 2.0], 'penalty': ['l2'], 'solver': ['lbfgs']}
    if model_name == 'XGBoost' and HAS_XGB:
        return {
            'max_depth': [3, 4, 5],
            'learning_rate': [0.05, 0.1],
            'subsample': [0.8, 1.0],
            'colsample_bytree': [0.8, 1.0],
        }
    return None

if args.tune:
    print('\nHyperparameter tuning (light grid) for top models...')
    top2 = sorted(results.items(), key=lambda kv: kv[1]['cv_mean'], reverse=True)[:2]
    for name, info in top2:
        grid = get_param_grid(name)
        if not grid:
            print(f'- Skipping tuning for {name} (no grid)')
            continue
        base = models[name]
        search = GridSearchCV(base, grid, cv=5, n_jobs=-1)
        search.fit(X_train, y_train)
        tuned = search.best_estimator_
        preds = tuned.predict(X_test)
        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds, average='macro', zero_division=0)
        rec = recall_score(y_test, preds, average='macro', zero_division=0)
        f1 = f1_score(y_test, preds, average='macro', zero_division=0)
        print(f'- {name} tuned: best_params={search.best_params_}, accuracy={acc:.4f}, f1_macro={f1:.4f}')
        results[name] = {'model': tuned, 'accuracy': acc, 'precision_macro': prec, 'recall_macro': rec, 'f1_macro': f1, 'cv_mean': results[name]['cv_mean']}

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
plt.title(f'Confusion Matrix ({best_name})')
plt.tight_layout()
plt.savefig('outputs/confusion_matrix.png')
print('Saved confusion matrix to outputs/confusion_matrix.png')

# Save per-model metrics CSV and comparison bars
metrics_df = pd.DataFrame([
    {
        'model': name,
        'accuracy': info['accuracy'],
        'precision_macro': info['precision_macro'],
        'recall_macro': info['recall_macro'],
        'f1_macro': info['f1_macro'],
        'cv_mean_acc': info['cv_mean'],
    }
    for name, info in results.items()
]).sort_values('f1_macro', ascending=False)
metrics_df.to_csv('outputs/model_metrics.csv', index=False)
print('Saved per-model metrics to outputs/model_metrics.csv')

plt.figure(figsize=(7,4))
sns.barplot(data=metrics_df, x='model', y='accuracy', palette='viridis')
plt.title('Model Accuracy (Test)')
plt.ylim(0, 1)
plt.tight_layout()
plt.savefig('outputs/model_comparison_accuracy.png')
print('Saved model accuracy comparison to outputs/model_comparison_accuracy.png')

plt.figure(figsize=(7,4))
sns.barplot(data=metrics_df, x='model', y='f1_macro', palette='magma')
plt.title('Model F1 (macro, Test)')
plt.ylim(0, 1)
plt.tight_layout()
plt.savefig('outputs/model_comparison_f1.png')
print('Saved model F1 comparison to outputs/model_comparison_f1.png')

# Feature distributions (EDA)
plt.figure(figsize=(8,6))
for i, col in enumerate(features, 1):
    plt.subplot(2,2,i)
    sns.histplot(df[col].dropna(), kde=True)
    plt.title(col)
plt.tight_layout()
plt.savefig('outputs/feature_distributions.png')
print('Saved feature distributions to outputs/feature_distributions.png')

# Correlation heatmap
plt.figure(figsize=(6,5))
corr = df[features].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Feature Correlation Heatmap')
plt.tight_layout()
plt.savefig('outputs/correlation_heatmap.png')
print('Saved correlation heatmap to outputs/correlation_heatmap.png')

# Attendance vs Performance trend (boxplot)
plt.figure(figsize=(6,4))
sns.boxplot(data=df, x='Category', y='Attendance', order=list(le.classes_))
plt.title('Attendance vs Performance (Category)')
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig('outputs/attendance_vs_performance.png')
print('Saved attendance vs performance to outputs/attendance_vs_performance.png')

# ROC Curves (multiclass OvR) for best model if probabilities available
classes = np.arange(len(le.classes_))
if hasattr(best_model, 'predict_proba') or hasattr(best_model, 'decision_function'):
    if hasattr(best_model, 'predict_proba'):
        y_score = best_model.predict_proba(X_test)
    else:
        # decision_function may return scores; scale via a softmax-like normalization for plotting AUC approximation
        scores = best_model.decision_function(X_test)
        # Ensure 2D
        if scores.ndim == 1:
            scores = np.vstack([1 - scores, scores]).T
        e_x = np.exp(scores - np.max(scores, axis=1, keepdims=True))
        y_score = e_x / e_x.sum(axis=1, keepdims=True)
    y_test_bin = label_binarize(y_test, classes=classes)
    plt.figure(figsize=(7,5))
    for i, label in enumerate(le.classes_):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_score[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f'{label} (AUC={roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--', lw=1)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curves (OvR) - {best_name}')
    plt.legend(loc='lower right', fontsize='small')
    plt.tight_layout()
    plt.savefig('outputs/roc_curves.png')
    print('Saved ROC curves to outputs/roc_curves.png')
else:
    print('Skipping ROC curves (model does not provide probability or decision scores).')

print('Done.')
