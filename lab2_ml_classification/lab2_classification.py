"""
Lab 2: Machine Learning Classification
Dataset: Heart Failure Prediction
Methods: Random Forest, XGBoost, Gradient Boosting
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (classification_report, confusion_matrix,
                             accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, roc_curve)
import xgboost as xgb
import warnings
import kagglehub
import os

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-whitegrid')

def load_and_preprocess():
    path = kagglehub.dataset_download("fedesoriano/heart-failure-prediction")
    files = os.listdir(path)
    csv_file = [f for f in files if f.endswith('.csv')][0]
    df = pd.read_csv(os.path.join(path, csv_file))

    le = LabelEncoder()
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = le.fit_transform(df[col])

    X = df.drop('HeartDisease', axis=1)
    y = df['HeartDisease']

    return X, y, df

def evaluate_model(model, X_train, X_test, y_train, y_test, name):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    results = {
        'name': name,
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_pred_proba),
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba
    }

    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
    results['cv_mean'] = cv_scores.mean()
    results['cv_std'] = cv_scores.std()

    return results

def plot_confusion_matrices(results_list):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    for idx, result in enumerate(results_list):
        cm = confusion_matrix(y_test, result['y_pred'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                   xticklabels=['No Disease', 'Disease'],
                   yticklabels=['No Disease', 'Disease'])
        axes[idx].set_title(f"{result['name']}\nAccuracy: {result['accuracy']:.3f}")
        axes[idx].set_xlabel('Predicted')
        axes[idx].set_ylabel('Actual')

    plt.tight_layout()
    plt.savefig('confusion_matrices.png', dpi=150)
    plt.show()

def plot_roc_curves(results_list):
    plt.figure(figsize=(10, 8))

    for result in results_list:
        fpr, tpr, _ = roc_curve(y_test, result['y_pred_proba'])
        plt.plot(fpr, tpr, linewidth=2,
                label=f"{result['name']} (AUC = {result['roc_auc']:.3f})")

    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curves Comparison', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('roc_curves.png', dpi=150)
    plt.show()

def plot_feature_importance(models, model_names, feature_names):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for idx, (model, name) in enumerate(zip(models, model_names)):
        importance = model.feature_importances_
        indices = np.argsort(importance)[::-1]

        axes[idx].barh(range(len(importance)), importance[indices], color='steelblue')
        axes[idx].set_yticks(range(len(importance)))
        axes[idx].set_yticklabels([feature_names[i] for i in indices])
        axes[idx].set_xlabel('Feature Importance')
        axes[idx].set_title(f'{name} Feature Importance')
        axes[idx].invert_yaxis()

    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=150)
    plt.show()

def plot_metrics_comparison(results_list):
    metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    x = np.arange(len(metrics))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 6))

    for idx, result in enumerate(results_list):
        values = [result[m] for m in metrics]
        bars = ax.bar(x + idx * width, values, width, label=result['name'])
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{val:.3f}', ha='center', va='bottom', fontsize=8)

    ax.set_xlabel('Metrics', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width)
    ax.set_xticklabels(metrics)
    ax.legend()
    ax.set_ylim(0, 1.15)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('metrics_comparison.png', dpi=150)
    plt.show()

def plot_learning_curves(X, y):
    train_sizes = np.linspace(0.1, 0.95, 10)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    model_names = ['Random Forest', 'XGBoost', 'Gradient Boosting']
    models = [rf_model, xgb_model, gb_model]

    for idx, (model, name) in enumerate(zip(models, model_names)):
        train_scores, test_scores = [], []

        for size in train_sizes:
            X_train, X_test_split, y_train, y_test_split = train_test_split(
                X, y, train_size=float(size), random_state=42, stratify=y
            )
            model.fit(X_train, y_train)
            train_scores.append(model.score(X_train, y_train))
            test_scores.append(model.score(X_test_split, y_test_split))

        train_sizes_abs = [int(s * len(X)) for s in train_sizes]

        axes[idx].plot(train_sizes_abs, train_scores, 'o-', color='blue', label='Training score')
        axes[idx].plot(train_sizes_abs, test_scores, 'o-', color='orange', label='Test score')
        axes[idx].set_xlabel('Training Set Size')
        axes[idx].set_ylabel('Accuracy')
        axes[idx].set_title(f'{name} Learning Curve')
        axes[idx].legend()
        axes[idx].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('learning_curves.png', dpi=150)
    plt.show()

def print_detailed_results(results_list):
    print("\n" + "="*80)
    print("DETAILED CLASSIFICATION RESULTS")
    print("="*80)

    for result in results_list:
        print(f"\n{'='*40}")
        print(f"Model: {result['name']}")
        print(f"{'='*40}")
        print(f"\nPerformance Metrics:")
        print(f"  Accuracy:  {result['accuracy']:.4f}")
        print(f"  Precision: {result['precision']:.4f}")
        print(f"  Recall:    {result['recall']:.4f}")
        print(f"  F1-Score:  {result['f1']:.4f}")
        print(f"  ROC-AUC:   {result['roc_auc']:.4f}")
        print(f"\nCross-Validation (5-fold):")
        print(f"  Mean: {result['cv_mean']:.4f} (+/- {result['cv_std']*2:.4f})")
        print(f"\nClassification Report:")
        print(classification_report(y_test, result['y_pred'],
                                   target_names=['No Disease', 'Disease']))

if __name__ == "__main__":
    print("Loading Heart Failure Prediction dataset...")
    X, y, df = load_and_preprocess()

    print(f"\nDataset shape: {X.shape}")
    print(f"Target distribution:\n{y.value_counts()}")
    print(f"\nFeatures: {list(X.columns)}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("\n" + "="*60)
    print("TRAINING MODELS")
    print("="*60)

    rf_model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    print("\n1. Random Forest - Training...")

    xgb_model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss'
    )
    print("2. XGBoost - Training...")

    gb_model = GradientBoostingClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.8,
        random_state=42
    )
    print("3. Gradient Boosting - Training...")

    results_rf = evaluate_model(rf_model, X_train, X_test, y_train, y_test, 'Random Forest')
    results_xgb = evaluate_model(xgb_model, X_train, X_test, y_train, y_test, 'XGBoost')
    results_gb = evaluate_model(gb_model, X_train, X_test, y_train, y_test, 'Gradient Boosting')

    results_list = [results_rf, results_xgb, results_gb]

    plot_confusion_matrices(results_list)
    plot_roc_curves(results_list)
    plot_feature_importance([rf_model, xgb_model, gb_model],
                           ['Random Forest', 'XGBoost', 'Gradient Boosting'],
                           X.columns)
    plot_metrics_comparison(results_list)
    plot_learning_curves(X, y)

    print_detailed_results(results_list)

    summary_df = pd.DataFrame([{
        'Model': r['name'],
        'Accuracy': r['accuracy'],
        'Precision': r['precision'],
        'Recall': r['recall'],
        'F1': r['f1'],
        'ROC-AUC': r['roc_auc'],
        'CV Mean': r['cv_mean'],
        'CV Std': r['cv_std']
    } for r in results_list])

    print("\n" + "="*80)
    print("SUMMARY TABLE")
    print("="*80)
    print(summary_df.to_string(index=False))

    summary_df.to_csv('results_summary.csv', index=False)

    print("\n" + "="*60)
    print("Lab 2 completed! Check 'lab2_ml_classification/' for plots and results.")
    print("="*60)
