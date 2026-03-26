"""
Домашнее задание: Сравнение RandomForest и XGBoost
Датасет: Датасет: Titanic (Kaggle Competition)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import xgboost as xgb
import warnings

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-whitegrid')

def load_data():
    df = pd.read_csv('train.csv')
    df = df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)
    df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
    df['Embarked'] = df['Embarked'].fillna('S').map({'S': 0, 'C': 1, 'Q': 2})
    df['Age'] = df['Age'].fillna(df['Age'].median())
    df['Fare'] = df['Fare'].fillna(df['Fare'].median())
    return df

def evaluate_model(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    return {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'cv_mean': cross_val_score(model, X_train, y_train, cv=5).mean()
    }

def experiment_max_depth(X_train, X_test, y_train, y_test):
    print("\n" + "="*60)
    print("ЭКСПЕРИМЕНТ 1: ВЛИЯНИЕ max_depth")
    print("="*60)

    depths = [3, 5, 7, 10, 15, 20, None]
    rf_results = []
    xgb_results = []

    print("\nmax_depth |  RF Accuracy  | XGB Accuracy")
    print("-" * 45)

    for depth in depths:
        depth_str = str(depth) if depth else 'None'

        rf = RandomForestClassifier(n_estimators=100, max_depth=depth, random_state=42)
        rf_scores = evaluate_model(rf, X_train, X_test, y_train, y_test)
        rf_results.append(rf_scores['accuracy'])

        xgb_model = xgb.XGBClassifier(n_estimators=100, max_depth=depth if depth else 3,
                                       random_state=42, use_label_encoder=False,
                                       eval_metric='logloss')
        xgb_scores = evaluate_model(xgb_model, X_train, X_test, y_train, y_test)
        xgb_results.append(xgb_scores['accuracy'])

        print(f"   {depth_str:>4}    |    {rf_scores['accuracy']:.4f}     |    {xgb_scores['accuracy']:.4f}")

    plt.figure(figsize=(10, 6))
    plt.plot([str(d) if d else 'None' for d in depths], rf_results, 'o-',
             label='Random Forest', linewidth=2, markersize=8)
    plt.plot([str(d) if d else 'None' for d in depths], xgb_results, 's-',
             label='XGBoost', linewidth=2, markersize=8)
    plt.xlabel('max_depth', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.title('Влияние max_depth на качество модели', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig('experiment_max_depth.png', dpi=150)
    plt.show()

    return depths[np.argmax(rf_results)], depths[np.argmax(xgb_results)]

def experiment_n_estimators(X_train, X_test, y_train, y_test):
    print("\n" + "="*60)
    print("ЭКСПЕРИМЕНТ 2: ВЛИЯНИЕ n_estimators")
    print("="*60)

    n_estimators_list = [10, 25, 50, 100, 150, 200, 300, 500]
    rf_results = []
    xgb_results = []

    print("\nn_estimators |  RF Accuracy  | XGB Accuracy")
    print("-" * 45)

    for n in n_estimators_list:
        rf = RandomForestClassifier(n_estimators=n, max_depth=10, random_state=42)
        rf_scores = evaluate_model(rf, X_train, X_test, y_train, y_test)
        rf_results.append(rf_scores['accuracy'])

        xgb_model = xgb.XGBClassifier(n_estimators=n, max_depth=5,
                                       random_state=42, use_label_encoder=False,
                                       eval_metric='logloss')
        xgb_scores = evaluate_model(xgb_model, X_train, X_test, y_train, y_test)
        xgb_results.append(xgb_scores['accuracy'])

        print(f"    {n:>4}     |    {rf_scores['accuracy']:.4f}     |    {xgb_scores['accuracy']:.4f}")

    plt.figure(figsize=(10, 6))
    plt.plot(n_estimators_list, rf_results, 'o-', label='Random Forest', linewidth=2, markersize=8)
    plt.plot(n_estimators_list, xgb_results, 's-', label='XGBoost', linewidth=2, markersize=8)
    plt.xlabel('n_estimators', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.title('Влияние n_estimators на качество модели', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('experiment_n_estimators.png', dpi=150)
    plt.show()

    return n_estimators_list[np.argmax(rf_results)], n_estimators_list[np.argmax(xgb_results)]

def final_comparison(X_train, X_test, y_train, y_test):
    print("\n" + "="*60)
    print("ФИНАЛЬНОЕ СРАВНЕНИЕ ЛУЧШИХ МОДЕЛЕЙ")
    print("="*60)

    rf_best = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
    xgb_best = xgb.XGBClassifier(n_estimators=200, max_depth=5, random_state=42,
                                  use_label_encoder=False, eval_metric='logloss')

    rf_results = evaluate_model(rf_best, X_train, X_test, y_train, y_test)
    xgb_results = evaluate_model(xgb_best, X_train, X_test, y_train, y_test)

    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'CV Mean']

    print("\nМетрика      | Random Forest | XGBoost")
    print("-" * 50)
    print(f"Accuracy     |    {rf_results['accuracy']:.4f}     |   {xgb_results['accuracy']:.4f}")
    print(f"Precision    |    {rf_results['precision']:.4f}     |   {xgb_results['precision']:.4f}")
    print(f"Recall       |    {rf_results['recall']:.4f}     |   {xgb_results['recall']:.4f}")
    print(f"F1-Score     |    {rf_results['f1']:.4f}     |   {xgb_results['f1']:.4f}")
    print(f"CV Mean      |    {rf_results['cv_mean']:.4f}     |   {xgb_results['cv_mean']:.4f}")

    rf_values = [rf_results['accuracy'], rf_results['precision'],
                 rf_results['recall'], rf_results['f1'], rf_results['cv_mean']]
    xgb_values = [xgb_results['accuracy'], xgb_results['precision'],
                  xgb_results['recall'], xgb_results['f1'], xgb_results['cv_mean']]

    x = np.arange(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width/2, rf_values, width, label='Random Forest', color='steelblue')
    bars2 = ax.bar(x + width/2, xgb_values, width, label='XGBoost', color='coral')

    for bar, val in zip(bars1, rf_values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
               f'{val:.3f}', ha='center', va='bottom', fontsize=9)
    for bar, val in zip(bars2, xgb_values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
               f'{val:.3f}', ha='center', va='bottom', fontsize=9)

    ax.set_xlabel('Метрика', fontsize=12)
    ax.set_ylabel('Значение', fontsize=12)
    ax.set_title('Сравнение Random Forest vs XGBoost', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()
    ax.set_ylim(0, 1.1)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('final_comparison.png', dpi=150)
    plt.show()

    rf_importance = pd.DataFrame({
        'Feature': X_train.columns,
        'Importance': rf_best.feature_importances_
    }).sort_values('Importance', ascending=False)

    xgb_importance = pd.DataFrame({
        'Feature': X_train.columns,
        'Importance': xgb_best.feature_importances_
    }).sort_values('Importance', ascending=False)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].barh(rf_importance['Feature'], rf_importance['Importance'], color='steelblue')
    axes[0].set_xlabel('Importance')
    axes[0].set_title('Random Forest - Feature Importance')
    axes[0].invert_yaxis()

    axes[1].barh(xgb_importance['Feature'], xgb_importance['Importance'], color='coral')
    axes[1].set_xlabel('Importance')
    axes[1].set_title('XGBoost - Feature Importance')
    axes[1].invert_yaxis()

    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=150)
    plt.show()

    winner = 'Random Forest' if rf_results['accuracy'] > xgb_results['accuracy'] else 'XGBoost'
    print(f"\n>>> Победитель: {winner}")

    return rf_results, xgb_results

def main():
    print("="*60)
    print("ДОМАШНЕЕ ЗАДАНИЕ: Сравнение Random Forest и XGBoost")
    print("Датасет: Titanic (Kaggle Competition)")
    print("="*60)

    print("\nЗагрузка данных...")
    df = load_data()

    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")

    X = df.drop('Survived', axis=1)
    y = df['Survived']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"\nX_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"\nTarget distribution:\n{y.value_counts()}")

    best_depth_rf, best_depth_xgb = experiment_max_depth(X_train, X_test, y_train, y_test)
    best_n_rf, best_n_xgb = experiment_n_estimators(X_train, X_test, y_train, y_test)

    final_comparison(X_train, X_test, y_train, y_test)

    print("\n" + "="*60)
    print("ВЫВОДЫ")
    print("="*60)
    print("""
    1. max_depth:
       - При слишком маленькой глубине (<5) модели недообучаются
       - При большой глубине (>15) возможно переобучение
       - Оптимально: RF ~10, XGBoost ~5

    2. n_estimators:
       - Больше деревьев = стабильнее результат
       - После ~100-200 деревьев прирост качества замедляется
       - RF обычно требует больше деревьев чем XGBoost

    3. Сравнение моделей:
       - XGBoost часто показывает лучшие результаты за счет бустинга
       - Random Forest более устойчив к переобучению
       - Важность признаков может отличаться
    """)

    print("\nЭксперимент завершен!")
    print("Сохранены графики: experiment_max_depth.png, experiment_n_estimators.png, final_comparison.png, feature_importance.png")

if __name__ == "__main__":
    main()