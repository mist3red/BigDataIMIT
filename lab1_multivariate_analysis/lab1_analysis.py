"""
Лабораторная работа 1: Многомерный статистический анализ
Набор данных: Прогнозирование сердечной недостаточности
Методы: PCA, факторный анализ, LDA, t-SNE, MDS, UMAP
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA, FactorAnalysis
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.manifold import TSNE, MDS
import umap
import warnings
import kagglehub
import os

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-whitegrid')

def load_data():
    path = kagglehub.dataset_download("fedesoriano/heart-failure-prediction")
    files = os.listdir(path)
    csv_file = [f for f in files if f.endswith('.csv')][0]
    df = pd.read_csv(os.path.join(path, csv_file))
    return df

def preprocess_data(df):
    le = LabelEncoder()
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        df[col] = le.fit_transform(df[col])

    X = df.drop('HeartDisease', axis=1)
    y = df['HeartDisease']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, df

def plot_comparison(results, titles, n_components=2):
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    colors = ['#FF6B6B', '#4ECDC4']

    for idx, (result, title) in enumerate(zip(results, titles)):
        ax = axes[idx]
        result = np.array(result)

        if result.ndim == 1 or result.shape[1] == 1:
            mask_0 = y == 0
            mask_1 = y == 1
            ax.scatter(result[mask_0], np.zeros(mask_0.sum()),
                      c=colors[0], label='Class 0', alpha=0.6, s=50)
            ax.scatter(result[mask_1], np.zeros(mask_1.sum()),
                      c=colors[1], label='Class 1', alpha=0.6, s=50)
            ax.set_xlabel('Component 1')
            ax.set_ylabel('')
            ax.set_yticks([])
        else:
            for i, label in enumerate([0, 1]):
                mask = y == label
                ax.scatter(result[mask, 0], result[mask, 1],
                          c=colors[i], label=f'Class {label}', alpha=0.6, s=50)
            ax.set_xlabel('Component 1')
            ax.set_ylabel('Component 2')

        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend()

    plt.tight_layout()
    plt.savefig('comparison.png', dpi=150)
    plt.show()

def pca_analysis(X_scaled, y):
    print("\n" + "="*60)
    print("1. МЕТОД ГЛАВНЫХ КОМПОНЕНТ (PCA)")
    print("="*60)

    pca = PCA()
    X_pca = pca.fit_transform(X_scaled)

    explained_var = pca.explained_variance_ratio_
    cumsum_var = np.cumsum(explained_var)

    print(f"\nДоля объясненной дисперсии (первые 10 компонент):")
    for i, (ev, cv) in enumerate(zip(explained_var[:10], cumsum_var[:10])):
        print(f"  PC{i+1}: {ev:.4f} (Cumulative: {cv:.4f})")

    print(f"\nКоличество компонент для 95% дисперсии: {np.argmax(cumsum_var >= 0.95) + 1}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].bar(range(1, len(explained_var)+1), explained_var, alpha=0.7, color='steelblue')
    axes[0].plot(range(1, len(explained_var)+1), explained_var, 'ro-', markersize=4)
    axes[0].set_xlabel('Principal Component')
    axes[0].set_ylabel('Explained Variance Ratio')
    axes[0].set_title('Scree Plot')

    axes[1].plot(range(1, len(cumsum_var)+1), cumsum_var, 'bo-', markersize=4)
    axes[1].axhline(y=0.95, color='r', linestyle='--', label='95% threshold')
    axes[1].set_xlabel('Number of Components')
    axes[1].set_ylabel('Cumulative Explained Variance')
    axes[1].set_title('Cumulative Variance Plot')
    axes[1].legend()

    plt.tight_layout()
    plt.savefig('pca_analysis.png', dpi=150)
    plt.show()

    loadings = pd.DataFrame(pca.components_.T,
                           columns=[f'PC{i+1}' for i in range(pca.n_components_)],
                           index=df.columns[:-1])
    print(f"\nНагрузки PCA (первые 5 компонент):")
    print(loadings.iloc[:, :5].round(3).to_string())

    return X_pca[:, :2]

def factor_analysis(X_scaled, df):
    print("\n" + "="*60)
    print("2. ФАКТОРНЫЙ АНАЛИЗ (FA)")
    print("="*60)

    n_factors_range = [2, 3, 4, 5]
    scores = []

    for n_factors in n_factors_range:
        fa = FactorAnalysis(n_components=n_factors, random_state=42)
        X_fa = fa.fit_transform(X_scaled)
        score = fa.score(X_scaled)
        scores.append(score)
        print(f"  Factors: {n_factors}, Log-likelihood: {score:.2f}")

    optimal_factors = n_factors_range[np.argmax(scores)]
    print(f"Оптимальное количество факторов: {optimal_factors}")

    fa = FactorAnalysis(n_components=optimal_factors, random_state=42)
    X_fa = fa.fit_transform(X_scaled)

    loadings = pd.DataFrame(fa.components_.T,
                           columns=[f'Factor{i+1}' for i in range(optimal_factors)],
                           index=df.columns[:-1])
    print(f"\nФакторные нагрузки:")
    print(loadings.round(3).to_string())

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for i, score in enumerate(scores):
        axes[0].bar(n_factors_range[i], score, alpha=0.7, color='teal')
    axes[0].set_xlabel('Number of Factors')
    axes[0].set_ylabel('Log-likelihood')
    axes[0].set_title('Factor Analysis Model Selection')
    # Добавить ограничение по оси Y
    min_score = min(scores)
    max_score = max(scores)
    # Установить границы: от минимального значения до минимального + 1
    # (или более узкий диапазон для лучшей детализации)
    axes[0].set_ylim(min_score - 0.1, min_score + 1.0)  # Например: от -14.5 до -13.5

    im = axes[1].imshow(loadings.values, cmap='RdBu_r', aspect='auto')
    axes[1].set_xticks(range(optimal_factors))
    axes[1].set_xticklabels([f'Factor{i+1}' for i in range(optimal_factors)])
    axes[1].set_yticks(range(len(loadings.index)))
    axes[1].set_yticklabels(loadings.index, fontsize=8)
    axes[1].set_title('Factor Loadings Heatmap')
    plt.colorbar(im, ax=axes[1])

    plt.tight_layout()
    plt.savefig('factor_analysis.png', dpi=150)
    plt.show()

    return X_fa[:, :2]

def lda_analysis(X_scaled, y):
    print("\n" + "="*60)
    print("3. ЛИНЕЙНЫЙ ДИСКРИМИНАНТНЫЙ АНАЛИЗ (LDA)")
    print("="*60)

    n_classes = len(np.unique(y))
    n_lda = min(n_classes - 1, X_scaled.shape[1])

    lda = LDA(n_components=n_lda)
    X_lda = lda.fit_transform(X_scaled, y)

    explained_var = lda.explained_variance_ratio_
    print(f"\nДоля объясненной дисперсии:")
    for i, ev in enumerate(explained_var):
        print(f"  LD{i+1}: {ev:.4f} ({ev*100:.2f}%)")

    print(f"\nКоэффициенты для LD1 (наиболее дискриминативные признаки):")
    coef_df = pd.DataFrame(lda.coef_,
                          columns=df.columns[:-1],
                          index=[f'LD{i+1}' for i in range(n_lda)])
    print(coef_df.T.sort_values('LD1', key=abs, ascending=False).head(10).round(3).to_string())

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    colors = ['#FF6B6B', '#4ECDC4']
    for i, label in enumerate([0, 1]):
        mask = y == label
        axes[0].scatter(X_lda[mask, 0], X_lda[mask, 1] if n_lda > 1 else np.zeros(sum(mask)),
                      c=colors[i], label=f'Class {label}', alpha=0.6, s=50)
    axes[0].set_xlabel('LD1')
    axes[0].set_ylabel('LD2' if n_lda > 1 else '')
    axes[0].set_title('LDA Projection')
    axes[0].legend()

    # ИСПРАВЛЕНИЕ: улучшенный график для LDA explained variance
    if n_lda == 1:
        # Для единственной компоненты используем горизонтальную линию или текст
        axes[1].bar(1, explained_var[0], alpha=0.7, color='purple', width=0.5)
        axes[1].set_xlim(0.5, 1.5)
        axes[1].set_xticks([1])
        axes[1].set_xticklabels(['LD1'])
        # Добавляем значение на график
        axes[1].text(1, explained_var[0]/2, f'{explained_var[0]*100:.1f}%',
                    ha='center', va='center', fontsize=12, fontweight='bold')
    else:
        # Для нескольких компонент используем обычный bar plot
        bars = axes[1].bar(range(1, n_lda+1), explained_var, alpha=0.7, color='purple')
        axes[1].set_xticks(range(1, n_lda+1))
        axes[1].set_xticklabels([f'LD{i}' for i in range(1, n_lda+1)])
        # Добавляем значения на столбцы
        for bar, val in zip(bars, explained_var):
            axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height()/2,
                        f'{val*100:.1f}%', ha='center', va='center', fontsize=10)

    axes[1].set_xlabel('Linear Discriminant')
    axes[1].set_ylabel('Explained Variance Ratio')
    axes[1].set_title('LDA Explained Variance')
    axes[1].set_ylim(0, 1.05)  # Фиксируем шкалу от 0 до 1.05
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('lda_analysis.png', dpi=150)
    plt.show()

    return X_lda[:, :2] if n_lda > 1 else X_lda.reshape(-1, 1)

def tsne_analysis(X_scaled, y):
    print("\n" + "="*60)
    print("4. t-SNE (стохастическое вложение соседей с t-распределением)")
    print("="*60)

    perplexities = [5, 15, 30, 50]

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    colors = ['#FF6B6B', '#4ECDC4']

    for idx, perplexity in enumerate(perplexities):
        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, n_iter=1000)
        X_tsne = tsne.fit_transform(X_scaled)

        ax = axes[idx // 2, idx % 2]
        for i, label in enumerate([0, 1]):
            mask = y == label
            ax.scatter(X_tsne[mask, 0], X_tsne[mask, 1],
                      c=colors[i], label=f'Class {label}', alpha=0.6, s=50)
        ax.set_title(f't-SNE (perplexity={perplexity})', fontsize=12, fontweight='bold')
        ax.set_xlabel('t-SNE 1')
        ax.set_ylabel('t-SNE 2')
        ax.legend()

    plt.tight_layout()
    plt.savefig('tsne_analysis.png', dpi=150)
    plt.show()

    tsne = TSNE(n_components=2, perplexity=30, random_state=42, n_iter=1000)
    X_tsne = tsne.fit_transform(X_scaled)

    return X_tsne

def mds_analysis(X_scaled, y):
    print("\n" + "="*60)
    print("5. MDS (Многомерное шкалирование)")
    print("="*60)

    mds = MDS(n_components=2, random_state=42, n_init=4, max_iter=300)
    X_mds = mds.fit_transform(X_scaled)

    print(f"\nСтресс: {mds.stress_:.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors = ['#FF6B6B', '#4ECDC4']

    for i, label in enumerate([0, 1]):
        mask = y == label
        axes[0].scatter(X_mds[mask, 0], X_mds[mask, 1],
                       c=colors[i], label=f'Class {label}', alpha=0.6, s=50)
    axes[0].set_xlabel('MDS 1')
    axes[0].set_ylabel('MDS 2')
    axes[0].set_title('MDS Projection')
    axes[0].legend()

    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=5)
    nn.fit(X_scaled)
    distances, indices = nn.kneighbors(X_scaled)

    dmds = mds.dissimilarity_matrix_ if hasattr(mds, 'dissimilarity_matrix_') else None

    axes[1].hist(distances[:, 1:].flatten(), bins=30, alpha=0.7, color='teal')
    axes[1].set_xlabel('Distance to 5 Nearest Neighbors')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Distance Distribution (MDS)')

    plt.tight_layout()
    plt.savefig('mds_analysis.png', dpi=150)
    plt.show()

    return X_mds

def umap_analysis(X_scaled, y):
    print("\n" + "="*60)
    print("6. UMAP (Равномерное приближение и проекция многообразия)")
    print("="*60)

    n_neighbors_list = [5, 15, 30]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    colors = ['#FF6B6B', '#4ECDC4']

    for idx, n_neighbors in enumerate(n_neighbors_list):
        reducer = umap.UMAP(n_components=2, n_neighbors=n_neighbors,
                           random_state=42, min_dist=0.1)
        X_umap = reducer.fit_transform(X_scaled)

        ax = axes[idx]
        for i, label in enumerate([0, 1]):
            mask = y == label
            ax.scatter(X_umap[mask, 0], X_umap[mask, 1],
                      c=colors[i], label=f'Class {label}', alpha=0.6, s=50)
        ax.set_title(f'UMAP (n_neighbors={n_neighbors})', fontsize=12, fontweight='bold')
        ax.set_xlabel('UMAP 1')
        ax.set_ylabel('UMAP 2')
        ax.legend()

    plt.tight_layout()
    plt.savefig('umap_analysis.png', dpi=150)
    plt.show()

    reducer = umap.UMAP(n_components=2, n_neighbors=15, random_state=42, min_dist=0.1)
    X_umap = reducer.fit_transform(X_scaled)

    print("\nСтатистика UMAP-вложения:")
    print(f"  Min UMAP 1: {X_umap[:, 0].min():.4f}")
    print(f"  Max UMAP 1: {X_umap[:, 0].max():.4f}")
    print(f"  Mean UMAP 1: {X_umap[:, 0].mean():.4f}")
    print(f"  Min UMAP 2: {X_umap[:, 1].min():.4f}")
    print(f"  Max UMAP 2: {X_umap[:, 1].max():.4f}")
    print(f"  Mean UMAP 2: {X_umap[:, 1].mean():.4f}")

    return X_umap

def print_summary():
    print("\n" + "="*60)
    print("\nИТОГ: СРАВНЕНИЕ МЕТОДОВ СНИЖЕНИЯ РАЗМЕРНОСТИ")
    print("="*60)
    print("""
    Method      | Type           | Preserves    | Speed  | Best For
    ------------|----------------|--------------|--------|-----------------------
    PCA         | Linear         | Variance     | Fast   | Linear patterns
    FA          | Linear         | Covariance   | Medium | Latent factors
    LDA         | Linear         | Class sep.   | Fast   | Supervised dim. red.
    t-SNE       | Non-linear     | Local struct.| Slow   | Cluster visualization
    MDS         | Non-linear     | Distances    | Slow   | Metric preservation
    UMAP        | Non-linear     | Local/global| Medium | Best overall quality

    Recommendations:
    - For variance-based reduction: PCA
    - For supervised classification: LDA
    - For cluster visualization: UMAP or t-SNE
    - For distance preservation: MDS or UMAP
    """)

if __name__ == "__main__":
    print("Загрузка набора данных для прогнозирования сердечной недостаточности...")
    df = load_data()
    print(f"Размер набора данных: {df.shape}")
    print(f"Столбцы: {list(df.columns)}")
    print(f"\nРаспределение классов:\n{df['HeartDisease'].value_counts()}")

    X_scaled, y, df = preprocess_data(df)

    X_pca = pca_analysis(X_scaled, y)
    X_fa = factor_analysis(X_scaled, df)
    X_lda = lda_analysis(X_scaled, y)
    X_tsne = tsne_analysis(X_scaled, y)
    X_mds = mds_analysis(X_scaled, y)
    X_umap = umap_analysis(X_scaled, y)

    results = [X_pca, X_fa, X_lda, X_tsne, X_mds, X_umap]
    titles = ['PCA', 'Factor Analysis', 'LDA', 't-SNE', 'MDS', 'UMAP']

    plot_comparison(results, titles)

    print_summary()

    print("\n" + "="*60)
    print("Lab 1 completed! Check 'lab1_multivariate_analysis/' for plots.")
    print("="*60)
