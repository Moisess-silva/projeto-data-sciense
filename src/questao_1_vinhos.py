"""Questão 1 — classificação de vinhos com PCA, LDA e KNN.

O script treina os modelos apenas com o conjunto de treino, calcula a
acurácia no conjunto de teste e salva os gráficos em ../assets/.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.datasets import load_wine
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


RANDOM_STATE = 42
ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "assets"


def save_scatter(values, targets, title, filename, xlabel, ylabel):
    """Salva um gráfico 2D das três classes de vinho."""
    ASSETS.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 7))
    markers = ["o", "s", "X"]
    colors = ["#e15759", "#59a14f", "#4c78a8"]
    labels = ["Classe 0", "Classe 1", "Classe 2"]
    for cls, marker, color, label in zip(range(3), markers, colors, labels):
        mask = targets == cls
        plt.scatter(values[mask, 0], values[mask, 1], marker=marker, c=color,
                    label=label, s=85, alpha=0.9, edgecolors="white", linewidths=0.5)
    plt.title(title, fontsize=16, weight="bold", pad=14)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(title="Classe do vinho")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(ASSETS / filename, dpi=300, bbox_inches="tight")
    plt.close()


def main():
    wine = load_wine()
    X_train, X_test, y_train, y_test = train_test_split(
        wine.data,
        wine.target,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=wine.target,
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    X_all_scaled = scaler.transform(wine.data)

    lda = LinearDiscriminantAnalysis(n_components=2)
    X_train_lda = lda.fit_transform(X_train_scaled, y_train)
    X_test_lda = lda.transform(X_test_scaled)
    X_all_lda = lda.transform(X_all_scaled)

    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)
    X_all_pca = pca.transform(X_all_scaled)

    knn_lda = KNeighborsClassifier(n_neighbors=5)
    knn_lda.fit(X_train_lda, y_train)
    lda_accuracy = accuracy_score(y_test, knn_lda.predict(X_test_lda))

    knn_pca = KNeighborsClassifier(n_neighbors=5)
    knn_pca.fit(X_train_pca, y_train)
    pca_accuracy = accuracy_score(y_test, knn_pca.predict(X_test_pca))

    sns.set_theme(style="whitegrid", context="notebook")
    save_scatter(
        X_all_pca,
        wine.target,
        "PCA — projeção dos vinhos em 2 dimensões",
        "q1_pca_scatter.png",
        "Componente Principal 1",
        "Componente Principal 2",
    )
    save_scatter(
        X_all_lda,
        wine.target,
        "LDA — separação supervisionada das classes de vinho",
        "q1_lda_scatter.png",
        "Componente LDA 1",
        "Componente LDA 2",
    )

    plt.figure(figsize=(9, 6))
    labels = ["PCA", "LDA"]
    values = [pca_accuracy, lda_accuracy]
    bars = plt.bar(labels, values, color=["#4C78A8", "#F58518"], width=0.55)
    plt.ylim(0, 1.08)
    plt.ylabel("Acurácia no conjunto de teste")
    plt.title("KNN (k=5) — comparação de acurácia", fontsize=16, weight="bold", pad=14)
    plt.grid(axis="y", alpha=0.25)
    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width() / 2, value + 0.02, f"{value:.1%}",
                 ha="center", va="bottom", fontsize=13, weight="bold")
    plt.tight_layout()
    plt.savefig(ASSETS / "q1_accuracy_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Acurácia PCA: {pca_accuracy:.4f} ({pca_accuracy:.1%})")
    print(f"Acurácia LDA: {lda_accuracy:.4f} ({lda_accuracy:.1%})")
    print(f"Melhor método: {'LDA' if lda_accuracy > pca_accuracy else 'PCA' if pca_accuracy > lda_accuracy else 'empate'}")
    print(f"Gráficos salvos em: {ASSETS}")


if __name__ == "__main__":
    main()

