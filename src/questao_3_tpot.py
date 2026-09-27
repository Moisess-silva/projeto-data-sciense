"""Questão 3 — AutoML com TPOT no diagnóstico de câncer de mama."""

from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


RANDOM_STATE = 42
ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "assets"
OUTPUTS = ROOT / "outputs"


def save_accuracy_chart(tpot_accuracy, baseline_accuracy):
    plt.figure(figsize=(9, 6))
    bars = plt.bar(["Baseline\n(Regressão Logística)", "TPOT\n(AutoML)"],
                   [baseline_accuracy, tpot_accuracy],
                   color=["#4C78A8", "#F58518"], width=0.55)
    plt.ylim(0, 1.08)
    plt.ylabel("Acurácia no conjunto de teste")
    plt.title("TPOT vs. Baseline — diagnóstico automatizado", fontsize=16, weight="bold", pad=14)
    plt.grid(axis="y", alpha=0.25)
    for bar in bars:
        value = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, value + 0.02,
                 f"{value:.1%}", ha="center", va="bottom", fontsize=13, weight="bold")
    plt.tight_layout()
    plt.savefig(ASSETS / "q3_tpot_vs_baseline.png", dpi=300, bbox_inches="tight")
    plt.close()


def save_pipeline_diagram(pipeline_text):
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.axis("off")
    steps = ["Dados\nde treino", "Pré-processamento\nselecionado pelo TPOT", "Modelo final\nselecionado pelo TPOT", "Predição\ndo diagnóstico"]
    x_positions = [0.08, 0.33, 0.63, 0.88]
    colors = ["#D9EAF7", "#B6D7A8", "#F9CB9C", "#D9D2E9"]
    for x, label, color in zip(x_positions, steps, colors):
        ax.text(x, 0.52, label, ha="center", va="center", fontsize=13, weight="bold",
                bbox=dict(boxstyle="round,pad=0.8", facecolor=color, edgecolor="#555555", linewidth=1.2),
                transform=ax.transAxes)
    for left, right in zip(x_positions[:-1], x_positions[1:]):
        ax.annotate("", xy=(right - 0.09, 0.52), xytext=(left + 0.09, 0.52),
                    xycoords=ax.transAxes, textcoords=ax.transAxes,
                    arrowprops=dict(arrowstyle="->", lw=2, color="#555555"))
    ax.text(0.5, 0.12, "Pipeline exportado pelo TPOT:\n" + pipeline_text[:180],
            ha="center", va="center", fontsize=10, color="#333333",
            bbox=dict(boxstyle="round,pad=0.6", facecolor="#F7F7F7", edgecolor="#CCCCCC"),
            transform=ax.transAxes)
    ax.set_title("Fluxo simplificado do pipeline vencedor", fontsize=16, weight="bold", pad=14)
    plt.tight_layout()
    plt.savefig(ASSETS / "q3_pipeline_diagram.png", dpi=300, bbox_inches="tight")
    plt.close()


def main():
    from tpot import TPOTClassifier

    ASSETS.mkdir(parents=True, exist_ok=True)
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    dataset = load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(
        dataset.data, dataset.target, test_size=0.20,
        random_state=RANDOM_STATE, stratify=dataset.target,
    )

    baseline = Pipeline([
        ("padronizacao", StandardScaler()),
        ("regressao_logistica", LogisticRegression(max_iter=2000)),
    ])
    baseline.fit(X_train, y_train)
    baseline_accuracy = accuracy_score(y_test, baseline.predict(X_test))

    tpot = TPOTClassifier(
        generations=5, population_size=20, cv=5,
        random_state=RANDOM_STATE, verbosity=2,
        n_jobs=1, max_time_mins=None,
    )
    tpot.fit(X_train, y_train)
    tpot_accuracy = accuracy_score(y_test, tpot.predict(X_test))
    exported_path = OUTPUTS / "q3_tpot_best_pipeline.py"
    tpot.export(str(exported_path))
    pipeline_text = str(tpot.fitted_pipeline_)

    save_accuracy_chart(tpot_accuracy, baseline_accuracy)
    save_pipeline_diagram(pipeline_text)
    print(f"Baseline: {baseline_accuracy:.4f} ({baseline_accuracy:.1%})")
    print(f"TPOT: {tpot_accuracy:.4f} ({tpot_accuracy:.1%})")
    print(f"Pipeline selecionado: {pipeline_text}")
    print(f"Pipeline exportado em: {exported_path}")


if __name__ == "__main__":
    main()

