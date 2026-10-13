"""Questão 2 — comparação de modelos Voting e Stacking para crédito."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import StackingClassifier


RANDOM_STATE = 42
ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "assets"
OUTPUTS = ROOT / "outputs"


def save_metrics_chart(results: pd.DataFrame) -> None:
    """Salva um gráfico comparando acurácia e F1-score."""
    ASSETS.mkdir(parents=True, exist_ok=True)

    chart_data = results.set_index("Modelo")[["Acurácia", "F1-Score"]]
    chart_data.plot(kind="bar", figsize=(11, 6), color=["#17365D", "#E31B32"])

    plt.title("Comparação dos modelos de classificação de crédito")
    plt.ylabel("Pontuação")
    plt.ylim(0, 1)
    plt.xticks(rotation=20, ha="right")
    plt.grid(axis="y", alpha=0.25)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(ASSETS / "q2_metrics_comparison.png", dpi=300)
    plt.close()


def main() -> None:
    """Treina os modelos, compara métricas e salva os resultados."""
    ASSETS.mkdir(parents=True, exist_ok=True)
    OUTPUTS.mkdir(parents=True, exist_ok=True)

    # Base sintética para permitir executar o exemplo sem um CSV externo.
    X, y = make_classification(
        n_samples=1200,
        n_features=12,
        n_informative=8,
        n_redundant=2,
        weights=[0.65, 0.35],
        flip_y=0.03,
        random_state=RANDOM_STATE,
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    models = {
        "Regressão Logística": LogisticRegression(max_iter=2000),
        "Árvore de Decisão": DecisionTreeClassifier(
            max_depth=6,
            random_state=RANDOM_STATE,
        ),
        "SVM": SVC(probability=True, random_state=RANDOM_STATE),
    }

    for model in models.values():
        model.fit(X_train, y_train)

    voting = VotingClassifier(
        estimators=[
            ("logistic", LogisticRegression(max_iter=2000)),
            ("tree", DecisionTreeClassifier(
                max_depth=6,
                random_state=RANDOM_STATE,
            )),
            ("svm", SVC(probability=True, random_state=RANDOM_STATE)),
        ],
        voting="hard",
    )

    stacking = StackingClassifier(
        estimators=[
            ("logistic", LogisticRegression(max_iter=2000)),
            ("tree", DecisionTreeClassifier(
                max_depth=6,
                random_state=RANDOM_STATE,
            )),
            ("svm", SVC(probability=True, random_state=RANDOM_STATE)),
        ],
        final_estimator=LogisticRegression(max_iter=2000),
        cv=5,
    )

    models["Voting"] = voting
    models["Stacking"] = stacking

    rows = []

    for name, model in models.items():
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        rows.append({
            "Modelo": name,
            "Acurácia": accuracy_score(y_test, predictions),
            "Precision": precision_score(
                y_test, predictions, zero_division=0
            ),
            "Recall": recall_score(
                y_test, predictions, zero_division=0
            ),
            "F1-Score": f1_score(
                y_test, predictions, zero_division=0
            ),
        })

    results = pd.DataFrame(rows)
    results.to_csv(
        OUTPUTS / "q2_results.csv",
        index=False,
        encoding="utf-8-sig",
    )

    save_metrics_chart(results)

    print("\nResultados da classificação de crédito:")
    print(results.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"\nTabela salva em: {OUTPUTS / 'q2_results.csv'}")
    print(f"Gráfico salvo em: {ASSETS / 'q2_metrics_comparison.png'}")


if __name__ == "__main__":
    main()