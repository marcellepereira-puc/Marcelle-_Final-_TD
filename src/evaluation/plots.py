import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix

from metrics import metrics_evaluation

ROOT = Path(__file__).resolve().parents[2]
SAIDA = ROOT / "results" / "plots"


def _history(history):
    return history if isinstance(history, dict) else history.history


def _salvar(fig, nome, saida_dir=None):
    pasta = Path(saida_dir) if saida_dir else SAIDA
    pasta.mkdir(parents=True, exist_ok=True)
    caminho = pasta / nome
    fig.savefig(caminho, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Gráfico salvo em: {caminho}")


def plot_training(history, prefix="modelo", saida_dir=None):
    h = _history(history)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(h["loss"], label="Treino")
    axes[0].plot(h["val_loss"], label="Validação")
    axes[0].set(title="Perda", xlabel="Época", ylabel="Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    acc = h.get("accuracy", h.get("acc", []))
    val_acc = h.get("val_accuracy", h.get("val_acc", []))
    axes[1].plot(acc, label="Treino")
    axes[1].plot(val_acc, label="Validação")
    axes[1].set(title="Acurácia", xlabel="Época", ylabel="Accuracy")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    _salvar(fig, f"{prefix}_training.png", saida_dir)


def plot_confusion_matrix(y_true, y_pred, class_names, prefix="modelo", saida_dir=None):
    cm = confusion_matrix(y_true, y_pred)
    total = cm.sum()
    vmax = cm.max() if cm.size else 1

    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(cm, cmap="Blues")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set(
        xticks=range(len(class_names)),
        yticks=range(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        xlabel="Predito",
        ylabel="Verdadeiro",
        title="Matriz de Confusão",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            valor = int(cm[i, j])
            pct = (valor / total * 100) if total else 0
            cor = "white" if cm[i, j] > vmax / 2 else "black"
            ax.text(
                j,
                i,
                f"{valor}\n({pct:.1f}%)",
                ha="center",
                va="center",
                color=cor,
                fontsize=9,
                fontweight="bold",
            )

    fig.tight_layout()
    _salvar(fig, f"{prefix}_confusion_matrix.png", saida_dir)
    return cm


def plot_roc_curves(curvas, class_names, prefix="modelo", saida_dir=None):
    """Curvas ROC por classe (one-vs-rest). curvas: dict nome -> {fpr, tpr, auc}."""
    if not curvas:
        return None

    fig, ax = plt.subplots(figsize=(8, 7))
    for nome in class_names:
        if nome not in curvas:
            continue
        dados = curvas[nome]
        ax.plot(
            dados["fpr"],
            dados["tpr"],
            label=f"{nome} (AUC={dados['auc']:.3f})",
            linewidth=2,
        )

    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Aleatório")
    ax.set(
        xlim=(0.0, 1.0),
        ylim=(0.0, 1.05),
        xlabel="Taxa de falso positivo (FPR)",
        ylabel="Taxa de verdadeiro positivo (TPR)",
        title="Curvas ROC — one-vs-rest",
    )
    ax.legend(loc="lower right", framealpha=0.9)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    _salvar(fig, f"{prefix}_roc.png", saida_dir)
    return fig


def evaluate_and_plot(
    y_true,
    y_pred,
    class_names,
    prefix="modelo",
    saida_dir=None,
    y_score=None,
):
    pasta = Path(saida_dir) if saida_dir else SAIDA
    pasta.mkdir(parents=True, exist_ok=True)

    metricas = metrics_evaluation(y_true, y_pred, class_names)
    plot_confusion_matrix(y_true, y_pred, class_names, prefix, pasta)

    if y_score is not None:
        from metrics import roc_auc_metrics

        roc_metricas, curvas = roc_auc_metrics(y_true, y_score, class_names)
        metricas.update(roc_metricas)
        plot_roc_curves(curvas, class_names, prefix, pasta)

    caminho = pasta / f"{prefix}_metrics.json"
    with open(caminho, "w", encoding="utf-8") as f:
        json.dump(metricas, f, indent=2, ensure_ascii=False)
    print(f"Métricas salvas em: {caminho}")

    return metricas
