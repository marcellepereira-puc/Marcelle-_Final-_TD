from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import label_binarize

import matplotlib.pyplot as plt
import numpy as np

# Função para calcular as métricas de avaliação
def metrics_evaluation(y_true, y_pred, class_names=None):
    accuracy = accuracy_score(y_true, y_pred)

    precision_weighted = precision_score(
        y_true, y_pred, average="weighted", zero_division=0
    )
    recall_weighted = recall_score(
        y_true, y_pred, average="weighted", zero_division=0
    )
    f1_weighted = f1_score(
        y_true, y_pred, average="weighted", zero_division=0
    )

    precision_macro = precision_score(
        y_true, y_pred, average="macro", zero_division=0
    )
    recall_macro = recall_score(
        y_true, y_pred, average="macro", zero_division=0
    )
    f1_macro = f1_score(
        y_true, y_pred, average="macro", zero_division=0
    )

    confusion = confusion_matrix(y_true, y_pred)

    metrics_json = {
        "accuracy": float(accuracy),
        "precision_weighted": float(precision_weighted),
        "recall_weighted": float(recall_weighted),
        "f1_weighted": float(f1_weighted),
        "precision_macro": float(precision_macro),
        "recall_macro": float(recall_macro),
        "f1_macro": float(f1_macro),
        "confusion_matrix": confusion.tolist(),
    }
    if class_names:
        metrics_json["class_names"] = list(class_names)

    # Imprimir as métricas de avaliação
    print("\n=== Teste ===")
    print(f"Acurácia:             {accuracy:.4f}")
    print(f"Precisão (weighted):  {precision_weighted:.4f}")
    print(f"Recall (weighted):    {recall_weighted:.4f}")
    print(f"F1 (weighted):        {f1_weighted:.4f}")

    return metrics_json


def roc_auc_metrics(y_true, y_score, class_names):
    """
    AUC one-vs-rest para cada classe e AUC macro (4 classes no projeto).
    y_score: matriz (n_amostras, n_classes) — maior = mais provável.
    """
    n_classes = len(class_names)
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)

    y_bin = label_binarize(y_true, classes=list(range(n_classes)))
    if y_bin.shape[1] == 1 and n_classes == 2:
        y_bin = np.hstack([1 - y_bin, y_bin])

    auc_por_classe = {}
    curvas = {}

    for indice, nome in enumerate(class_names):
        coluna = y_bin[:, indice]
        if len(np.unique(coluna)) < 2:
            continue
        try:
            fpr, tpr, _ = roc_curve(coluna, y_score[:, indice])
            valor_auc = float(auc(fpr, tpr))
            auc_por_classe[nome] = valor_auc
            curvas[nome] = {"fpr": fpr.tolist(), "tpr": tpr.tolist(), "auc": valor_auc}
        except ValueError:
            continue

    valores = list(auc_por_classe.values())
    auc_macro = float(np.mean(valores)) if valores else None

    try:
        auc_ovr = float(
            roc_auc_score(
                y_bin,
                y_score,
                multi_class="ovr",
                average="macro",
                labels=list(range(n_classes)),
            )
        )
    except ValueError:
        auc_ovr = auc_macro

    resultado = {
        "auc_macro": auc_macro,
        "auc_ovr": auc_ovr,
        "auc_por_classe": auc_por_classe,
    }

    print("\n=== ROC / AUC (one-vs-rest) ===")
    for nome, valor in auc_por_classe.items():
        print(f"AUC — {nome}: {valor:.4f}")
    if auc_macro is not None:
        print(f"AUC macro: {auc_macro:.4f}")

    return resultado, curvas
