import sys
from pathlib import Path

import numpy as np
import tensorflow as tf

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "utils"))

from config import BATCH_SIZE, CLASS_NAMES, EXTENSOES, IMAGE_SIZE, SPLIT_DIR
from image_loader import mapear_imagem_tf

SPLIT_DIR = SPLIT_DIR.resolve()


def _garantir_split(pasta: Path) -> Path:
    pasta = pasta.resolve()
    try:
        pasta.relative_to(SPLIT_DIR)
    except ValueError as exc:
        raise ValueError(
            f"Caminho fora do split: {pasta}\n"
            f"O treino usa apenas imagens de: {SPLIT_DIR}"
        ) from exc
    return pasta


def _mapear_label(nome_pasta: str, nome_dataset: str) -> str | None:
    if nome_pasta in CLASS_NAMES:
        return nome_pasta

    nome = nome_pasta.strip().lower()

    if nome_dataset == "covid-chestxray-dataset-master" and nome == "pneumonia":
        return "Convid"
    if "covid" in nome or "convid" in nome:
        return "Convid"
    if nome in {"no finding", "normal"}:
        return "No Finding"
    if "tuberculosis" in nome:
        return "Tuberculosis"
    if "pneumonia" in nome:
        return "Pneumonia"

    return None


def split_disponivel(dataset=None, subset="train") -> bool:
    if not SPLIT_DIR.exists():
        return False
    if dataset:
        return (SPLIT_DIR / dataset / subset).is_dir()
    return any((p / subset).is_dir() for p in SPLIT_DIR.iterdir() if p.is_dir())


def experimento_split_disponivel(dataset_train, dataset_val, dataset_test) -> bool:
    pastas = (
        SPLIT_DIR / dataset_train / "train",
        SPLIT_DIR / dataset_val / "val",
        SPLIT_DIR / dataset_test / "test",
    )
    return SPLIT_DIR.exists() and all(pasta.is_dir() for pasta in pastas)


def _pastas_dataset(dataset=None) -> list[Path]:
    if not SPLIT_DIR.exists():
        raise FileNotFoundError(f"Split não encontrado: {SPLIT_DIR}")

    if dataset:
        pasta = _garantir_split(SPLIT_DIR / dataset)
        if not pasta.is_dir():
            raise FileNotFoundError(f"Dataset não encontrado no split: {pasta}")
        return [pasta]

    return [_garantir_split(p) for p in SPLIT_DIR.iterdir() if p.is_dir()]


def _coletar_imagens(subset="train", dataset=None) -> tuple[list[str], list[int], dict[str, int]]:
    """Coleta todas as imagens do subset (train/val/test) do repositório informado."""
    arquivos: list[str] = []
    labels: list[int] = []
    por_classe = {classe: 0 for classe in CLASS_NAMES}

    for pasta_dataset in _pastas_dataset(dataset):
        pasta = _garantir_split(pasta_dataset / subset)
        if not pasta.is_dir():
            continue

        nome_dataset = pasta_dataset.name
        for label_dir in pasta.iterdir():
            if not label_dir.is_dir():
                continue

            label = _mapear_label(label_dir.name, nome_dataset)
            if label is None:
                continue

            label_id = CLASS_NAMES.index(label)
            for arquivo in label_dir.rglob("*"):
                if not arquivo.is_file() or arquivo.suffix.lower() not in EXTENSOES:
                    continue

                arquivos.append(str(_garantir_split(arquivo)))
                labels.append(label_id)
                por_classe[label] += 1

    return arquivos, labels, por_classe


def contar_imagens(subset="train", dataset=None) -> tuple[int, dict[str, int]]:
    arquivos, _, por_classe = _coletar_imagens(subset, dataset)
    return len(arquivos), por_classe


def carregar_dados(subset="train", dataset=None):
    """Carrega todas as imagens do subset do repositório em datasets/split."""
    arquivos, labels, _ = _coletar_imagens(subset, dataset)

    if not arquivos:
        alvo = f"{dataset}/{subset}" if dataset else f"*/{subset}"
        raise FileNotFoundError(f"Nenhuma imagem em {SPLIT_DIR}/{alvo}")

    ds = tf.data.Dataset.from_tensor_slices((arquivos, labels))
    if subset == "train":
        ds = ds.shuffle(len(arquivos))
    return (
        ds.map(mapear_imagem_tf, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(BATCH_SIZE)
        .prefetch(tf.data.AUTOTUNE)
    )


def prever(model, dataset):
    y_true = []
    y_pred = []

    for imagens, labels in dataset:
        preds = model.predict(imagens, verbose=0).argmax(axis=1)
        y_true.extend(labels.numpy())
        y_pred.extend(preds)

    return np.array(y_true), np.array(y_pred)
