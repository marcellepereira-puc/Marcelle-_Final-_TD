"""Visualização t-SNE dos embeddings da rede Siamese."""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.manifold import TSNE

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "utils"))

from config import (
    CLASS_NAMES,
    EXPERIMENTS_SIAMESE_DIR,
    EXPERIMENTS_SIAMESE_IMAGENET_DIR,
)

from evaluate_siamese import _coletar_teste, _ler_imagem, prototipos_por_classe

DATASETS = {
    "01": "covid-chestxray-dataset-master",
    "02": "CXR8",
    "03": "Imbalanced-Tuberculosis",
    "04": "makedataset",
    "05": "all",
}

CORES = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]


def _subamostrar(
    arquivos: list[str],
    labels: list[int],
    max_por_classe: int | None,
) -> tuple[list[str], list[int]]:
    if not max_por_classe or max_por_classe <= 0:
        return arquivos, labels

    por_classe: dict[int, list[str]] = {i: [] for i in range(len(CLASS_NAMES))}
    for caminho, label in zip(arquivos, labels):
        if len(por_classe[label]) < max_por_classe:
            por_classe[label].append(caminho)

    novos_arquivos = []
    novos_labels = []
    for label, lista in por_classe.items():
        novos_arquivos.extend(lista)
        novos_labels.extend([label] * len(lista))
    return novos_arquivos, novos_labels


def _extrair_embeddings(rede, arquivos: list[str]) -> np.ndarray:
    embeddings = []
    for caminho in arquivos:
        imagem = np.expand_dims(_ler_imagem(caminho), axis=0)
        embeddings.append(rede.predict(imagem, verbose=0)[0])
    return np.array(embeddings)


def _perplexidade(n_amostras: int) -> float:
    if n_amostras < 2:
        return 1.0
    return float(min(30, max(5, n_amostras // 4)))


def plot_tsne_siamese(
    rede,
    numero_pair: str,
    nome_dataset: str,
    shot: int,
    prefixo: str,
    pasta_saida: Path,
    max_por_classe: int | None = 4,
    mostrar_prototipos: bool = True,
) -> Path | None:
    """
    Gera scatter t-SNE (2D) dos embeddings do conjunto de teste.

    Retorna o caminho do PNG ou None se não houver dados suficientes.
    """
    pasta_saida = Path(pasta_saida)
    pasta_saida.mkdir(parents=True, exist_ok=True)

    arquivos, labels = _coletar_teste(nome_dataset)
    if not arquivos:
        print(f"[aviso] t-SNE: teste não encontrado para {nome_dataset}")
        return None

    arquivos, labels = _subamostrar(arquivos, labels, max_por_classe)
    labels = np.array(labels)

    print(f"\nGerando t-SNE ({shot}-shot)...")
    print(f"Imagens: {len(arquivos)} (max {max_por_classe or 'todas'}/classe)")

    embeddings = _extrair_embeddings(rede, arquivos)
    n_amostras = len(embeddings)

    if n_amostras < 2:
        print("[aviso] t-SNE: poucas amostras (mínimo 2).")
        return None

    vetores_extra = []
    labels_extra = []
    eh_prototipo = [False] * n_amostras

    if mostrar_prototipos:
        for classe, vetor in prototipos_por_classe(rede, numero_pair, shot).items():
            if classe in CLASS_NAMES:
                vetores_extra.append(vetor)
                labels_extra.append(CLASS_NAMES.index(classe))
                eh_prototipo.append(True)

    todos = np.vstack([embeddings, *vetores_extra]) if vetores_extra else embeddings
    todos_labels = np.concatenate([labels, np.array(labels_extra)]) if vetores_extra else labels
    todos_proto = np.array(eh_prototipo)

    coords = TSNE(
        n_components=2,
        random_state=42,
        perplexity=_perplexidade(len(todos)),
        init="pca",
        learning_rate="auto",
    ).fit_transform(todos)

    fig, ax = plt.subplots(figsize=(10, 8))

    for indice, classe in enumerate(CLASS_NAMES):
        mascara = (todos_labels == indice) & (~todos_proto)
        if not np.any(mascara):
            continue
        ax.scatter(
            coords[mascara, 0],
            coords[mascara, 1],
            c=CORES[indice % len(CORES)],
            label=classe,
            alpha=0.55,
            s=28,
            edgecolors="white",
            linewidths=0.3,
        )

    if mostrar_prototipos and np.any(todos_proto):
        for indice, classe in enumerate(CLASS_NAMES):
            mascara = (todos_labels == indice) & todos_proto
            if not np.any(mascara):
                continue
            ax.scatter(
                coords[mascara, 0],
                coords[mascara, 1],
                c=CORES[indice % len(CORES)],
                marker="*",
                s=320,
                edgecolors="black",
                linewidths=0.8,
                zorder=5,
                label="_nolegend_",
            )

    ax.set(
        title=f"Embeddings Siamese — t-SNE ({numero_pair}, {shot}-shot)\n"
        f"4 classes, até {max_por_classe or 'todas'} imagens/classe; ★ = protótipo",
        xlabel="t-SNE 1",
        ylabel="t-SNE 2",
    )
    ax.legend(loc="best", framealpha=0.9)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()

    caminho = pasta_saida / f"{prefixo}_tsne.png"
    fig.savefig(caminho, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Gráfico t-SNE salvo em: {caminho}")
    return caminho


def _importar_modulo_siamese():
    import importlib.util

    caminho = ROOT / "src" / "base_model" / "model.Siamese.py"
    spec = importlib.util.spec_from_file_location("model_siamese", caminho)
    modulo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modulo)
    return modulo


def _carregar_rede(pasta: Path, nome_modelo: str, imagenet: bool):
    caminho_pesos = pasta / f"{nome_modelo}.melhor_pesos.weights.h5"
    caminho_keras = pasta / f"{nome_modelo}.embedding.keras"

    if caminho_pesos.exists():
        modulo = _importar_modulo_siamese()
        rede = modulo.criar_modelo_embedding(weights="imagenet" if imagenet else None)
        rede.load_weights(caminho_pesos)
        return rede

    if caminho_keras.exists():
        return tf.keras.models.load_model(caminho_keras)

    raise FileNotFoundError(
        f"Nenhum checkpoint encontrado em {pasta} "
        f"({caminho_pesos.name} ou {caminho_keras.name})"
    )


def main():
    parser = argparse.ArgumentParser(description="t-SNE dos embeddings Siamese")
    parser.add_argument("--exp", default="04", help="Experimento 01-05")
    parser.add_argument("--shot", type=int, default=1, choices=[1, 3, 5])
    parser.add_argument(
        "--imagenet",
        action="store_true",
        help="Usar pasta results/04_Siamese_imagenet",
    )
    parser.add_argument(
        "--modelo",
        help="Caminho para .melhor_pesos.weights.h5 ou .embedding.keras",
    )
    parser.add_argument(
        "--max-por-classe",
        type=int,
        default=4,
        help="Máximo de imagens por classe — 4 classes no projeto (0 = todas)",
    )
    parser.add_argument(
        "--sem-prototipos",
        action="store_true",
        help="Não marcar protótipos das âncoras no gráfico",
    )
    args = parser.parse_args()

    numero = args.exp.zfill(2)
    if numero not in DATASETS:
        raise SystemExit(f"Experimento inválido: {args.exp}")

    nome_dataset = DATASETS[numero]
    pasta_resultados = (
        EXPERIMENTS_SIAMESE_IMAGENET_DIR if args.imagenet else EXPERIMENTS_SIAMESE_DIR
    )
    pasta = pasta_resultados / numero / f"{args.shot}-shot"
    nome_modelo = (
        "siamese_resnet50v2_imagenet" if args.imagenet else "siamese_resnet50v2"
    )
    prefixo = f"{nome_modelo}_{numero}_{args.shot}shot"

    if args.modelo:
        caminho = Path(args.modelo)
        if caminho.suffix == ".h5" or caminho.name.endswith(".weights.h5"):
            modulo = _importar_modulo_siamese()
            rede = modulo.criar_modelo_embedding(weights="imagenet" if args.imagenet else None)
            rede.load_weights(caminho)
        else:
            rede = tf.keras.models.load_model(caminho)
    else:
        rede = _carregar_rede(pasta, nome_modelo, args.imagenet)
    max_classe = args.max_por_classe if args.max_por_classe > 0 else None

    plot_tsne_siamese(
        rede,
        numero,
        nome_dataset,
        args.shot,
        prefixo,
        pasta,
        max_por_classe=max_classe,
        mostrar_prototipos=not args.sem_prototipos,
    )


if __name__ == "__main__":
    main()
