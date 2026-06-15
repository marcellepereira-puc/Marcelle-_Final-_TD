import random
import sys
from pathlib import Path

import tensorflow as tf

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "utils"))

from config import (
    BATCH_SIZE,
    EXTENSOES,
    SIAMESE_BATCH_SIZE,
    SIAMESE_VARIANTS_PER_ANCHOR,
    SPLIT_DIR,
)
from image_loader import mapear_imagem_tf
from siamese_transforms import augmentar_treino_tf

PAIR_DIR = ROOT / "datasets" / "pair"
SHOTS = [1, 3, 5]


def _listar_imagens(pasta: Path) -> list[str]:
    """Lista caminhos de imagens dentro de uma pasta."""
    if not pasta.is_dir():
        return []

    imagens = []
    for arquivo in pasta.iterdir():
        if arquivo.is_file() and arquivo.suffix.lower() in EXTENSOES:
            imagens.append(str(arquivo.resolve()))
    return imagens


def _listar_imagens_recursivo(pasta: Path) -> list[str]:
    if not pasta.is_dir():
        return []
    return [
        str(arquivo.resolve())
        for arquivo in pasta.rglob("*")
        if arquivo.is_file() and arquivo.suffix.lower() in EXTENSOES
    ]


def _pool_positivas(
    positivas_pair: list[str],
    nome_dataset: str | None,
    nome_classe: str,
) -> list[str]:
    """Une positivas do pair (test) com imagens do train — mais variedade no treino."""
    pool = list(positivas_pair)
    if nome_dataset:
        pasta_treino = SPLIT_DIR / nome_dataset / "train" / nome_classe
        pool.extend(_listar_imagens_recursivo(pasta_treino))
    return list(dict.fromkeys(pool))


def pair_disponivel(numero_experimento: str, shot: int) -> bool:
    """Verifica se ancora, positiva e negativa existem para o experimento."""
    pasta_pair = PAIR_DIR / numero_experimento
    pasta_shot = f"{shot}-shot"

    pastas = (
        pasta_pair / "ancora" / pasta_shot,
        pasta_pair / "positiva" / pasta_shot,
        pasta_pair / "negativa" / pasta_shot,
    )
    return PAIR_DIR.exists() and all(pasta.is_dir() for pasta in pastas)


def montar_tripletas(
    numero_experimento: str,
    shot: int,
    nome_dataset: str | None = None,
    variantes_por_ancora: int = SIAMESE_VARIANTS_PER_ANCHOR,
) -> list[tuple[str, str, str]]:
    """
    Monta tripletas (ancora, positiva, negativa).

    Ancoras continuam do pair (few-shot). Positivas podem vir tambem do train.
    """
    pasta_pair = PAIR_DIR / numero_experimento
    pasta_shot = f"{shot}-shot"

    pasta_ancora = pasta_pair / "ancora" / pasta_shot
    pasta_positiva = pasta_pair / "positiva" / pasta_shot
    pasta_negativa = pasta_pair / "negativa" / pasta_shot

    tripletas = []

    for pasta_classe in sorted(pasta_ancora.iterdir()):
        if not pasta_classe.is_dir():
            continue

        nome_classe = pasta_classe.name
        ancoras = _listar_imagens(pasta_classe)
        positivas = _pool_positivas(
            _listar_imagens(pasta_positiva / nome_classe),
            nome_dataset,
            nome_classe,
        )
        negativas = _listar_imagens(pasta_negativa / nome_classe)

        if not ancoras or not positivas or not negativas:
            continue

        for caminho_ancora in ancoras:
            for _ in range(max(1, variantes_por_ancora)):
                caminho_positiva = random.choice(positivas)
                caminho_negativa = random.choice(negativas)
                tripletas.append((caminho_ancora, caminho_positiva, caminho_negativa))

    return tripletas


def contar_ancoras(numero_experimento: str, shot: int) -> int:
    pasta_ancora = PAIR_DIR / numero_experimento / "ancora" / f"{shot}-shot"
    if not pasta_ancora.is_dir():
        return 0
    return sum(len(_listar_imagens(pasta)) for pasta in pasta_ancora.iterdir() if pasta.is_dir())


def _ler_imagem(caminho):
    return mapear_imagem_tf(caminho)


def _ler_tripleta_treino(caminho_ancora, caminho_positiva, caminho_negativa):
    ancora = augmentar_treino_tf(_ler_imagem(caminho_ancora))
    positiva = augmentar_treino_tf(_ler_imagem(caminho_positiva))
    negativa = augmentar_treino_tf(_ler_imagem(caminho_negativa))
    y_dummy = tf.zeros((2,), dtype=tf.float32)
    return (ancora, positiva, negativa), y_dummy


def _ler_tripleta_val(caminho_ancora, caminho_positiva, caminho_negativa):
    ancora = _ler_imagem(caminho_ancora)
    positiva = _ler_imagem(caminho_positiva)
    negativa = _ler_imagem(caminho_negativa)
    y_dummy = tf.zeros((2,), dtype=tf.float32)
    return (ancora, positiva, negativa), y_dummy


def _batch_size_siamese(qtd_tripletas: int, treino: bool) -> int:
    if qtd_tripletas <= 0:
        return 1
    alvo = SIAMESE_BATCH_SIZE if treino else min(BATCH_SIZE, qtd_tripletas)
    return max(1, min(alvo, qtd_tripletas))


def carregar_tripletas(
    numero_experimento: str,
    shot: int,
    treino=True,
    nome_dataset: str | None = None,
):
    """
    Carrega tripletas de datasets/pair/XX/.

    treino=True  -> 80% das tripletas, augmentacao e batch menor
    treino=False -> 20% restante para validacao
    """
    tripletas = montar_tripletas(numero_experimento, shot, nome_dataset=nome_dataset)

    if not tripletas:
        raise FileNotFoundError(
            f"Nenhuma tripleta encontrada em {PAIR_DIR / numero_experimento / f'{shot}-shot'}"
        )

    random.shuffle(tripletas)

    corte = int(len(tripletas) * 0.8)
    if treino:
        tripletas = tripletas[:corte] if corte > 0 else tripletas
    else:
        tripletas = tripletas[corte:] if corte < len(tripletas) else tripletas[-1:]

    ancoras, positivas, negativas = zip(*tripletas)
    qtd = len(tripletas)
    batch_size = _batch_size_siamese(qtd, treino)

    dataset = tf.data.Dataset.from_tensor_slices((list(ancoras), list(positivas), list(negativas)))
    if treino:
        dataset = dataset.shuffle(min(qtd, 4096), reshuffle_each_iteration=True)

    map_fn = _ler_tripleta_treino if treino else _ler_tripleta_val
    dataset = (
        dataset.map(map_fn, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )
    return dataset, qtd, batch_size
