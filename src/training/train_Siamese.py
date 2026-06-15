import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import tensorflow as tf

ROOT = Path(__file__).resolve().parents[2]
BASE_MODEL_DIR = ROOT / "src" / "base_model"

sys.path.insert(0, str(ROOT / "src" / "utils"))
sys.path.insert(0, str(ROOT / "src" / "evaluation"))
sys.path.insert(0, str(BASE_MODEL_DIR))
sys.path.insert(0, str(ROOT / "src" / "training"))

from config import (
    CLASS_NAMES,
    EXPERIMENTS_SIAMESE_DIR,
    IMAGE_SIZE,
    SIAMESE_EPOCHS,
    SIAMESE_FREEZE_LAYERS,
    SIAMESE_LEARNING_RATE,
    SIAMESE_TRIPLET_MARGIN,
)
from dataset_loader_siamese import (
    PAIR_DIR,
    SHOTS,
    carregar_tripletas,
    contar_ancoras,
    pair_disponivel,
)
from evaluate_siamese import avaliar_classificacao, salvar_grafico_treino
from plot_embeddings_tsne import plot_tsne_siamese
from gpu_config import configurar_gpu


DIM_EMBEDDING = 128


def _carregar_modulo_siamese():
    """Importa model.Siamese.py (nome com ponto no arquivo)."""
    caminho = BASE_MODEL_DIR / "model.Siamese.py"
    spec = importlib.util.spec_from_file_location("model_siamese", caminho)
    modulo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modulo)
    return modulo


def formatar_tempo(segundos: float) -> str:
    horas = int(segundos // 3600)
    minutos = int((segundos % 3600) // 60)
    segs = int(segundos % 60)
    if horas:
        return f"{horas}h {minutos}m {segs}s"
    if minutos:
        return f"{minutos}m {segs}s"
    return f"{segs}s"


def _caminho_pesos_embedding(nome_modelo: str, pasta_saida: Path) -> Path:
    return pasta_saida / f"{nome_modelo}.melhor_pesos.weights.h5"


class _SalvarMelhorPesos(tf.keras.callbacks.Callback):
    """Salva somente os pesos do embedding quando val_loss melhora.

    Evita gravar dois .keras ao mesmo tempo (OOM/errno 22 no Windows com h5py).
    """

    def __init__(self, rede, caminho: Path):
        super().__init__()
        self.rede = rede
        self.caminho = Path(caminho)
        self.melhor_loss = float("inf")

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        val_loss = logs.get("val_loss")
        if val_loss is None or val_loss >= self.melhor_loss:
            return

        self.melhor_loss = val_loss
        self.caminho.parent.mkdir(parents=True, exist_ok=True)
        self.rede.save_weights(str(self.caminho))
        print(
            f"\nEpoch {epoch + 1}: val_loss improved to {val_loss:.5f}, "
            f"saving model to {self.caminho}"
        )


def _carregar_melhor_embedding(rede, nome_modelo: str, pasta_saida: Path):
    """Recarrega o melhor embedding (pesos ou artefato legado .keras)."""
    caminho_pesos = _caminho_pesos_embedding(nome_modelo, pasta_saida)
    caminho_embedding = pasta_saida / f"{nome_modelo}.embedding.keras"

    if caminho_pesos.exists():
        rede.load_weights(caminho_pesos)
        return rede, caminho_pesos

    if caminho_embedding.exists():
        return tf.keras.models.load_model(caminho_embedding), caminho_embedding

    return rede, None


def avaliar_tripletas(modelo, dataset) -> float:
    """
    Mede quantas vezes a distancia positiva ficou menor que a negativa.
    Quanto maior, melhor.
    """
    acertos = 0
    total = 0

    for entradas, _ in dataset:
        ancora, positiva, negativa = entradas
        predicoes = modelo.predict([ancora, positiva, negativa], verbose=0)
        dist_positiva = predicoes[:, 0]
        dist_negativa = predicoes[:, 1]
        acertos += int(np.sum(dist_positiva < dist_negativa))
        total += len(dist_positiva)

    if total == 0:
        return 0.0
    return round(acertos / total, 4)


def treinar_experimento(numero_pair, nome_dataset, shot, modulo_siamese, nome_modelo, pasta_base):
    if not pair_disponivel(numero_pair, shot):
        print(f"\n[aviso] Pair nao encontrado: {PAIR_DIR / numero_pair / f'{shot}-shot'}")
        print("Execute: python src/data/ger_pair.py")
        return None

    pasta_saida = pasta_base / numero_pair / f"{shot}-shot"
    pasta_saida.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 50}")
    print(f"Experimento {numero_pair} — {nome_dataset} — {shot}-shot")
    print(f"Fonte: {PAIR_DIR / numero_pair}")
    print(f"Saida: {pasta_saida}")
    print(f"{'=' * 50}\n")

    train_ds, qtd_train, batch_train = carregar_tripletas(
        numero_pair, shot, treino=True, nome_dataset=nome_dataset
    )
    val_ds, qtd_val, batch_val = carregar_tripletas(
        numero_pair, shot, treino=False, nome_dataset=nome_dataset
    )
    qtd_ancoras = contar_ancoras(numero_pair, shot)
    steps_treino = max(1, (qtd_train + batch_train - 1) // batch_train)

    print(f"Ancoras: {qtd_ancoras}")
    print(f"Tripletas treino: {qtd_train} (batch {batch_train}, ~{steps_treino} passos/epoca)")
    print(f"Tripletas validacao: {qtd_val} (batch {batch_val})\n")

    modelo, rede = modulo_siamese.criar_modelo_siamese(
        input_shape=(*IMAGE_SIZE, 3),
        dim_embedding=DIM_EMBEDDING,
        freeze_layers=SIAMESE_FREEZE_LAYERS,
    )
    modulo_siamese.compilar_modelo(
        modelo,
        learning_rate=SIAMESE_LEARNING_RATE,
        margem=SIAMESE_TRIPLET_MARGIN,
    )

    caminho_pesos = _caminho_pesos_embedding(nome_modelo, pasta_saida)
    caminho_tempo = pasta_saida / f"{nome_modelo}_tempo.json"

    inicio = time.time()
    historico = modelo.fit(
        train_ds,
        validation_data=val_ds,
        epochs=SIAMESE_EPOCHS,
        callbacks=[
            _SalvarMelhorPesos(rede, caminho_pesos),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=8,
                min_lr=1e-6,
                verbose=1,
            ),
        ],
    )
    duracao = time.time() - inicio

    print(f"\nTempo de treino: {formatar_tempo(duracao)}")

    rede, caminho_carregado = _carregar_melhor_embedding(rede, nome_modelo, pasta_saida)

    acuracia_tripletas = avaliar_tripletas(modelo, val_ds)
    prefixo = f"{nome_modelo}_{numero_pair}_{shot}shot"
    salvar_grafico_treino(historico, prefixo, pasta_saida)

    metricas_teste = avaliar_classificacao(
        rede,
        numero_pair,
        nome_dataset,
        shot,
        prefixo,
        pasta_saida,
    )

    plot_tsne_siamese(
        rede,
        numero_pair,
        nome_dataset,
        shot,
        prefixo,
        pasta_saida,
        max_por_classe=4,
    )

    info = {
        "modelo": nome_modelo,
        "experimento": numero_pair,
        "dataset": nome_dataset,
        "shot": shot,
        "ancoras": qtd_ancoras,
        "tripletas_treino": qtd_train,
        "tripletas_validacao": qtd_val,
        "batch_treino": batch_train,
        "passos_por_epoca": steps_treino,
        "tempo_segundos": round(duracao, 2),
        "tempo_formatado": formatar_tempo(duracao),
        "epochs": SIAMESE_EPOCHS,
        "acuracia_tripletas_val": acuracia_tripletas,
        "metricas_teste_few_shot": metricas_teste,
        "melhor_pesos": str(caminho_pesos),
        "modelo_carregado": str(caminho_carregado) if caminho_carregado else None,
        "classes": CLASS_NAMES,
    }

    with open(caminho_tempo, "w", encoding="utf-8") as arquivo:
        json.dump(info, arquivo, indent=2, ensure_ascii=False)

    print(f"Acuracia tripletas (val): {acuracia_tripletas:.2%}")
    if metricas_teste:
        print(f"Acuracia few-shot (test): {metricas_teste['accuracy']:.2%}")
    print(f"Resumo salvo em: {caminho_tempo}")

    return modelo, historico, info


def avaliar_experimento(numero_pair, nome_dataset, shot, modulo_siamese, nome_modelo, pasta_base):
    """Apenas few-shot no test + t-SNE (requer pesos ja treinados)."""
    pasta_saida = Path(pasta_base) / numero_pair / f"{shot}-shot"
    caminho_pesos = _caminho_pesos_embedding(nome_modelo, pasta_saida)

    if not caminho_pesos.exists():
        print(f"\n[aviso] Pesos nao encontrados — pulando teste: {caminho_pesos}")
        return None

    print(f"\n{'=' * 50}")
    print(f"Teste few-shot {numero_pair} — {nome_dataset} — {shot}-shot")
    print(f"Pesos: {caminho_pesos}")
    print(f"{'=' * 50}")

    rede = modulo_siamese.criar_modelo_embedding(
        input_shape=(*IMAGE_SIZE, 3),
        dim_embedding=DIM_EMBEDDING,
        freeze_layers=SIAMESE_FREEZE_LAYERS,
    )
    rede.load_weights(caminho_pesos)

    prefixo = f"{nome_modelo}_{numero_pair}_{shot}shot"
    metricas = avaliar_classificacao(
        rede,
        numero_pair,
        nome_dataset,
        shot,
        prefixo,
        pasta_saida,
    )
    plot_tsne_siamese(
        rede,
        numero_pair,
        nome_dataset,
        shot,
        prefixo,
        pasta_saida,
        max_por_classe=4,
    )

    if metricas:
        print(f"Acuracia few-shot (test): {metricas['accuracy']:.2%}")
    return metricas


def rodar_avaliacoes(experimentos, nome_modelo, pasta_base, shots=None, titulo="Siamese"):
    """Executa o teste few-shot nos experimentos 01--05 (sem retreinar)."""
    if not PAIR_DIR.exists():
        print(f"[aviso] Pasta pair nao encontrada: {PAIR_DIR}")
        return

    if not configurar_gpu():
        print("Nenhum teste sera executado sem GPU.")
        return

    shots = shots or SHOTS
    pasta_base = Path(pasta_base)
    modulo_siamese = _carregar_modulo_siamese()
    inicio_total = time.time()

    for numero_pair, nome_dataset in experimentos:
        for shot in shots:
            avaliar_experimento(
                numero_pair,
                nome_dataset,
                shot,
                modulo_siamese,
                nome_modelo,
                pasta_base,
            )

    print(f"\nTempo total — testes {titulo}: {formatar_tempo(time.time() - inicio_total)}")


def rodar_experimentos(experimentos, nome_modelo, pasta_base, shots=None, titulo="Siamese"):
    if not PAIR_DIR.exists():
        print(f"[aviso] Pasta pair nao encontrada: {PAIR_DIR}")
        print("Execute: python src/data/ger_pair.py")
        return

    if not configurar_gpu():
        print("Nenhum experimento sera executado sem GPU.")
        return

    shots = shots or SHOTS
    pasta_base = Path(pasta_base)
    modulo_siamese = _carregar_modulo_siamese()

    print(f"Usando somente imagens de: {PAIR_DIR}\n")
    inicio_total = time.time()

    for numero_pair, nome_dataset in experimentos:
        for shot in shots:
            treinar_experimento(
                numero_pair,
                nome_dataset,
                shot,
                modulo_siamese,
                nome_modelo,
                pasta_base,
            )

    print(f"\nTempo total — {titulo}: {formatar_tempo(time.time() - inicio_total)}")


def treinar_siamese():
    """Atalho para rodar os experimentos padrao."""
    experimentos = [
        ("01", "covid-chestxray-dataset-master"),
        ("02", "CXR8"),
        ("03", "Imbalanced-Tuberculosis"),
        ("04", "makedataset"),
        ("05", "all"),
    ]
    rodar_experimentos(
        experimentos,
        "siamese_resnet50v2",
        EXPERIMENTS_SIAMESE_DIR,
        SHOTS,
        "Siamese ResNet50v2",
    )


if __name__ == "__main__":
    treinar_siamese()
