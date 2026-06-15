import json
import sys
import time
from pathlib import Path

import tensorflow as tf

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "utils"))
sys.path.insert(0, str(ROOT / "src" / "evaluation"))

from config import CLASS_NAMES, EPOCHS, SPLIT_DIR
from dataset_loader import carregar_dados, contar_imagens, experimento_split_disponivel, prever, split_disponivel
from gpu_config import configurar_gpu
from plots import evaluate_and_plot, plot_training

# Formata o tempo de treino.
def formatar_tempo(segundos: float) -> str:
    horas = int(segundos // 3600)
    minutos = int((segundos % 3600) // 60)
    segs = int(segundos % 60)
    if horas:
        return f"{horas}h {minutos}m {segs}s"
    if minutos:
        return f"{minutos}m {segs}s"
    return f"{segs}s"

# Compila o modelo.
def _compilar(model):
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model

# Callback para salvar o melhor modelo.
def _callback_melhor_modelo(caminho: Path) -> tf.keras.callbacks.ModelCheckpoint:
    return tf.keras.callbacks.ModelCheckpoint(
        filepath=str(caminho),
        monitor="val_loss",
        save_best_only=True,
        mode="min",
        verbose=1,
    )

# Treina o modelo e avalia.
def treinar_e_avaliar(
    model,
    train_ds,
    val_ds,
    test_ds,
    nome_modelo: str,
    pasta_saida: Path,
    class_names,
    epochs: int = EPOCHS,
):
    if not configurar_gpu():
        print("[aviso] Treino pulado — GPU não disponível.\n")
        return None

    pasta_saida = Path(pasta_saida)
    pasta_saida.mkdir(parents=True, exist_ok=True)

    caminho_modelo = pasta_saida / f"{nome_modelo}.melhor_modelo.pt"
    caminho_tempo = pasta_saida / f"{nome_modelo}_tempo.json"

    print(f"Saída dos resultados: {pasta_saida}")
    print("Salvando o melhor modelo com base em val_loss...\n")

    inicio = time.time()
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=[_callback_melhor_modelo(caminho_modelo)],
    )
    duracao = time.time() - inicio

    print(f"\nTempo de treino: {formatar_tempo(duracao)}")

    model = tf.keras.models.load_model(caminho_modelo)
    print(f"Melhor modelo carregado de: {caminho_modelo}")

    plot_training(history, nome_modelo, pasta_saida)

    y_true, y_pred = prever(model, test_ds)
    metricas = evaluate_and_plot(y_true, y_pred, class_names, nome_modelo, pasta_saida)

    info_tempo = {
        "modelo": nome_modelo,
        "tempo_segundos": round(duracao, 2),
        "tempo_formatado": formatar_tempo(duracao),
        "epochs": epochs,
        "melhor_modelo": str(caminho_modelo),
        "metricas_teste": metricas,
    }
    with open(caminho_tempo, "w", encoding="utf-8") as arquivo:
        json.dump(info_tempo, arquivo, indent=2, ensure_ascii=False)
    print(f"Tempo e resumo salvos em: {caminho_tempo}")

    return model, history, info_tempo

# Treina o modelo com todos os datasets do split.
def treinar_geral(nome_modelo, pasta_saida, criar_modelo):
    
    if not split_disponivel(subset="train") or not split_disponivel(subset="val") or not split_disponivel(subset="test"):
        print(f"[aviso] Split incompleto em {SPLIT_DIR}. Treino {nome_modelo} pulado.")
        return None

    print(f"Classes: {CLASS_NAMES}")
    print(f"Usando somente imagens de: {SPLIT_DIR}\n")

    train_ds = carregar_dados("train")
    val_ds = carregar_dados("val")
    test_ds = carregar_dados("test")

    model = _compilar(criar_modelo())
    return treinar_e_avaliar(
        model, train_ds, val_ds, test_ds,
        nome_modelo, pasta_saida, CLASS_NAMES, EPOCHS,
    )

# Imprime a contagem de imagens por classe.
def _imprimir_contagem(subset: str, dataset: str) -> int:
    total, por_classe = contar_imagens(subset, dataset)
    detalhes = ", ".join(f"{classe}={por_classe[classe]}" for classe in CLASS_NAMES)
    print(f"  {subset:5} ({dataset}): {total} imagens — {detalhes}")
    return total

# Treina o experimento.
def treinar_experimento(numero, dataset_train, dataset_val, dataset_test, nome_modelo, pasta_base, criar_modelo, titulo):
    if not experimento_split_disponivel(dataset_train, dataset_val, dataset_test):
        print(f"\n[aviso] Split não encontrado para experimento {numero} — pulando.")
        print(f"  Train: {SPLIT_DIR / dataset_train / 'train'}")
        print(f"  Val:   {SPLIT_DIR / dataset_val / 'val'}")
        print(f"  Test:  {SPLIT_DIR / dataset_test / 'test'}")
        return None

    pasta_saida = pasta_base / numero

    print(f"\n{'=' * 50}")
    print(f"Experimento {numero} — {titulo}")
    print(f"Fonte: {SPLIT_DIR}")
    print(f"Train: {SPLIT_DIR / dataset_train / 'train'}")
    print(f"Val:   {SPLIT_DIR / dataset_val / 'val'}")
    print(f"Test:  {SPLIT_DIR / dataset_test / 'test'}")
    print(f"Saída: {pasta_saida}")
    print(f"{'=' * 50}\n")

    print("Imagens carregadas (todas do split):")
    _imprimir_contagem("train", dataset_train)
    _imprimir_contagem("val", dataset_val)
    _imprimir_contagem("test", dataset_test)
    print()

    train_ds = carregar_dados("train", dataset_train)
    val_ds = carregar_dados("val", dataset_val)
    test_ds = carregar_dados("test", dataset_test)

    model = _compilar(criar_modelo())
    return treinar_e_avaliar(
        model, train_ds, val_ds, test_ds,
        nome_modelo, pasta_saida, CLASS_NAMES, EPOCHS,
    )

# Roda os experimentos.
def rodar_experimentos(experimentos, nome_modelo, pasta_base, criar_modelo, titulo):
    if not SPLIT_DIR.exists():
        print(f"[aviso] Split não encontrado em {SPLIT_DIR}.")
        print("Execute ger_split.py antes.")
        return

    if not configurar_gpu():
        print("Nenhum experimento será executado sem GPU.")
        return

    print(f"Usando somente imagens de: {SPLIT_DIR}\n")

    inicio = time.time()
    for numero, dataset_train, dataset_val, dataset_test in experimentos:
        treinar_experimento(
            numero, dataset_train, dataset_val, dataset_test,
            nome_modelo, pasta_base, criar_modelo, titulo,
        )

    print(f"\nTempo total — {titulo}: {formatar_tempo(time.time() - inicio)}")
