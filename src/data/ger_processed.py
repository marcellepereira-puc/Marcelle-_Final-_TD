import random
from pathlib import Path

import numpy as np
from PIL import Image

# Script para pré-processar as imagens dos datasets, redimensionando-as, convertendo para escala de cinza (opcional) e salvando em uma nova pasta, mantendo a estrutura de pastas original.
_BASE = Path(__file__).resolve().parent
_DATASETS = _BASE.parents[1] / "datasets"

# Configurações de entrada e saída, extensões de imagem, tamanho das imagens e número de amostras para plotagem.
PASTA_ENTRADA = _DATASETS / "processed" / "processed_v1"
PASTA_SAIDA = _DATASETS / "processed" / "processed_v2"

EXTENSOES_IMAGEM = (".jpg", ".jpeg", ".png")
IMAGE_SIZE = (224, 224)
AMOSTRAS_PLOT = 6

# Função para calcular a perda triplet, que é a média da diferença entre as distâncias positiva e negativa, com uma margem, usando Keras backend.
def listar_imagens(pasta_entrada: Path) -> list[tuple[Path, Path]]:
    imagens = []

    # Verifica se a pasta de entrada existe antes de tentar listar as imagens.
    if not pasta_entrada.exists():
        print(f"[aviso] Pasta não encontrada, pulando: {pasta_entrada}")
        return imagens

    # Itera sobre os datasets, labels e arquivos, verificando se cada caminho existe e se o arquivo é uma imagem válida, antes de adicioná-lo à lista de imagens.
    for pasta_dataset in pasta_entrada.iterdir():
        if not pasta_dataset.is_dir():
            continue
        if not pasta_dataset.exists():
            print(f"[aviso] Dataset não encontrado, pulando: {pasta_dataset}")
            continue

        for pasta_label in pasta_dataset.iterdir():
            if not pasta_label.is_dir():
                continue
            if not pasta_label.exists():
                print(f"[aviso] Label não encontrada, pulando: {pasta_label}")
                continue

            for arquivo in pasta_label.rglob("*"):
                if arquivo.is_file() and arquivo.suffix.lower() in EXTENSOES_IMAGEM:
                    imagens.append((arquivo, pasta_label))

    return imagens

# Função para pré-processar uma imagem, redimensionando-a e normalizando os valores dos pixels para o intervalo [0, 1].
def preprocess_image(image, label):
    import tensorflow as tf

    image = tf.image.resize(image, IMAGE_SIZE)
    image = tf.cast(image, tf.float32) / 255.0
    return image, label

# Função para criar um pipeline de pré-processamento usando TensorFlow, aplicando a função de pré-processamento a cada elemento do dataset e otimizando o desempenho com cache e pré-busca.
def aumento_dados():
    import tensorflow as tf
    from tensorflow.keras import Sequential, layers

    return Sequential(
        [
            layers.RandomFlip("horizontal_and_vertical"),
            layers.RandomRotation(0.2),
            layers.RandomZoom(0.2),
            layers.RandomContrast(0.2),
        ]
    )

# Função para deixar as imagens em escala de cinza, verificando se a imagem já tem um canal e, caso contrário, convertendo-a usando TensorFlow.
def preto_e_branco(image):
    import tensorflow as tf

    if image.shape[-1] == 1:
        return image
    return tf.image.rgb_to_grayscale(image)

# Carregar imagem
def carregar_imagem(caminho: Path) -> np.ndarray:
    with Image.open(caminho) as imagem:
        imagem = imagem.convert("RGB")
        imagem = imagem.resize(IMAGE_SIZE)
        return np.asarray(imagem, dtype=np.float32) / 255.0

# Salvar imagem, criando a pasta de destino se necessário, e convertendo o array para uint8 antes de salvar usando PIL.
def salvar_imagem(array: np.ndarray, destino: Path) -> None:
    destino.parent.mkdir(parents=True, exist_ok=True)
    if array.ndim == 2:
        imagem = Image.fromarray((np.clip(array, 0.0, 1.0) * 255).astype(np.uint8), mode="L")
    else:
        imagem = Image.fromarray((np.clip(array, 0.0, 1.0) * 255).astype(np.uint8))
    imagem.save(destino)

#  Função para definir caminho de destino para uma imagem processada, mantendo a estrutura de pastas original, mas alterando a extensão para .png.
def caminho_destino(arquivo: Path, pasta_label: Path, pasta_entrada: Path, pasta_saida: Path) -> Path:
    relativo = arquivo.relative_to(pasta_label)
    destino_label = pasta_saida / pasta_label.relative_to(pasta_entrada)
    return destino_label / relativo.with_suffix(".png")

# Função principal para processar todas as imagens, iterando sobre a lista de imagens, pré-processando cada uma, salvando a imagem processada no destino e mantendo um resumo do número de imagens processadas, puladas e com erros.
def processar_todas(
    pasta_entrada: Path = PASTA_ENTRADA,
    pasta_saida: Path = PASTA_SAIDA,
    converter_pb: bool = False,
) -> dict[str, int]:
    pasta_entrada = Path(pasta_entrada)
    pasta_saida = Path(pasta_saida)

    if not pasta_entrada.exists():
        print(f"[aviso] Pasta de entrada não encontrada, pulando: {pasta_entrada}")
        return {"processadas": 0, "puladas": 0, "erros": 0}

    imagens = listar_imagens(pasta_entrada)
    resumo = {"processadas": 0, "puladas": 0, "erros": 0}

    print(f"Processando {len(imagens)} imagens de: {pasta_entrada}")
    print(f"Saída: {pasta_saida}\n")

    for indice, (arquivo, pasta_label) in enumerate(imagens, start=1):
        if not arquivo.exists():
            print(f"[aviso] Imagem não encontrada, pulando: {arquivo}")
            resumo["puladas"] += 1
            continue

        destino = caminho_destino(arquivo, pasta_label, pasta_entrada, pasta_saida)
        if destino.exists():
            resumo["puladas"] += 1
            continue

        try:
            array = carregar_imagem(arquivo)
            if converter_pb:
                with Image.open(arquivo) as imagem:
                    imagem = imagem.convert("L").resize(IMAGE_SIZE)
                    array = np.asarray(imagem, dtype=np.float32) / 255.0

            salvar_imagem(array, destino)
            resumo["processadas"] += 1
        except OSError as erro:
            print(f"[aviso] Erro ao processar {arquivo}: {erro}")
            resumo["erros"] += 1

        if indice % 250 == 0:
            print(f"Progresso: {indice}/{len(imagens)}")

# Imprime o resumo final do pré-processamento, incluindo o número de imagens processadas, puladas e com erros.
    print(
        f"\nConcluído — processadas={resumo['processadas']} | "
        f"puladas={resumo['puladas']} | erros={resumo['erros']}"
    )
    return resumo

# Função para criar um dataset do TensorFlow a partir das imagens pré-processadas
def criar_dataset_tensorflow(pasta_entrada: Path = PASTA_SAIDA):
    import tensorflow as tf

    pasta_entrada = Path(pasta_entrada)
    imagens = listar_imagens(pasta_entrada)
    if not imagens:
        raise FileNotFoundError(f"Nenhuma imagem encontrada em: {pasta_entrada}")

    labels_nome = sorted({pasta_label.name for _, pasta_label in imagens})
    label_para_id = {nome: indice for indice, nome in enumerate(labels_nome)}

    caminhos = [str(arquivo) for arquivo, _ in imagens]
    labels = [label_para_id[pasta_label.name] for _, pasta_label in imagens]

    def carregar(caminho, label):
        imagem = tf.io.read_file(caminho)
        imagem = tf.image.decode_image(imagem, channels=3, expand_animations=False)
        imagem = tf.image.resize(imagem, IMAGE_SIZE)
        imagem = tf.cast(imagem, tf.float32) / 255.0
        return imagem, label

    dataset = tf.data.Dataset.from_tensor_slices((caminhos, labels))
    return dataset.map(carregar, num_parallel_calls=tf.data.AUTOTUNE)

# Função para plotar um número específico de imagens do dataset pré-processado, exibindo-as em uma grade usando Matplotlib, e mostrando o caminho e a pasta de cada imagem como título.
def plot_images(pasta_entrada: Path = PASTA_SAIDA, quantidade: int = AMOSTRAS_PLOT) -> None:
    import matplotlib.pyplot as plt

    pasta_entrada = Path(pasta_entrada)
    imagens = listar_imagens(pasta_entrada)

    if not imagens:
        print("[aviso] Nenhuma imagem encontrada para plotar.")
        return

    amostras = random.sample(imagens, min(quantidade, len(imagens)))
    fig, axes = plt.subplots(1, len(amostras), figsize=(4 * len(amostras), 4))
    if len(amostras) == 1:
        axes = [axes]

    for ax, (arquivo, pasta_label) in zip(axes, amostras):
        array = carregar_imagem(arquivo)
        ax.imshow(array)
        ax.set_title(f"{pasta_label.parent.name}/{pasta_label.name}")
        ax.axis("off")

    plt.tight_layout()
    plt.show()

# Função para salvar a tabela de contagem de imagens em um arquivo CSV, criando a pasta se necessário.
if __name__ == "__main__":
    resumo = processar_todas()
    print(f"Pré-processamento concluído: {resumo['processadas']} imagens.")

    
