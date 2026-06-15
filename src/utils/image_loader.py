"""
Padrão de imagem do projeto (CNN e Siamese):
  - RGB 224×224 (LANCZOS)
  - float32 em [0, 1]  (÷ 255)
"""

from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image

from config import IMAGE_SIZE


def carregar_imagem_rgb(caminho, tamanho=IMAGE_SIZE) -> np.ndarray:
    """
    Carrega imagem com Pillow — redimensiona antes de alocar tensor completo.

    Evita OOM em raios-X de alta resolução no split/test.
    Mesmo padrão de ger_pair.py e dataset_loader (÷255).
    """
    largura, altura = tamanho
    with Image.open(Path(caminho)) as imagem:
        imagem = imagem.convert("RGB")
        imagem = imagem.resize((largura, altura), Image.Resampling.LANCZOS)
    return np.asarray(imagem, dtype=np.float32) / 255.0


def mapear_imagem_tf(path, label=None):
    """Wrapper para tf.data: path string → tensor float32 (224, 224, 3)."""

    def _carregar(path_tensor):
        caminho = path_tensor.numpy()
        if isinstance(caminho, bytes):
            caminho = caminho.decode("utf-8")
        return carregar_imagem_rgb(caminho)

    imagem = tf.py_function(_carregar, [path], tf.float32)
    imagem.set_shape((*IMAGE_SIZE, 3))
    if label is None:
        return imagem
    return imagem, label
