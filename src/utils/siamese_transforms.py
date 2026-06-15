# Aumenta a imagem para o treino e o prototipo.

import tensorflow as tf

# Aumenta a imagem para o treino.
def augmentar_treino_tf(imagem: tf.Tensor) -> tf.Tensor:
    imagem = tf.image.random_brightness(imagem, max_delta=0.06)
    imagem = tf.image.random_contrast(imagem, lower=0.92, upper=1.08)
    imagem = tf.image.random_saturation(imagem, lower=0.95, upper=1.05)
    imagem = tf.clip_by_value(imagem, 0.0, 1.0)

    altura = tf.shape(imagem)[0]
    largura = tf.shape(imagem)[1]
    escala = tf.random.uniform([], 0.94, 1.06)
    nova_altura = tf.cast(tf.round(tf.cast(altura, tf.float32) * escala), tf.int32)
    nova_largura = tf.cast(tf.round(tf.cast(largura, tf.float32) * escala), tf.int32)
    imagem = tf.image.resize(imagem, [nova_altura, nova_largura])
    imagem = tf.image.resize_with_crop_or_pad(imagem, altura, largura)
    return imagem

# Aumenta a imagem para o prototipo.
def augmentar_prototipo_tf(imagem: tf.Tensor, semente: int) -> tf.Tensor:

    imagem = tf.convert_to_tensor(imagem)
    gerador = tf.random.Generator.from_seed(semente)

    delta = gerador.uniform([], -0.04, 0.04)
    imagem = tf.clip_by_value(imagem + delta, 0.0, 1.0)

    fator = gerador.uniform([], 0.96, 1.04)
    media = tf.reduce_mean(imagem)
    imagem = tf.clip_by_value((imagem - media) * fator + media, 0.0, 1.0)
    return imagem
