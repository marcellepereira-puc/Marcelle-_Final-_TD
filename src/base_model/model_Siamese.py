import os

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Concatenate, Dense, Input, Layer
from tensorflow.keras.models import Model

# Camada customizada para normalização L2, usada no embedding.
class L2Normalize(Layer):

    def call(self, vetor):
        return tf.math.l2_normalize(vetor, axis=1)

# Camada customizada para calcular a distância euclidiana entre dois vetores de embedding.
class DistanciaEuclidiana(Layer):
    
    def call(self, vetores):
        x, y = vetores
        diferenca = x - y
        soma_quadrados = tf.reduce_sum(tf.square(diferenca), axis=1, keepdims=True)
        return tf.sqrt(tf.maximum(soma_quadrados, tf.keras.backend.epsilon()))

# Função auxiliar para calcular a distância euclidiana entre dois vetores de embedding.
def distancia_euclidiana(vetores):
    return DistanciaEuclidiana()(vetores)

# Função de perda triplet, que incentiva a distância entre a ancora e a positiva a ser menor que a distância entre a ancora e a negativa por uma margem.
def loss_triplet(y_verdadeiro, y_predito, margem=1.0):
    dist_positiva = y_predito[:, 0]
    dist_negativa = y_predito[:, 1]
    return K.mean(K.maximum(dist_positiva - dist_negativa + margem, 0.0))

# Função auxiliar para congelar as camadas iniciais do backbone.
def _congelar_backbone(base, camadas_congeladas: int) -> None:
    base.trainable = True
    if camadas_congeladas <= 0:
        return
    limite = min(camadas_congeladas, len(base.layers))
    for camada in base.layers[:limite]:
        camada.trainable = False

# Função para criar o modelo de embedding usando ResNet50V2 como backbone, seguido por uma camada densa linear e normalização L2.
def criar_modelo_embedding(
    input_shape=(224, 224, 3),
    dim_embedding=128,
    weights="imagenet",
    freeze_layers=140,
):
    # Cria o modelo de embedding usando ResNet50V2 como backbone, seguido por uma camada densa linear e normalização L2.
    entrada = Input(shape=input_shape, name="embedding_input")
    base = tf.keras.applications.ResNet50V2(
        include_top=False,
        weights=weights,
        input_shape=input_shape,
        pooling="avg",
    )
    _congelar_backbone(base, freeze_layers)
    features = base(entrada)
    vetor = Dense(dim_embedding, activation=None, name="embedding")(features)
    vetor = L2Normalize(name="l2_norm")(vetor)
    return Model(entrada, vetor, name="embedding_resnet50v2_imagenet")

# Função para criar o modelo siamese, que recebe três entradas (ancora, positiva e negativa) e retorna as distâncias entre a ancora e as outras duas entradas.
def criar_modelo_siamese(
    input_shape=(224, 224, 3),
    dim_embedding=128,
    rede=None,
    weights="imagenet",
    freeze_layers=140,
):
    # Cria o modelo siamese, que recebe três entradas (ancora, positiva e negativa) e retorna as distâncias entre a ancora e as outras duas entradas.
    if rede is None:
        rede = criar_modelo_embedding(
            input_shape, dim_embedding, weights=weights, freeze_layers=freeze_layers
        )

    entrada_ancora = Input(shape=input_shape, name="ancora")
    entrada_positiva = Input(shape=input_shape, name="positiva")
    entrada_negativa = Input(shape=input_shape, name="negativa")

    emb_ancora = rede(entrada_ancora)
    emb_positiva = rede(entrada_positiva)
    emb_negativa = rede(entrada_negativa)

    dist_positiva = DistanciaEuclidiana(name="dist_positiva")(
        [emb_ancora, emb_positiva]
    )
    dist_negativa = DistanciaEuclidiana(name="dist_negativa")(
        [emb_ancora, emb_negativa]
    )

    saida = Concatenate(name="distancias")([dist_positiva, dist_negativa])

    modelo = Model(
        inputs=[entrada_ancora, entrada_positiva, entrada_negativa],
        outputs=saida,
        name="siamese_resnet50v2",
    )
    return modelo, rede

# Função auxiliar para criar uma função de perda triplet com uma margem específica, que pode ser usada na compilação do modelo.
def _loss_triplet_com_margem(margem):
    def _loss(y_true, y_pred):
        return loss_triplet(y_true, y_pred, margem=margem)

    _loss.__name__ = "loss_triplet"
    return _loss

# Função para compilar o modelo siamese, usando o otimizador Adam e a função de perda triplet com margem.
def compilar_modelo(modelo, learning_rate=5e-5, margem=0.5):
    modelo.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=_loss_triplet_com_margem(margem),
        metrics=["mae"],
    )
    return modelo
