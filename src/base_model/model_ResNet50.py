#Removendo warnings do TensorFlow
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

#importando as bibliotecas
import tensorflow as tf
from tensorflow.keras.layers import ( Input, Conv2D, BatchNormalization, Activation, Add, GlobalAveragePooling2D, Dense)
from tensorflow.keras.models import Model

# Bloco Bottleneck REESNET 50
def bottleneck_block(x, filters, stride=1, projection=False, name="block"):
    shortcut = x

    # 1x1
    x = Conv2D(filters, (1,1), strides=stride, padding='same', name=name+'_conv1')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # 3x3
    x = Conv2D(filters, (3,3), padding='same', name=name+'_conv2')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # 1x1
    x = Conv2D(filters*4, (1,1), padding='same', name=name+'_conv3')(x)
    x = BatchNormalization()(x)


    if projection:
        shortcut = Conv2D(filters*4, (1,1), strides=stride, padding='same', name=name+'_proj')(shortcut)
        shortcut = BatchNormalization()(shortcut)

    x = Add()([x, shortcut])
    x = Activation('relu')(x)

    return x

def bottleneck_v2(x, filters, stride=1, projection=False, name="block"):
    shortcut = x

    # PRE-ACTIVATION
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # Projection shortcut
    if projection:
        shortcut = Conv2D(filters * 4, (1,1), strides=stride, padding='same', name=name+'_proj')(x)

    # 1x1
    x = Conv2D(filters, (1,1), strides=stride, padding='same', name=name+'_conv1')(x)

    # 3x3
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(filters, (3,3), padding='same', name=name+'_conv2')(x)

    # 1x1
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(filters * 4, (1,1), padding='same', name=name+'_conv3')(x)

    x = Add()([x, shortcut])
    return x



# Modelo ResNet50
def ResNet50v1_manual(input_shape=(224,224,3), num_classes=128, softmax_head=False):
    """
    num_classes: dimensão do embedding (siamese) ou número de classes (se softmax_head=True).
    softmax_head=False: vetor de características linear (ex.: rede siamesa com num_classes=128).
    softmax_head=True: cabeça softmax para classificação supervisionada.
    """
    inputs = Input(shape=input_shape)

    # Stem
    x = Conv2D(64, (7,7), strides=2, padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = tf.keras.layers.MaxPooling2D((3,3), strides=2, padding='same')(x)

    # Stage 1 (3 blocos)
    x = bottleneck_block(x, 64, projection=True, name='conv2_block1')
    for i in range(2):
        x = bottleneck_block(x, 64, name=f'conv2_block{i+2}')

    # Stage 2 (4 blocos)
    x = bottleneck_block(x, 128, stride=2, projection=True, name='conv3_block1')
    for i in range(3):
        x = bottleneck_block(x, 128, name=f'conv3_block{i+2}')

    # Stage 3 (6 blocos)
    x = bottleneck_block(x, 256, stride=2, projection=True, name='conv4_block1')
    for i in range(5):
        x = bottleneck_block(x, 256, name=f'conv4_block{i+2}')

    # Stage 4 (3 blocos)
    x = bottleneck_block(x, 512, stride=2, projection=True, name='conv5_block1')
    for i in range(2):
        x = bottleneck_block(x, 512, name=f'conv5_block{i+2}')

    x = GlobalAveragePooling2D()(x)
    if softmax_head:
        outputs = Dense(num_classes, activation="softmax")(x)
    else:
        outputs = Dense(num_classes, activation=None)(x)

    model = Model(inputs, outputs)
    return model


# Modelo ResNet50V2
def ResNet50V2_manual(input_shape=(224,224,3), num_classes=128, softmax_head=False):
    """
    num_classes: dimensao do embedding (siamese) ou numero de classes.
    softmax_head=False: vetor de caracteristicas linear para rede siamesa.
    softmax_head=True: cabeca softmax para classificacao.
    """
    inputs = Input(shape=input_shape)

    # Stem
    x = Conv2D(64, (7,7), strides=2, padding='same')(inputs)
    x = tf.keras.layers.MaxPooling2D((3,3), strides=2, padding='same')(x)

    # Stage 1 (3 blocos)
    x = bottleneck_v2(x, 64, projection=True, name='conv2_block1')
    for i in range(2):
        x = bottleneck_v2(x, 64, name=f'conv2_block{i+2}')

    # Stage 2 (4 blocos)
    x = bottleneck_v2(x, 128, stride=2, projection=True, name='conv3_block1')
    for i in range(3):
        x = bottleneck_v2(x, 128, name=f'conv3_block{i+2}')

    # Stage 3 (6 blocos)
    x = bottleneck_v2(x, 256, stride=2, projection=True, name='conv4_block1')
    for i in range(5):
        x = bottleneck_v2(x, 256, name=f'conv4_block{i+2}')

    # Stage 4 (3 blocos)
    x = bottleneck_v2(x, 512, stride=2, projection=True, name='conv5_block1')
    for i in range(2):
        x = bottleneck_v2(x, 512, name=f'conv5_block{i+2}')

    # Final (V2 tem BN antes do output)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = GlobalAveragePooling2D()(x)
    if softmax_head:
        outputs = Dense(num_classes, activation='softmax')(x)
    else:
        outputs = Dense(num_classes, activation=None)(x)

    model = Model(inputs, outputs)
    return model


# Para vizualizar o modelo 
#if __name__ == "__main__":
#    model = ResNet50_manual()
#    model.summary()
#
#    model = ResNet50V2_manual()
#    model.summary()