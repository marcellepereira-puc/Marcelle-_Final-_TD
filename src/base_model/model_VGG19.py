# Removendo warnings do TensorFlow
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# importando as bibliotecas
import tensorflow as tf
from tensorflow.keras.layers import ( Input, Conv2D, Dense, MaxPooling2D, Flatten)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam


#Bloco padrão 
def bloco_padrao(x, filtros, kernel_size, num_conv, name):
    i =0

    for i in range(num_conv):
        x = Conv2D(filtros, kernel_size, padding='same', activation='relu', name=name + f'_conv{i+1}')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name=name + '_pool')(x)
    return x

#Modelo VGG19
def modelo_vgg19(input_shape, num_classes):

    inputs = Input(shape=input_shape)

    x = bloco_padrao(inputs, 64, (3, 3), 2, 'block1')
    x = bloco_padrao(x, 128, (3, 3), 2, 'block2')
    x = bloco_padrao(x, 256, (3, 3), 4, 'block3')
    x = bloco_padrao(x, 512, (3, 3), 4, 'block4')
    x = bloco_padrao(x, 512, (3, 3), 4, 'block5')

    x = Flatten()(x)
    x = Dense(4096, activation='relu', name='fc1')(x)
    x = Dense(4096, activation='relu', name='fc2')(x)
    outputs = Dense(num_classes, activation='softmax', name='output')(x)

    model = Model(inputs, outputs, name='vgg19')
    return model


#Para vizualizar o modelo 
#if __name__ == "__main__":
#    model = modelo_vgg19(input_shape=(224, 224, 3), num_classes=20)

#    model.compile(
#        optimizer=Adam(learning_rate=0.001), 
#        loss='categorical_crossentropy',
#         metrics=['accuracy'])

#    print(model.summary())

    