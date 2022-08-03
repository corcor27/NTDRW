import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Model
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
from tensorflow.keras import backend
import os
import sys
import random
import cv2
import matplotlib.pyplot as plt
import time
from tensorflow.keras.applications.resnet50 import ResNet50

## FUNCTIONS


def focal_loss_fixed(y_true, y_pred):
    gamma=2.0
    alpha=0.25
    pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
    pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
    return -K.mean(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1+K.epsilon())) - K.mean((1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0 + K.epsilon()))


def dice_coeff(y_true, y_pred, smooth=1):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_loss(y_true, y_pred):
    loss = 1 - dice_coeff(y_true, y_pred)
    return loss


    return (1 - intersection /(np.sum(y_true_f) + np.sum(y_pred_f) - intersection))
def dice_coef_np(y_true, y_pred):
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()

    intersection = np.sum(y_true_f * y_pred_f)

    return (2. * intersection) / (np.sum(y_true_f) + np.sum(y_pred_f))


def voe_coef_np(y_true, y_pred):
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()

    intersection = np.sum(y_true_f * y_pred_f)

    return (1 - intersection /
            (np.sum(y_true_f) + np.sum(y_pred_f) - intersection))
   
## LAYERS

   

def expend_as(x, n):
    y = Lambda(lambda x, repnum: K.repeat_elements(x, repnum, axis=3),
               arguments={'repnum': n})(x)

    return y


def conv_bn_act(x, filters, drop_out=0.0):
    x = Conv2D(filters, (3, 3), activation=None, padding='same')(x)

    if drop_out > 0:
        x = Dropout(drop_out)(x)

    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    return x


def bn_act_conv_dense(x, filters, drop_out=0.0):
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(filters, (1, 1), padding='same')(x)

    if drop_out > 0:
        x = Dropout(drop_out)(x)

    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(filters, (3, 3), padding='same')(x)

    if drop_out > 0:
        x = Dropout(drop_out)(x)

    return x


def dense_block(x, elements=3, filters=8, drop_out=0.0):
    
    blocks = []
    blocks.append(x)

    for i in range(elements):
        temp = bn_act_conv_dense(x, filters, drop_out)       
        blocks.append(temp)
#         for j in blocks:
#             print(K.int_shape(j))
        x = Concatenate(axis=-1)(blocks[:])
#         print(K.int_shape(x))
#         x = concatenate(blocks)

    return x

def back_layer(x, filters, drop_out):
    
    x = Conv2D(filters, (3, 3), padding='same', kernel_initializer = 'he_normal', kernel_regularizer=l2(1e-4))(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.01)(x)
    if drop_out > 0:
        x = Dropout(drop_out)(x)
    x = Conv2D(filters, (3, 3), padding='same', kernel_initializer = 'he_normal', kernel_regularizer=l2(1e-4))(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.01)(x)
    if drop_out > 0:
        x = Dropout(drop_out)(x)

    return x
   
def centre_layer(x, filters, drop_out):
    x = Conv2D(filters, (3, 3), padding='same', kernel_regularizer=l2(0.), bias_regularizer=l2(0.))(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.01)(x)
    x = Conv2D(filters, (3, 3), padding='same', kernel_regularizer=l2(0.), bias_regularizer=l2(0.))(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.01)(x)
    if drop_out > 0:
        x = Dropout(drop_out)(x)

    return x
   
def conv2dtranspose(x, filters):
    x = Conv2DTranspose(filters, kernel_size=(2, 2), strides=(2, 2), padding='same')(x)
    return x

def conv2d_o(filters):
    return Conv2D(filters=filters,
                  kernel_size=(3, 3),
                  padding='same',
                  kernel_regularizer=l2(0.),
                  bias_regularizer=l2(0.))


def conv2dtranspose_o(filters):
    return Conv2DTranspose(filters=filters,
                           kernel_size=(2, 2),
                           strides=(2, 2),
                           padding='same')

def BatchActivate(x):
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x


def selective_identity_block(input_tensor, kernel_size, filters):
    nb_filter1, nb_filter2, nb_filter3 = filters

    x = Conv2D(nb_filter1, (1, 1))(input_tensor)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = ZeroPadding2D((1, 1))(x)
    x = selective_layer(x, nb_filter2, compression=0.5,
                         drop_out=0)

    x = Conv2D(nb_filter3, (1, 1))(x)
    x = BatchNormalization()(x)

    x = Add()([x, input_tensor])
    x = Activation('relu')(x)
    return x

def identity_block(input_tensor, kernel_size, filters):
    nb_filter1, nb_filter2, nb_filter3 = filters

    x = Conv2D(nb_filter1, (1, 1))(input_tensor)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = ZeroPadding2D((1, 1))(x)
    x = Conv2D(nb_filter2, (kernel_size, kernel_size))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(nb_filter3, (1, 1))(x)
    x = BatchNormalization()(x)

    x = Add()([x, input_tensor])
    x = Activation('relu')(x)
    return x

def conv_block(input_tensor, kernel_size, filters, strides=(2, 2)):
    
    nb_filter1, nb_filter2, nb_filter3 = filters
    
    x = Conv2D(nb_filter1, (1, 1), strides=strides)(input_tensor)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = ZeroPadding2D((1, 1))(x)
    x = Conv2D(nb_filter2, (kernel_size, kernel_size))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(nb_filter3, (1, 1))(x)
    x = BatchNormalization()(x)

    shortcut = Conv2D(nb_filter3, (1, 1), strides=strides)(input_tensor)
    shortcut = BatchNormalization(axis=-1)(shortcut)

    x = Add()([x, shortcut])
    x = Activation('relu')(x)
    return x

def hidden_layers(inputs, filters, dropout):
        
        x = Conv2D(filters, (3, 3), padding="same")(inputs)
        x = Activation("relu")(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D(pool_size=(3, 3))(x)
        x = Dropout(dropout)(x)
        x = Conv2D(2*filters, (3, 3), padding="same")(x)
        x = Activation("relu")(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Dropout(dropout)(x)
        x = Conv2D(2*filters, (3, 3), padding="same")(x)
        x = Activation("relu")(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Dropout(dropout)(x)
        return x

def dense_branch(inputs, filters, num_grades, dropout):
        """
        Used to build the race branch of our face recognition network.
        This branch is composed of three Conv -> BN -> Pool -> Dropout blocks, 
        followed by the Dense output layer.
        """
        x = hidden_layers(inputs, filters, dropout)
        x = Flatten()(x)
        x = Dense(128)(x)
        x = Activation("relu")(x)
        x = BatchNormalization()(x)
        x = Dropout(dropout)(x)
        x = Dense(num_grades)(x)
        x = Activation("softmax")(x)
        return x

def selective_layer(x, filters, compression=0.5, drop_out=0.0):
    x1 = Conv2D(filters, (3, 3), dilation_rate=2, padding='same')(x)

    if drop_out > 0:
        x1 = Dropout(drop_out)(x1)

    x1 = BatchNormalization()(x1)
    x1 = Activation('relu')(x1)

    x2 = Conv2D(filters, (3, 3), padding='same')(x)

    if drop_out > 0:
        x2 = Dropout(drop_out)(x2)

    x2 = BatchNormalization()(x2)
    x2 = Activation('relu')(x2)

    x3 = add([x1, x2])

    x3 = GlobalAveragePooling2D()(x3)

    x3 = Dense(int(filters * compression))(x3)
    x3 = BatchNormalization()(x3)
    x3 = Activation('relu')(x3)

    x3 = Dense(filters)(x3)

    x3p = Activation('sigmoid')(x3)

    x3m = Lambda(lambda x: 1 - x)(x3p)

    x4 = multiply([x1, x3p])
    x5 = multiply([x2, x3m])

    return add([x4, x5])


def selective_transition_layer(x, filters, drop_out=0.0):
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = selective_layer(x, filters, drop_out=drop_out)

    return x


def transition_layer(x, compression, drop_out=0.0):
    n = K.int_shape(x)[-1]

    n = int(n * compression)

    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(n, (1, 1), padding='same')(x)

    if drop_out > 0:
        x = Dropout(drop_out)(x)

    return x


def attention_layer(d, e, n):
    d1 = Conv2D(n, (1, 1), activation=None, padding='same')(d)

    e1 = Conv2D(n, (1, 1), activation=None, padding='same')(e)

    concat_de = add([d1, e1])

    relu_de = Activation('relu')(concat_de)
    conv_de = Conv2D(1, (1, 1), padding='same')(relu_de)
    sigmoid_de = Activation('sigmoid')(conv_de)

    shape_e = K.int_shape(e)
    upsample_psi = expend_as(sigmoid_de, shape_e[3])

    return multiply([upsample_psi, e])

def unet(filters=16, dropout=0, size=(224, 224, 1), attention_gates=False):
    inp = Input(size)

    c1 = conv_bn_act(inp, filters)
    c1 = conv_bn_act(c1, filters)
    p1 = MaxPooling2D((2, 2))(c1)
    filters = 2 * filters

    c2 = conv_bn_act(p1, filters)
    c2 = conv_bn_act(c2, filters)
    p2 = MaxPooling2D((2, 2))(c2)
    filters = 2 * filters

    c3 = conv_bn_act(p2, filters)
    c3 = conv_bn_act(c3, filters)
    p3 = MaxPooling2D((2, 2))(c3)
    filters = 2 * filters

    c4 = conv_bn_act(p3, filters)
    c4 = conv_bn_act(c4, filters)
    p4 = MaxPooling2D((2, 2))(c4)
    filters = 2 * filters

    cm = conv_bn_act(p4, filters)
    cm = conv_bn_act(cm, filters)

    filters = filters // 2

    u4 = Conv2DTranspose(filters, (2, 2), strides=(2, 2), padding='same')(cm)

    if attention_gates:

        u4 = concatenate([u4, attention_layer(u4, c4, 1)], axis=3)

    else:

        u4 = concatenate([u4, c4], axis=3)

    c5 = conv_bn_act(u4, filters)
    c5 = conv_bn_act(c5, filters)

    filters = filters // 2

    u3 = Conv2DTranspose(filters, (2, 2), strides=(2, 2), padding='same')(c5)

    if attention_gates:

        u3 = concatenate([u3, attention_layer(u3, c3, 1)], axis=3)

    else:

        u3 = concatenate([u3, c3], axis=3)

    c6 = conv_bn_act(u3, filters)
    c6 = conv_bn_act(c6, filters)

    filters = filters // 2

    u2 = Conv2DTranspose(filters, (2, 2), strides=(2, 2), padding='same')(c6)

    if attention_gates:

        u2 = concatenate([u2, attention_layer(u2, c2, 1)], axis=3)

    else:

        u2 = concatenate([u2, c2], axis=3)

    c7 = conv_bn_act(u2, filters)
    c7 = conv_bn_act(c7, filters)

    filters = filters // 2

    u1 = Conv2DTranspose(filters, (2, 2), strides=(2, 2), padding='same')(c7)

    if attention_gates:

        u1 = concatenate([u1, attention_layer(u1, c1, 1)], axis=3)

    else:

        u1 = concatenate([u1, c1], axis=3)

    c8 = conv_bn_act(u1, filters)
    c8 = conv_bn_act(c8, filters)

    c9 = Conv2D(1, (1, 1), padding="same", activation='sigmoid')(c8)

    model = keras.models.Model(inputs = inp, outputs = c9)

    return model
   
def selective_unet(filters=16, drop_out=0, compression=0.5, size=(512, 512, 3),
                   half_net=False, attention_gates=False):
    inp = Input(size)

    c1 = selective_layer(inp, filters, compression=compression,
                         drop_out=drop_out)
    c1 = selective_layer(c1, filters, compression=compression,
                         drop_out=drop_out)
    p1 = MaxPooling2D((2, 2))(c1)
    filters = 2 * filters

    c2 = selective_layer(p1, filters, compression=compression,
                         drop_out=drop_out)
    c2 = selective_layer(c2, filters, compression=compression,
                         drop_out=drop_out)
    p2 = MaxPooling2D((2, 2))(c2)
    filters = 2 * filters

    c3 = selective_layer(p2, filters, compression=compression,
                         drop_out=drop_out)
    c3 = selective_layer(c3, filters, compression=compression,
                         drop_out=drop_out)
    p3 = MaxPooling2D((2, 2))(c3)
    filters = 2 * filters

    c4 = selective_layer(p3, filters, compression=compression,
                         drop_out=drop_out)
    c4 = selective_layer(c4, filters, compression=compression,
                         drop_out=drop_out)
    p4 = MaxPooling2D((2, 2))(c4)
    filters = 2 * filters

    cm = selective_layer(p4, filters, compression=compression,
                         drop_out=drop_out)
    cm = selective_layer(cm, filters, compression=compression,
                         drop_out=drop_out)

    filters = filters // 2

    u4 = Conv2DTranspose(filters, (2, 2), strides=(2, 2), padding='same')(cm)

    if attention_gates:

        u4 = concatenate([u4, attention_layer(u4, c4, 1)], axis=3)

    else:

        u4 = concatenate([u4, c4], axis=3)

    if half_net:

        c5 = conv_bn_act(u4, filters, drop_out=drop_out)
        c5 = conv_bn_act(c5, filters, drop_out=drop_out)

    else:

        c5 = selective_layer(u4, filters, compression=compression,
                             drop_out=drop_out)
        c5 = selective_layer(c5, filters, compression=compression,
                             drop_out=drop_out)

    filters = filters // 2

    u3 = Conv2DTranspose(filters, (2, 2), strides=(2, 2), padding='same')(c5)

    if attention_gates:

        u3 = concatenate([u3, attention_layer(u3, c3, 1)], axis=3)

    else:

        u3 = concatenate([u3, c3], axis=3)

    if half_net:

        c6 = conv_bn_act(u3, filters, drop_out=drop_out)
        c6 = conv_bn_act(c6, filters, drop_out=drop_out)

    else:

        c6 = selective_layer(u3, filters, compression=compression,
                             drop_out=drop_out)
        c6 = selective_layer(c6, filters, compression=compression,
                             drop_out=drop_out)

    filters = filters // 2

    u2 = Conv2DTranspose(filters, (2, 2), strides=(2, 2), padding='same')(c6)

    if attention_gates:

        u2 = concatenate([u2, attention_layer(u2, c2, 1)], axis=3)

    else:

        u2 = concatenate([u2, c2], axis=3)

    if half_net:

        c7 = conv_bn_act(u2, filters, drop_out=drop_out)
        c7 = conv_bn_act(c7, filters, drop_out=drop_out)

    else:

        c7 = selective_layer(u2, filters, compression=compression,
                             drop_out=drop_out)
        c7 = selective_layer(c7, filters, compression=compression,
                             drop_out=drop_out)

    filters = filters // 2

    u1 = Conv2DTranspose(filters, (2, 2), strides=(2, 2), padding='same')(c7)

    if attention_gates:

        u1 = concatenate([u1, attention_layer(u1, c1, 1)], axis=3)

    else:

        u1 = concatenate([u1, c1], axis=3)

    if half_net:

        c8 = conv_bn_act(u1, filters, drop_out=drop_out)
        c8 = conv_bn_act(c8, filters, drop_out=drop_out)

    else:

        c8 = selective_layer(u1, filters, compression=compression,
                             drop_out=drop_out)
        c8 = selective_layer(c8, filters, compression=compression,
                             drop_out=drop_out)

    c9 = Conv2D(1, (1, 1), padding="same", activation='sigmoid')(c8)
    model = keras.models.Model(inputs = inp, outputs = c9)
    
    return model
   
def denseunet(filters=8, blocks=3, layers=3, compression=0.5, drop_out=0,
               size=(224, 224, 1), half_net=True, attention_gates=True):
    
    inp = Input(size)

    x = Conv2D(filters, (3, 3), activation=None, padding='same')(inp)

    if drop_out > 0:
        x = Dropout(drop_out)(x)

    names = {}
   
    for i in range(layers):
        
        x = dense_block(x, blocks, filters, drop_out)
#         x = transition_layer(x, compression, drop_out)
#         x = BatchNormalization()(x)
#         x = Activation('relu')(x)

        name = 'x' + str(i + 1)
        names[name] = x
        
        x = transition_layer(x, compression, drop_out)
        x = MaxPooling2D((2, 2))(x)

        filters = 2 * filters

    x = dense_block(x, blocks, filters, drop_out)
    x = transition_layer(x, compression, drop_out)
#     x = BatchNormalization()(x)
#     x = Activation('relu')(x)

    for i in range(layers):

        filters = filters // 2

        name = 'x' + str(layers - i)

        x = Conv2DTranspose(filters, (2, 2), strides=(2, 2), padding='same')(x)

        if attention_gates:

            x = Concatenate()([x, attention_layer(x, names[name], 1)])

        else:

            x = Concatenate()([x, names[name]])

        if half_net:

            x = conv_bn_act(x, filters, drop_out)
            x = conv_bn_act(x, filters, drop_out)

        else:

            x = dense_block(x, blocks, filters, drop_out)
            x = transition_layer(x, compression, drop_out)
#             x = BatchNormalization()(x)
#             x = Activation('relu')(x)
    
#     x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(1, (1, 1), padding="same", activation='sigmoid')(x)

    model = tf.keras.Model(inputs = inp, outputs = x)
 
    return model

def Unetplusplus(size=(224, 224, 1), drop_out=0):
    
    filters = [16,32,64,128,256]
    inp = Input(size)
    X00 = back_layer(inp, filters[0], drop_out)
    PL0 = MaxPooling2D(pool_size=(2, 2))(X00)

    X10 = back_layer(PL0, filters[1], drop_out)
    PL1 = MaxPooling2D(pool_size=(2, 2))(X10)

    X01 = conv2dtranspose(X10, filters[0])
    X01 = concatenate([X00, X01])
    X01 = centre_layer(X01, filters[0], drop_out)

    X20 = back_layer(PL1, filters[2], drop_out)
    PL2 = MaxPooling2D(pool_size=(2, 2))(X20)

    X11 = conv2dtranspose(X20, filters[0])
    X11 = concatenate([X10, X11])
    X11 = centre_layer(X11, filters[0], drop_out)

    X02 = conv2dtranspose(X11, filters[0])
    X02 = concatenate([X00, X01, X02])
    X02 = centre_layer(X02, filters[0], drop_out)

    X30 = back_layer(PL2, filters[3], drop_out)
    PL3 = MaxPooling2D(pool_size=(2, 2))(X30)

    X21 = conv2dtranspose(X30, filters[0])
    X21 = concatenate([X20, X21])
    X21 = centre_layer(X21, filters[0], drop_out)

    X12 = conv2dtranspose(X21, filters[0])
    X12 = concatenate([X10, X11, X12])
    X12 = centre_layer(X12, filters[0], drop_out)

    X03 = conv2dtranspose(X12, filters[0])
    X03 = concatenate([X00, X01, X02, X03])
    X03 = centre_layer(X03, filters[0], drop_out)

    M = centre_layer(PL3, filters[4], drop_out)

    X31 = conv2dtranspose(M, filters[3])
    X31 = concatenate([X31, X30])
    X31 = centre_layer(X31, filters[3], drop_out)

    X22 = conv2dtranspose(X31, filters[2])
    X22 = concatenate([X22, X20, X21])
    X22 = centre_layer(X22, filters[2], drop_out)

    X13 = conv2dtranspose(X22, filters[1])
    X13 = concatenate([X13, X10, X11, X12])
    X13 = centre_layer(X13, filters[1], drop_out)

    X04 = conv2dtranspose(X13, filters[0])
    X04 = concatenate([X04, X00, X01, X02, X03], axis=3)
    X04 = centre_layer(X04, filters[0], drop_out=0.0)

    output = Conv2D(1, kernel_size=(1, 1), activation='sigmoid')(X04)

    model = tf.keras.Model(inputs=inp, outputs=output)

    return model

    
def GRADE_MODEL(size, dropout=0.1, filters = 16, num_grades = 2):
    inp = Input(size)
    GRADE_branch = dense_branch(inp, filters, num_grades, dropout)
    model = tf.keras.Model(inputs=inp, outputs = GRADE_branch)
    return model

def ResNet101(size, classes = 2, filters=16, depth = 23):
    inp = Input(size)

    x = ZeroPadding2D((3, 3))(inp)
    x = Conv2D(8*filters, (7, 7), strides=(2, 2))(x)
    x = BatchNormalization(axis=-1)(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((7, 7), strides=(2, 2))(x)

    x = conv_block(x, 3, [filters, filters, 4*filters], strides=(1, 1))
    x = identity_block(x, 3, [2*filters, 2*filters, 4*filters])
    x = identity_block(x, 3, [2*filters, 2*filters, 4*filters])

    x = conv_block(x, 3, [4*filters, 4*filters, 16*filters])
    for i in range(1, 3):
        x = identity_block(x, 3, [4*filters, 4*filters, 16*filters])

    x = conv_block(x, 3, [8*filters, 8*filters, 32*filters])
    for i in range(1, 23):
        x = identity_block(x, 3, [8*filters, 8*filters, 32*filters])

    x = conv_block(x, 3, [16*filters, 16*filters, 64*filters])
    x = identity_block(x, 3, [16*filters, 16*filters, 64*filters])
    x = identity_block(x, 3, [16*filters, 16*filters, 64*filters]) #2048
    x = AveragePooling2D((7, 7),name="outlayer")(x)

    x = Flatten()(x)

    x = Dense(classes)(x)
    x = Activation("softmax")(x)

    model = tf.keras.Model(inputs=inp, outputs = x)
    return model  

def ResNet152(size, classes = 2, filters=16, depth = 23):
    inp = Input(size)

    x = ZeroPadding2D((3, 3))(inp)
    x = Conv2D(8*filters, (7, 7), strides=(2, 2))(x)
    x = BatchNormalization(axis=-1)(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((7, 7), strides=(2, 2))(x)

    x = conv_block(x, 3, [filters, filters, 4*filters], strides=(1, 1))
    x = identity_block(x, 3, [2*filters, 2*filters, 4*filters])
    x = identity_block(x, 3, [2*filters, 2*filters, 4*filters])

    x = conv_block(x, 3, [4*filters, 4*filters, 16*filters])
    for i in range(1, 8):
        x = identity_block(x, 3, [4*filters, 4*filters, 16*filters])

    x = conv_block(x, 3, [8*filters, 8*filters, 32*filters])
    for i in range(1, 36):
        x = identity_block(x, 3, [8*filters, 8*filters, 32*filters])

    x = conv_block(x, 3, [16*filters, 16*filters, 64*filters])
    x = identity_block(x, 3, [16*filters, 16*filters, 64*filters])
    x = identity_block(x, 3, [16*filters, 16*filters, 64*filters]) #2048
    x = AveragePooling2D((7, 7),name="outlayer")(x)

    x = Flatten()(x)

    x = Dense(classes)(x)
    x = Activation("softmax")(x)

    model = tf.keras.Model(inputs=inp, outputs = x)
    return model   

def ResNet254(size, classes = 2, filters=16, depth = 23):
    inp = Input(size)

    x = ZeroPadding2D((3, 3))(inp)
    x = Conv2D(8*filters, (7, 7), strides=(2, 2))(x)
    x = BatchNormalization(axis=-1)(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((7, 7), strides=(2, 2))(x)

    x = conv_block(x, 3, [filters, filters, 4*filters], strides=(1, 1))
    x = identity_block(x, 3, [2*filters, 2*filters, 4*filters])
    x = identity_block(x, 3, [2*filters, 2*filters, 4*filters])

    x = conv_block(x, 3, [4*filters, 4*filters, 16*filters])
    for i in range(1, 18):
        x = identity_block(x, 3, [4*filters, 4*filters, 16*filters])

    x = conv_block(x, 3, [8*filters, 8*filters, 32*filters])
    for i in range(1, 62):
        x = identity_block(x, 3, [8*filters, 8*filters, 32*filters])

    x = conv_block(x, 3, [16*filters, 16*filters, 64*filters])
    x = identity_block(x, 3, [16*filters, 16*filters, 64*filters])
    x = identity_block(x, 3, [16*filters, 16*filters, 64*filters]) #2048
    x = AveragePooling2D((7, 7),name="outlayer")(x)

    x = Flatten()(x)

    x = Dense(classes)(x)
    x = Activation("softmax")(x)

    model = tf.keras.Model(inputs=inp, outputs = x)
    return model   

def SK_ResNet101(size, num_grades = 3):
    inp = Input(size)

    x = ZeroPadding2D((3, 3))(inp)
    x = Conv2D(64, (7, 7), strides=(2, 2))(x)
    x = BatchNormalization(axis=-1)(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = conv_block(x, 3, [8, 8, 32], strides=(1, 1))
    x = selective_identity_block(x, 3, [16, 16, 32])
    x = selective_identity_block(x, 3, [16, 16, 32])

    x = conv_block(x, 3, [32, 32, 128])
    for i in range(1, 3):
        x = selective_identity_block(x, 3, [32, 32, 128])

    x = conv_block(x, 3, [64, 64, 256])
    for i in range(1, 23):
        x = selective_identity_block(x, 3, [64, 64, 256])

    x = conv_block(x, 3, [128, 128, 512])
    x = selective_identity_block(x, 3, [128, 128, 512])
    x = selective_identity_block(x, 3, [128, 128, 512]) #2048

    x = Flatten()(x)
    x = Dense(128)(x)
    x = Activation("relu")(x)
    x = BatchNormalization()(x)

    x = Dense(num_grades)(x)
    x = Activation("softmax")(x)

    model = tf.keras.Model(inputs=inp, outputs = x)
    return model

def fire_module(x, fire_id, squeeze=16, expand=64):
    s_id = 'fire' + str(fire_id) + '/'

    if K.image_data_format() == 'channels_first':
        channel_axis = 1
    else:
        channel_axis = 3
    
    x = Convolution2D(squeeze, (1, 1), padding='valid', name=s_id + "sq1x1")(x)
    x = Activation('relu', name=s_id + "relu" + "sq1x1")(x)

    left = Convolution2D(expand, (1, 1), padding='valid', name=s_id + "exp1x1")(x)
    left = Activation('relu', name=s_id + "relu" + "exp1x1")(left)

    right = Convolution2D(expand, (3, 3), padding='same', name=s_id + "exp3x3")(x)
    right = Activation('relu', name=s_id + "relu" + "exp3x3")(right)

    x = concatenate([left, right], axis=channel_axis, name=s_id + 'concat')
    return x

def SqueezeNet(size, include_top=True, pooling='ang', classes=2):
    

    inp = Input(size)
    x = Convolution2D(64, (3, 3), strides=(2, 2), padding='valid', name='conv1')(inp)
    x = Activation('gelu', name='relu_conv1')(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool1')(x)

    x = fire_module(x, fire_id=2, squeeze=16, expand=64)
    x = fire_module(x, fire_id=3, squeeze=16, expand=64)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool3')(x)

    x = fire_module(x, fire_id=4, squeeze=32, expand=128)
    x = fire_module(x, fire_id=5, squeeze=32, expand=128)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool5')(x)

    x = fire_module(x, fire_id=6, squeeze=48, expand=192)
    x = fire_module(x, fire_id=7, squeeze=48, expand=192)
    x = fire_module(x, fire_id=8, squeeze=64, expand=256)
    x = fire_module(x, fire_id=9, squeeze=64, expand=256)
    
    if include_top:
        # It's not obvious where to cut the network... 
        # Could do the 8th or 9th layer... some work recommends cutting earlier layers.
    
        x = Dropout(0.2, name='drop9')(x)

        x = Convolution2D(classes, (1, 1), padding='valid', name='conv10')(x)
        x = Activation('gelu', name='relu_conv10')(x)
        x = GlobalAveragePooling2D()(x)
        x = Activation('softmax', name='loss')(x)
    else:
        if pooling == 'avg':
            x = GlobalAveragePooling2D()(x)
        elif pooling=='max':
            x = GlobalMaxPooling2D()(x)
        elif pooling==None:
            pass
        else:
            raise ValueError("Unknown argument for 'pooling'=" + pooling)
    #x = Flatten()(x)
    #x = Dense(classes)(x)
    x = Activation("softmax")(x)
    #x = BatchNormalization()(x)

    #x = Dense(classes)(x)
    #x = Activation("softmax")(x)
    
    model = tf.keras.Model(inputs=inp, outputs = x, name='squeezenet')
    return model



def SIMPLE_CON(size, num_grades = 2, filters=8, drop_out = 0):
    inp = Input(size)

    x = ZeroPadding2D((3, 3))(inp)
    x = Conv2D(64, (5, 5), strides=(2, 2),name="layer1")(x)
    x = BatchNormalization(axis=-1)(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = conv_block(x, 3, [filters, filters, 4*filters], strides=(1, 1))
    x = identity_block(x, 3, [2*filters, 2*filters, 4*filters])
    x = AveragePooling2D((5, 5),name="last_layer")(x)

    x = Flatten()(x)
    

    x = Dense(num_grades)(x)
    x = Activation("softmax")(x)

    model = tf.keras.Model(inputs=inp, outputs = x)
    return model


def CON_1(size, num_grades = 2, filters=8, compression = 0.5, drop_out = 0):
    inp = Input(size)

    x = selective_layer(inp, filters, compression=compression,
                         drop_out=drop_out)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(8*filters, (7, 7), strides=(2, 2),name="layer1")(x)
    x = BatchNormalization(axis=-1)(x)
    x = Activation('relu')(x)
    x = selective_layer(inp, filters, compression=compression,
                         drop_out=drop_out)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = conv_block(x, 3, [filters, filters, 4*filters], strides=(1, 1))
    x = identity_block(x, 3, [2*filters, 2*filters, 4*filters])
    x = AveragePooling2D((5, 5),name="last_layer")(x)

    x = Flatten()(x)
    

    x = Dense(num_grades)(x)
    x = Activation("softmax")(x)

    model = tf.keras.Model(inputs=inp, outputs = x)
    return model


def CON_2(size, num_grades = 2, filters=8, compression = 0.5, drop_out = 0):
    inp = Input(size)

    x = ZeroPadding2D((3, 3))(inp)
    x = Conv2D(8*filters, (7, 7), strides=(2, 2))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    p1 = MaxPooling2D((7, 7), strides=(2, 2))(x)

    x = conv_block(p1, 3, [filters, filters, 4*filters], strides=(1, 1))
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = concatenate([x,p1])
    p2 = MaxPooling2D((2, 2))(x)
    
    x = conv_block(p2, 3, [4*filters, 4*filters, 16*filters], strides=(1, 1))
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = concatenate([x,p2])
    p3 = MaxPooling2D((2, 2))(x)

    x = conv_block(p3, 3, [16*filters, 16*filters, 64*filters], strides=(1, 1))

    for i in range(0, 5):
        x = identity_block(x, 3, [16*filters, 16*filters, 64*filters])
    

    x = concatenate([x,p3])
    x = AveragePooling2D((7, 7),name="outlayer")(x)
    x = Flatten()(x)
    

    x = Dense(num_grades)(x)
    x = Activation("softmax")(x)

    model = tf.keras.Model(inputs=inp, outputs = x)
    return model


def identity_block2(input_tensor, kernel_size, filters):
    nb_filter1, nb_filter2, nb_filter3 = filters

    x = Conv2D(nb_filter1, (1, 1))(input_tensor)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = ZeroPadding2D((1, 1))(x)
    x = Conv2D(nb_filter2, (kernel_size, kernel_size))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(nb_filter3, (1, 1))(x)
    x = BatchNormalization()(x)

    x = Add()([x, input_tensor])
    x = Activation('relu')(x)
    return x

def conv_block2(input_tensor, kernel_size, filters, strides=(2, 2)):
    
    nb_filter1, nb_filter2, nb_filter3 = filters
    
    x = Conv2D(nb_filter1, (1, 1), strides=strides)(input_tensor)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = ZeroPadding2D((1, 1))(x)
    x = Conv2D(nb_filter2, (kernel_size, kernel_size))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(nb_filter3, (1, 1))(x)
    x = BatchNormalization()(x)

    shortcut = Conv2D(nb_filter3, (1, 1), strides=strides)(input_tensor)
    shortcut = BatchNormalization(axis=-1)(shortcut)

    x = Add()([x, shortcut])
    x = Activation('relu')(x)
    return x

def ResNet2(depth, size, num_grades = 2, filters=8):
    inp = Input(size)

    x = ZeroPadding2D((3, 3))(inp)
    x = Conv2D(8*filters, (7, 7), strides=(2, 2))(x)
    x = BatchNormalization(axis=-1)(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = conv_block(x, 3, [filters, filters, 4*filters], strides=(1, 1))
    x = identity_block(x, 3, [2*filters, 2*filters, 4*filters])
    x = identity_block(x, 3, [2*filters, 2*filters, 4*filters])

    x = conv_block(x, 3, [4*filters, 4*filters, 16*filters])
    for i in range(1, 3):
        x = identity_block(x, 3, [4*filters, 4*filters, 16*filters])

    x = conv_block(x, 3, [8*filters, 8*filters, 32*filters])
    for i in range(1, depth):
        x = identity_block(x, 3, [8*filters, 8*filters, 32*filters])

    x = conv_block(x, 3, [16*filters, 16*filters, 64*filters])
    x = identity_block(x, 3, [16*filters, 16*filters, 64*filters])
    x = identity_block(x, 3, [16*filters, 16*filters, 64*filters]) #2048
    x = AveragePooling2D((7, 7),name="outlayer")(x)

    x = Flatten()(x)
    x = Dense(128)(x)
    x = Activation("relu")(x)
    x = BatchNormalization()(x)

    x = Dense(num_grades)(x)
    x = Activation("softmax")(x)

    model = tf.keras.Model(inputs=inp, outputs = x)
    return model   

def attention56(size, num_grades = 2, filters=8):
    
    
    inp = Input(size)
    x = Conv2D(8*filters, (7, 7), strides=(2, 2))(inp)
    p1 = MaxPooling2D((3, 3), strides=(2, 2))(x)
    x = conv_block(p1, 3, [8*filters, 8*filters, 32*filters], strides=(1, 1))
    x = identity_block(x, 3, [8*filters, 8*filters, 32*filters])
    x = concatenate([x, attention_layer(x, p1, 1)], axis=3)
    p2 = MaxPooling2D((3, 3), strides=(2, 2))(x)
    x = conv_block(p2, 3, [16*filters, 16*filters, 64*filters], strides=(1, 1))
    x = identity_block(x, 3, [16*filters, 16*filters, 64*filters])
    x = concatenate([x, attention_layer(x, p2, 1)], axis=3)
    p3 = MaxPooling2D((3, 3), strides=(2, 2))(x)
    x = conv_block(p3, 3, [32*filters, 32*filters, 128*filters], strides=(1, 1))
    x = identity_block(x, 3, [32*filters, 32*filters, 128*filters])
    x = concatenate([x, attention_layer(x, p3, 1)], axis=3)
    p4 = MaxPooling2D((3, 3), strides=(2, 2))(x)
    x = conv_block(p4, 3, [64*filters, 64*filters, 256*filters], strides=(1, 1))
    x = identity_block(x, 3, [64*filters, 64*filters, 256*filters])
    x = AveragePooling2D((2, 2),name="outlayer")(x)
    x = Flatten()(x)
    x = Dense(num_grades)(x)
    x = Activation("softmax")(x)

    model = tf.keras.Model(inputs=inp, outputs = x)
    return model   
    
    
  
def VGV(size, num_grades = 2, filters=16):
    
    inp = Input(size)
    x = Conv2D(8*filters, (3, 3), padding="same", activation='relu')(inp)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)
    x = Conv2D(16*filters, (3, 3), padding="same", activation='relu')(x)
    x = Conv2D(16*filters, (3, 3), padding="same", activation='relu')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)
    x = Conv2D(32*filters, (3, 3), padding="same", activation='relu')(x)
    x = Conv2D(32*filters, (3, 3), padding="same", activation='relu')(x)
    x = Conv2D(32*filters, (3, 3), padding="same", activation='relu')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)
    x = Conv2D(64*filters, (3, 3), padding="same", activation='relu')(x)
    x = Conv2D(64*filters, (3, 3), padding="same", activation='relu')(x)
    x = Conv2D(64*filters, (3, 3), padding="same", activation='relu')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)
    x = Conv2D(128*filters, (3, 3), padding="same", activation='relu')(x)
    x = Conv2D(128*filters, (3, 3), padding="same", activation='relu')(x)
    x = Conv2D(128*filters, (3, 3), padding="same", activation='relu')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)
    x = Flatten()(x)
    x = Dense(4069, activation="relu")(x)
    x = Dense(4069, activation="relu")(x)
    x = Dense(num_grades, activation="softmax")(x)
    model = tf.keras.Model(inputs=inp, outputs = x)
    return model 

def ResNet50(size, num_grades = 2, filters=64):

    inp = Input(size)

    x = ZeroPadding2D((3, 3))(inp)
    x = Conv2D(filters, (7, 7), strides=(2, 2))(x)
    x = BatchNormalization(axis=-1)(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((7, 7), strides=(2, 2))(x)

    x = conv_block(x, 3, [filters, filters, 4*filters], strides=(1, 1))
    x = identity_block(x, 3, [filters, filters, 4*filters])
    x = identity_block(x, 3, [filters, filters, 4*filters])

    x = conv_block(x, 3, [2*filters, 2*filters, 8*filters])
    for i in range(0, 3):
        x = identity_block(x, 3, [2*filters, 2*filters, 8*filters])

    x = conv_block(x, 3, [4*filters, 4*filters, 16*filters])
    for i in range(0, 5):
        x = identity_block(x, 3, [4*filters, 4*filters, 16*filters])

    x = conv_block(x, 3, [8*filters, 8*filters, 32*filters])
    x = identity_block(x, 3, [8*filters, 8*filters, 32*filters])
    x = identity_block(x, 3, [8*filters, 8*filters, 32*filters]) #2048
    x = AveragePooling2D((7, 7),name="outlayer")(x)

    x = Flatten()(x)

    x = Dense(num_grades)(x)
    x = Activation("softmax")(x)

    model = tf.keras.Model(inputs=inp, outputs = x)
    return model   

def GAO_NET(size, num_grades = 2, filters=8):
    inp = Input(size)
    x = Conv2D(filters, (7, 7), strides=(2, 2))(inp)
    x = BatchNormalization(axis=-1)(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), strides=(3, 3), padding='same')(x)
    x = Conv2D(96*filters, (5, 5), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)
    x = Conv2D(256*filters, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(384*filters, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(384*filters, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), strides=(3, 3))(x)
    x = Flatten()(x)
    x = Dense(256)(x)
    x = Activation("relu")(x)
    x = BatchNormalization()(x)
    x = Dense(2048)(x)
    x = Activation("relu")(x)
    x = BatchNormalization()(x)

    x = Dense(num_grades)(x)
    x = Activation("softmax")(x)

    model = tf.keras.Model(inputs=inp, outputs = x)
    return model
def MSD_NET(size, classes = 9, filters=32):
    inp = Input(size)
    x = ResNet50(include_top=False, weights="imagenet", classes=1000)(inp)
    x = GlobalAveragePooling2D()(x)
    x = Activation("gelu")(x)
    x = BatchNormalization()(x)
    x = Dense(classes)(x)
    x = Activation("softmax")(x)
    model = tf.keras.Model(inputs=inp, outputs = x)
    return model
    


def dense_block(x, blocks, name):
    """A dense block.
    # Arguments
        x: input tensor.
        blocks: integer, the number of building blocks.
        name: string, block label.
    # Returns
        output tensor for the block.
    """
    for i in range(blocks):
        x = dense_conv_block(x, 32, name=name + '_block' + str(i + 1))
    return x


def transition_block(x, reduction, name):
    """A transition block.
    # Arguments
        x: input tensor.
        reduction: float, compression rate at transition layers.
        name: string, block label.
    # Returns
        output tensor for the block.
    """
    
    x = BatchNormalization(axis=3, epsilon=1.001e-5,
                                  name=name + '_bn')(x)
    x = Activation('relu', name=name + '_relu')(x)
    x = Conv2D(int(backend.int_shape(x)[3] * reduction), 1,
                      use_bias=False,
                      name=name + '_conv')(x)
    x = AveragePooling2D(2, strides=2, name=name + '_pool')(x)
    return x


def dense_conv_block(x, growth_rate, name):
    
    x1 = BatchNormalization(axis=3,
                                   epsilon=1.001e-5,
                                   name=name + '_0_bn')(x)
    x1 = Activation('relu', name=name + '_0_relu')(x1)
    x1 = Conv2D(4 * growth_rate, 1,
                       use_bias=False,
                       name=name + '_1_conv')(x1)
    x1 = BatchNormalization(axis=3, epsilon=1.001e-5,
                                   name=name + '_1_bn')(x1)
    x1 = Activation('relu', name=name + '_1_relu')(x1)
    x1 = Conv2D(growth_rate, 3, padding='same', use_bias=False, name=name + '_2_conv')(x1)
    x = Concatenate(axis=3, name=name + '_concat')([x, x1])
    return x
 
    
def DenseNet(size, pooling="avg", classes=2):

    
    inp = Input(size)
    # Determine proper input shape
    blocks = [6, 12, 24, 16]

    x = ZeroPadding2D(padding=((3, 3), (3, 3)))(inp)
    x = Conv2D(64, 7, strides=2, use_bias=False, name='conv1/conv')(x)
    x = BatchNormalization(axis=3, epsilon=1.001e-5, name='conv1/bn')(x)
    x = Activation('relu', name='conv1/relu')(x)
    x = ZeroPadding2D(padding=((1, 1), (1, 1)))(x)
    x = MaxPooling2D(3, strides=2, name='pool1')(x)

    x = dense_block(x, blocks[0], name='conv2')
    x = transition_block(x, 0.5, name='pool2')
    x = dense_block(x, blocks[1], name='conv3')
    x = transition_block(x, 0.5, name='pool3')
    x = dense_block(x, blocks[2], name='conv4')
    x = transition_block(x, 0.5, name='pool4')
    x = dense_block(x, blocks[3], name='conv5')

    x = BatchNormalization(axis=3, epsilon=1.001e-5, name='bn')(x)
    x = Activation('relu', name='relu')(x)
    x = GlobalAveragePooling2D(name='avg_pool')(x)
    x = Dense(classes, activation='linear', name='fc1000')(x)
    



    model = tf.keras.Model(inputs=inp, outputs = x)

    return model


def AttentionResNet92(shape=(224, 224, 3), n_channels=64, n_classes=100,
                      dropout=0, regularization=0.01):
    """
    Attention-92 ResNet
    https://arxiv.org/abs/1704.06904
    """
    regularizer = l2(regularization)

    input_ = Input(shape=shape)
    x = Conv2D(n_channels, (7, 7), strides=(2, 2), padding='same')(input_) # 112x112
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)  # 56x56

    x = residual_block(x, output_channels=n_channels * 4)  # 56x56
    x = attention_block(x, encoder_depth=3)  # bottleneck 7x7

    x = residual_block(x, output_channels=n_channels * 8, stride=2)  # 28x28
    x = attention_block(x, encoder_depth=2)  # bottleneck 7x7
    x = attention_block(x, encoder_depth=2)  # bottleneck 7x7

    x = residual_block(x, output_channels=n_channels * 16, stride=2)  # 14x14
    x = attention_block(x, encoder_depth=1)  # bottleneck 7x7
    x = attention_block(x, encoder_depth=1)  # bottleneck 7x7
    x = attention_block(x, encoder_depth=1)  # bottleneck 7x7

    x = residual_block(x, output_channels=n_channels * 32, stride=2)  # 7x7
    x = residual_block(x, output_channels=n_channels * 32)
    x = residual_block(x, output_channels=n_channels * 32)

    pool_size = (x.get_shape()[1], x.get_shape()[2])
    x = AveragePooling2D(pool_size=pool_size, strides=(1, 1))(x)
    x = Flatten()(x)
    if dropout:
        x = Dropout(dropout)(x)
    output = Dense(n_classes, kernel_regularizer=regularizer, activation='softmax')(x)

    model = Model(input_, output)
    return model


def AttentionResNet56(shape=(224, 224, 3), n_channels=64, n_classes=2,
                      dropout=0, regularization=0.01):
    """
    Attention-56 ResNet
    https://arxiv.org/abs/1704.06904
    """

    regularizer = l2(regularization)

    input_ = Input(shape=shape)
    x = Conv2D(n_channels, (7, 7), strides=(2, 2), padding='same')(input_) # 112x112
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)  # 56x56

    x = residual_block(x, output_channels=n_channels * 4)  # 56x56
    x = attention_block(x, encoder_depth=3)  # bottleneck 7x7

    x = residual_block(x, output_channels=n_channels * 8, stride=2)  # 28x28
    x = attention_block(x, encoder_depth=2)  # bottleneck 7x7

    x = residual_block(x, output_channels=n_channels * 16, stride=2)  # 14x14
    x = attention_block(x, encoder_depth=1)  # bottleneck 7x7

    x = residual_block(x, output_channels=n_channels * 32, stride=2)  # 7x7
    x = residual_block(x, output_channels=n_channels * 32)
    x = residual_block(x, output_channels=n_channels * 32)

    pool_size = (x.get_shape()[1], x.get_shape()[2])
    x = AveragePooling2D(pool_size=pool_size, strides=(1, 1))(x)
    x = Flatten()(x)
    if dropout:
        x = Dropout(dropout)(x)
    output = Dense(n_classes, kernel_regularizer=regularizer, activation='softmax')(x)

    model = Model(input_, output)
    return model


def AttentionResNetCifar10(shape=(32, 32, 3), n_channels=32, n_classes=10):
    """
    Attention-56 ResNet for Cifar10 Dataset
    https://arxiv.org/abs/1704.06904
    """
    input_ = Input(shape=shape)
    x = Conv2D(n_channels, (5, 5), padding='same')(input_)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPool2D(pool_size=(2, 2))(x)  # 16x16

    x = residual_block(x, input_channels=32, output_channels=128)
    x = attention_block(x, encoder_depth=2)

    x = residual_block(x, input_channels=128, output_channels=256, stride=2)  # 8x8
    x = attention_block(x, encoder_depth=1)

    x = residual_block(x, input_channels=256, output_channels=512, stride=2)  # 4x4
    x = attention_block(x, encoder_depth=1)

    x = residual_block(x, input_channels=512, output_channels=1024)
    x = residual_block(x, input_channels=1024, output_channels=1024)
    x = residual_block(x, input_channels=1024, output_channels=1024)

    x = AveragePooling2D(pool_size=(4, 4), strides=(1, 1))(x)  # 1x1
    x = Flatten()(x)
    output = Dense(n_classes, activation='softmax')(x)

    model = Model(input_, output)
    return model

    
def residual_block(input, input_channels=None, output_channels=None, kernel_size=(3, 3), stride=1):
    """
    full pre-activation residual block
    https://arxiv.org/pdf/1603.05027.pdf
    """
    if output_channels is None:
        output_channels = input.get_shape()[-1]
    if input_channels is None:
        input_channels = output_channels // 4

    strides = (stride, stride)

    x = BatchNormalization()(input)
    x = Activation('relu')(x)
    x = Conv2D(input_channels, (1, 1))(x)

    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(input_channels, kernel_size, padding='same', strides=stride)(x)

    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(output_channels, (1, 1), padding='same')(x)

    if input_channels != output_channels or stride != 1:
        input = Conv2D(output_channels, (1, 1), padding='same', strides=strides)(input)

    x = Add()([x, input])
    return x


def attention_block(input, input_channels=None, output_channels=None, encoder_depth=1):
    """
    attention block
    https://arxiv.org/abs/1704.06904
    """

    p = 1
    t = 2
    r = 1

    if input_channels is None:
        input_channels = input.get_shape()[-1]
    if output_channels is None:
        output_channels = input_channels

    # First Residual Block
    for i in range(p):
        input = residual_block(input)

    # Trunc Branch
    output_trunk = input
    for i in range(t):
        output_trunk = residual_block(output_trunk)

    # Soft Mask Branch

    ## encoder
    ### first down sampling
    output_soft_mask = MaxPool2D(padding='same')(input)  # 32x32
    for i in range(r):
        output_soft_mask = residual_block(output_soft_mask)

    skip_connections = []
    for i in range(encoder_depth - 1):

        ## skip connections
        output_skip_connection = residual_block(output_soft_mask)
        skip_connections.append(output_skip_connection)
        # print ('skip shape:', output_skip_connection.get_shape())

        ## down sampling
        output_soft_mask = MaxPool2D(padding='same')(output_soft_mask)
        for _ in range(r):
            output_soft_mask = residual_block(output_soft_mask)

            ## decoder
    skip_connections = list(reversed(skip_connections))
    for i in range(encoder_depth - 1):
        ## upsampling
        for _ in range(r):
            output_soft_mask = residual_block(output_soft_mask)
        output_soft_mask = UpSampling2D()(output_soft_mask)
        ## skip connections
        output_soft_mask = Add()([output_soft_mask, skip_connections[i]])

    ### last upsampling
    for i in range(r):
        output_soft_mask = residual_block(output_soft_mask)
    output_soft_mask = UpSampling2D()(output_soft_mask)

    ## Output
    output_soft_mask = Conv2D(input_channels, (1, 1))(output_soft_mask)
    output_soft_mask = Conv2D(input_channels, (1, 1))(output_soft_mask)
    output_soft_mask = Activation('sigmoid')(output_soft_mask)

    # Attention: (1 + output_soft_mask) * output_trunk
    output = Lambda(lambda x: x + 1)(output_soft_mask)
    output = Multiply()([output, output_trunk])  #

    # Last Residual Block
    for i in range(p):
        output = residual_block(output)

    return output
    
def NEW_SQUEEZE(size, include_top=True, pooling='max', classes=2):
    

    inp = Input(size)
    x = Convolution2D(64, (3, 3), strides=(2, 2), padding='valid', name='conv1')(inp)
    x = Activation('relu', name='relu_conv1')(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool1')(x)

    y = fire_module(x, fire_id=2, squeeze=16, expand=64)
    z = fire_module(x, fire_id=3, squeeze=16, expand=64)
    x = concatenate([y, z])
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool3')(x)

    y = fire_module(x, fire_id=4, squeeze=32, expand=128)
    z = fire_module(x, fire_id=5, squeeze=32, expand=128)
    x = concatenate([y,z])
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool5')(x)

    y = fire_module(x, fire_id=6, squeeze=48, expand=192)
    z = fire_module(x, fire_id=7, squeeze=48, expand=192)
    x = concatenate([y,z])
    x = fire_module(x, fire_id=8, squeeze=64, expand=256)
    x = fire_module(x, fire_id=9, squeeze=64, expand=256)
    
    if include_top:
        # It's not obvious where to cut the network... 
        # Could do the 8th or 9th layer... some work recommends cutting earlier layers.
    
        x = Dropout(0.5, name='drop9')(x)

        x = Convolution2D(classes, (1, 1), padding='valid', name='conv10')(x)
        x = Activation('relu', name='relu_conv10')(x)
        x = GlobalAveragePooling2D()(x)
        x = Activation('softmax', name='loss')(x)
    else:
        if pooling == 'avg':
            x = GlobalAveragePooling2D()(x)
        elif pooling=='max':
            x = GlobalMaxPooling2D()(x)
        elif pooling==None:
            pass
        else:
            raise ValueError("Unknown argument for 'pooling'=" + pooling)
    x = Flatten()(x)
    x = Dense(100)(x)
    x = Activation("relu")(x)
    x = BatchNormalization()(x)

    x = Dense(classes)(x)
    x = Activation("softmax")(x)
    
    model = tf.keras.Model(inputs=inp, outputs = x, name='squeezenet')
    return model
    
def se_block(in_block, ch, ratio=16):
    x = GlobalAveragePooling2D()(in_block)
    x = Dense(ch//ratio, activation='relu')(x)
    x = Dense(ch, activation='sigmoid')(x)
    return tf.sigmoid(x) * in_block 

def BASE_BLOCK(x, filters):
    x = Conv2D(filters, (7, 7), strides=(2, 2), padding = 'same')(x)
    x = BatchNormalization(axis=-1)(x)
    x = Activation('gelu')(x)
    return x

def CONV_STEP(y, filters):
    x = Conv2D(filters, (1, 1), strides=(2, 2), padding = 'same')(y)
    x = BatchNormalization(axis=-1)(x)
    y = Activation('gelu')(x)
    x = Conv2D(filters, (1, 1), padding = 'same')(y)
    x = BatchNormalization(axis=-1)(x)
    x = Activation('gelu')(x)
    return Add()([y,x])

def MBCONV(y, filters, stride):
    x = Conv2D(filters, kernel_size = (1, 1), padding = 'same')(y)
    x = BatchNormalization()(x)
    x = Activation('gelu')(x)
    x = DepthwiseConv2D(kernel_size = (3,3), padding = "same")(x)
    x = BatchNormalization()(x)
    x = Activation('gelu')(x)
    z = se_block(x, filters, ratio=16)
    x = Conv2D(filters, kernel_size = (1, 1), padding = 'same')(x)
    x = BatchNormalization()(x)
    x = Activation('gelu')(x)
    out = Add()([y,x,z])
    return out

def MBCONV_FUSED(y, filters, stride):
    x = Conv2D(filters, kernel_size = (3, 3), padding = 'same')(y)
    x = BatchNormalization()(x)
    x = Activation('gelu')(x)
    z = se_block(x, filters, ratio=16)
    x = Conv2D(filters, kernel_size = (1, 1), padding = 'same')(x)
    x = BatchNormalization()(x)
    x = Activation('gelu')(x)
    out = Add()([y,x,z])
    return out
    
    

def OKO_NET(size, classes = 2, filters=32, drop_out = 0.2):
    inp = Input(size)
    x = BASE_BLOCK(inp, filters)
    x = MaxPooling2D(pool_size = (1, 1), strides=(1, 1), padding='same')(x)
    for ii in range(1, 3):
        x = MBCONV_FUSED(x, filters, 1)
    for ii in range(1, 3):
        x = CONV_STEP(x, ii * filters)
        x = MBCONV(x, ii *filters, 1)
    x = AveragePooling2D((3, 3))(x)
    
    x = Flatten()(x)
    x = Dense(100)(x)
    x = BatchNormalization()(x)
    x = Activation('gelu')(x)
    x = Dense(classes)(x)
    out = Activation("softmax")(x)
    
    model = tf.keras.Model(inputs=inp, outputs = out)

    return model
def semantic_model(): 
    base = keras.applications.DenseNet121(input_shape=[256,256,3], include_top=False, weights='imagenet')
    
    
    
    skip_names = ['conv1/relu', # size 64*64
                  'pool2_relu',  # size 32*32
                  'pool3_relu',  # size 16*16
                  'pool4_relu',  'relu'] 
    
    skip_outputs = [base.get_layer(name).output for name in skip_names]
    
        
        
        
    downstack = keras.Model(inputs=base.input,
                            outputs=skip_outputs)
    #downstack.trainable = False 
    
    # Four upstack blocks for upsampling sizes 
    # 4->8, 8->16, 16->32, 32->64 
    upstack = [pix2pix.upsample(512,3),
               pix2pix.upsample(256,3),
               pix2pix.upsample(128,3),
               pix2pix.upsample(64,3)] 
    #We can explore the individual layers in each upstack block.
    
    upstack[0].layers
    
    inputs = keras.layers.Input(shape=[256,256,3])
    # downsample 
    down = downstack(inputs)
    out = down[-1]
    # prepare skip-connections
    skips = reversed(down[:-1])
    # choose the last layer at first 4 --> 8
    # upsample with skip-connections
    for up, skip in zip(upstack,skips):
        out = up(out)
        out = keras.layers.Concatenate()([out,skip])
    # define the final transpose conv layer
    # image 128 by 128 with 59 classes
    out = keras.layers.Conv2DTranspose(8, 3,
                                       strides=2,
                                       padding='same',
                                       )(out)
     # complete unet model
    unet = keras.Model(inputs=inputs, outputs=out)
    return unet
    
def semantic_model2(): 
    base = keras.applications.EfficientNetB7(input_shape=[256,256,3], include_top=False, weights=None)
    base.load_weights("WIEGHTS/256_eff_base.h5")
    
    
    skip_names = ['block1a_activation', # size 64*64
                  'block2g_activation',  # size 32*32
                  'block3g_activation',  # size 16*16
                  'block5g_activation',  'top_activation'] 
    
    skip_outputs = [base.get_layer(name).output for name in skip_names]
    
        
        
        
    downstack = keras.Model(inputs=base.input,
                            outputs=skip_outputs)
    downstack.trainable = False 
    
    # Four upstack blocks for upsampling sizes 
    # 4->8, 8->16, 16->32, 32->64 
    upstack = [pix2pix.upsample(512,3),
               pix2pix.upsample(256,3),
               pix2pix.upsample(128,3),
               pix2pix.upsample(64,3)] 
    #We can explore the individual layers in each upstack block.
    
    upstack[0].layers
    
    inputs = keras.layers.Input(shape=[256,256,3])
    # downsample 
    down = downstack(inputs)
    out = down[-1]
    # prepare skip-connections
    skips = reversed(down[:-1])
    # choose the last layer at first 4 --> 8
    # upsample with skip-connections
    for up, skip in zip(upstack,skips):
        out = up(out)
        out = keras.layers.Concatenate()([out,skip])
    # define the final transpose conv layer
    # image 128 by 128 with 59 classes
    out = keras.layers.Conv2DTranspose(8, 3,
                                       strides=2,
                                       padding='same',
                                       )(out)
     # complete unet model
    unet = keras.Model(inputs=inputs, outputs=out)
    return unet

def identity_block2(input_tensor, kernel_size, filters):
    nb_filter1, nb_filter2, nb_filter3 = filters

    x = Conv2D(nb_filter1, (1, 1))(input_tensor)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = ZeroPadding2D((1, 1))(x)
    x = Conv2D(nb_filter2, (kernel_size, kernel_size))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(nb_filter3, (1, 1))(x)
    x = BatchNormalization()(x)
    att = GlobalAveragePooling2D()(x)
    d = Dense(att.shape[1]/16, activation='relu')(att)
    d = Dense(att.shape[1], activation='sigmoid')(d)
    att = tf.keras.layers.Multiply()([x, d])
    x = Add()([att, input_tensor])
    x = Activation('relu')(x)
    return x

def input_block(x, filters=32):
    x = ZeroPadding2D((3, 3))(x)
    x = Conv2D(filters, (7, 7), strides=(2, 2))(x)
    x = BatchNormalization(axis=-1)(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((7, 7), strides=(2, 2))(x)
    return x

def DualResNet50(size, num_grades = 2, filters=32):

    inp = Input(size)
    
    x = input_block(inp)

    
    
    x = conv_block(x, 3, [filters, filters, 4*filters], strides=(1, 1))
    x = identity_block2(x, 3, [filters, filters, 4*filters])
    x = identity_block2(x, 3, [filters, filters, 4*filters])
    

    x = conv_block(x, 3, [2*filters, 2*filters, 8*filters])
    for i in range(0, 3):
        x = identity_block2(x, 3, [2*filters, 2*filters, 8*filters])

    x = conv_block(x, 3, [4*filters, 4*filters, 16*filters])
    for i in range(0, 5):
        x = identity_block2(x, 3, [4*filters, 4*filters, 16*filters])

    x = conv_block(x, 3, [8*filters, 8*filters, 32*filters])
    x = identity_block2(x, 3, [8*filters, 8*filters, 32*filters])
    x = identity_block2(x, 3, [8*filters, 8*filters, 32*filters]) #2048
    x = AveragePooling2D((7, 7),name="outlayer")(x)

    x = Flatten()(x)

    x = Dense(num_grades)(x)
    x = Activation("softmax")(x)

    model = tf.keras.Model(inputs=inp, outputs = x)
    return model