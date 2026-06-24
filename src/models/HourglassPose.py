# -*- coding: utf-8 -*-
"""
Created on Thu Jul  4 23:01:54 2019

@author: Gabriel
"""
from keras.layers.convolutional import MaxPooling2D, Convolution2D, AveragePooling2D
from keras.layers import Input, Conv2DTranspose,ZeroPadding2D,Dropout, Dense, Flatten, Activation,GlobalAveragePooling2D,GlobalMaxPooling2D,Conv2D
from keras.layers.normalization import BatchNormalization
from keras.layers.merge import concatenate,add
from keras import regularizers
from keras import initializers
from keras.models import Model
from keras.regularizers import l1,l2
from keras import backend as K
from keras.utils import Sequence,plot_model
from keras import utils

import os
import warnings

def identity_block(input_tensor, kernel_size, filters, stage, block):
    """The identity block is the block that has no conv layer at shortcut.

    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of
            middle conv layer at main path
        filters: list of integers, the filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names

    # Returns
        Output tensor for the block.
    """
    filters1, filters2 = filters
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(filters1, (1, 1),
                      kernel_initializer='he_normal',
                      name=conv_name_base + '2a')(input_tensor)
    x = Activation('relu')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)

    x = Conv2D(filters2, kernel_size,
                      padding='same',
                      kernel_initializer='he_normal',
                      name=conv_name_base + '2b')(x)
    x = Activation('relu')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)


    x = add([x, input_tensor])
    #x = Activation('relu')(x)
    return x


def conv_block(input_tensor,
               kernel_size,
               filters,
               stage,
               block,
               strides=(2, 2)):
    """A block that has a conv layer at shortcut.

    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of
            middle conv layer at main path
        filters: list of integers, the filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
        strides: Strides for the first conv layer in the block.

    # Returns
        Output tensor for the block.

    Note that from stage 3,
    the first conv layer at main path is with strides=(2, 2)
    And the shortcut should have strides=(2, 2) as well
    """
    filters1, filters2 = filters
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(filters1, (1, 1), strides=strides,
                      kernel_initializer='he_normal',
                      name=conv_name_base + '2a')(input_tensor)
    
    x = Activation('relu')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)

    x = Conv2D(filters2, kernel_size, padding='same',
                      kernel_initializer='he_normal',
                      name=conv_name_base + '2b')(x)
    
    x = Activation('relu')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)

    #not sure about ordering here
    shortcut = Conv2D(filters2, (1, 1), strides=strides,
                             kernel_initializer='he_normal',
                             name=conv_name_base + '1')(input_tensor)
    
    x = Activation('relu')(x)
    shortcut = BatchNormalization(axis=bn_axis, name=bn_name_base + '1')(shortcut)

    x = add([x, shortcut])
    return x


def HourGlassPose(include_top=True,
             weights='imagenet',
             input_tensor=None,
             input_shape=None,
             pooling=None,
             classes=1000):
    """Instantiates the ResNet50 architecture.

    Optionally loads weights pre-trained on ImageNet.
    Note that the data format convention used by the model is
    the one specified in your Keras config at `~/.keras/keras.json`.

    # Arguments
        include_top: whether to include the fully-connected
            layer at the top of the network.
        weights: one of `None` (random initialization),
              'imagenet' (pre-training on ImageNet),
              or the path to the weights file to be loaded.
        input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
            to use as image input for the model.
        input_shape: optional shape tuple, only to be specified
            if `include_top` is False (otherwise the input shape
            has to be `(224, 224, 3)` (with `channels_last` data format)
            or `(3, 224, 224)` (with `channels_first` data format).
            It should have exactly 3 inputs channels,
            and width and height should be no smaller than 32.
            E.g. `(200, 200, 3)` would be one valid value.
        pooling: Optional pooling mode for feature extraction
            when `include_top` is `False`.
            - `None` means that the output of the model will be
                the 4D tensor output of the
                last convolutional block.
            - `avg` means that global average pooling
                will be applied to the output of the
                last convolutional block, and thus
                the output of the model will be a 2D tensor.
            - `max` means that global max pooling will
                be applied.
        classes: optional number of classes to classify images
            into, only to be specified if `include_top` is True, and
            if no `weights` argument is specified.

    # Returns
        A Keras model instance.

    # Raises
        ValueError: in case of invalid argument for `weights`,
            or invalid input shape.
    """
#    global backend, layers, models, keras_utils
#    backend, layers, models, keras_utils = get_submodules_from_kwargs(kwargs)

    if not (weights in {'imagenet', None} or os.path.exists(weights)):
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization), `imagenet` '
                         '(pre-training on ImageNet), '
                         'or the path to the weights file to be loaded.')

    if weights == 'imagenet' and include_top and classes != 1000:
        raise ValueError('If using `weights` as `"imagenet"` with `include_top`'
                         ' as true, `classes` should be 1000')

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor
            
            
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1

    
    #print(bn_axis)
    #x = ZeroPadding2D(padding=(3, 3), name='conv1_pad')(img_input)
    x = Conv2D(64, (7, 7),strides=(2, 2),
                      padding='same',
                      kernel_initializer='he_normal',
                      name='conv1')(img_input)
    
    #3 Lines below to be replaced by
    x = Activation('relu')(x)
    x = BatchNormalization(axis=bn_axis, name='bn_conv1')(x)    
    
    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)#ok

    ####Encoder
    x = conv_block(x, 3, [64, 64], stage=2, block='a', strides=(1, 1))
    x = identity_block(x, 3, [64, 64], stage=2, block='b')
    x1 = identity_block(x, 3, [64, 64], stage=2, block='c')

    x = conv_block(x1, 3, [128, 128], stage=3, block='a')
    x = identity_block(x, 3, [128, 128], stage=3, block='b')
    x = identity_block(x, 3, [128, 128], stage=3, block='c')
    x2 = identity_block(x, 3, [128, 128], stage=3, block='d')

    x = conv_block(x2, 3, [256, 256], stage=4, block='a')
    x = identity_block(x, 3, [256, 256], stage=4, block='b')
    x = identity_block(x, 3, [256, 256], stage=4, block='c')
    x = identity_block(x, 3, [256, 256], stage=4, block='d')
    x = identity_block(x, 3, [256, 256], stage=4, block='e')
    x3 = identity_block(x, 3, [256, 256], stage=4, block='f')

    x = conv_block(x3, 3, [512, 512], stage=5, block='a')
    x = identity_block(x, 3, [512, 512], stage=5, block='b')
    x = identity_block(x, 3, [512, 512], stage=5, block='c')
    
    #x = ZeroPadding2D(padding=(7, 7), data_format='channels_last')(x)
    x=Conv2DTranspose(256, kernel_size=4, strides=(2, 2), padding='same', output_padding=None, data_format=None, dilation_rate=(1, 1))(x)
    x = Activation('relu')(x)
    x = BatchNormalization(axis=bn_axis)(x)
    x = add([x, x3])
    x=Conv2DTranspose(128, kernel_size=4, strides=(2, 2), padding='same', output_padding=None, data_format=None, dilation_rate=(1, 1))(x)
    x = Activation('relu')(x)
    x = BatchNormalization(axis=bn_axis)(x)
    x = add([x, x2])
    x=Conv2DTranspose(64, kernel_size=4, strides=(2, 2), padding='same', output_padding=None, data_format=None, dilation_rate=(1, 1))(x)
    x = Activation('relu')(x)
    x = BatchNormalization(axis=bn_axis)(x)
    x = add([x, x1])
    x = Conv2D(8, (3, 3),strides=(1, 1),padding='same')(x)

#    x = Conv2D(256, (4, 4),strides=(2, 2),
#                  padding='same',
#                  kernel_initializer='he_normal',
#                  name='upconv1')(x)
#    x = Activation('relu')(x)
#    x = BatchNormalization(axis=bn_axis)(x)
    ####Decoder
    
#
    if include_top:
        x = GlobalAveragePooling2D(name='avg_pool')(x)
        x = Dense(classes, activation='softmax', name='fc1000')(x)
    else:
        if pooling == 'avg':
            x = GlobalAveragePooling2D()(x)
        elif pooling == 'max':
            x = GlobalMaxPooling2D()(x)
        else:
            pass
#            warnings.warn('The output shape of `ResNet34(include_top=False)` '
#                          'has been changed since Keras 2.2.0.')

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = utils.get_source_inputs(input_tensor)
    else:
        inputs = img_input
    # Create model.
    model = Model(inputs, x, name='HourglassPose')

#    # Load weights.
#    if weights == 'imagenet':
#        if include_top:
#            weights_path = keras_utils.get_file(
#                'resnet50_weights_tf_dim_ordering_tf_kernels.h5',
#                WEIGHTS_PATH,
#                cache_subdir='models',
#                md5_hash='a7b3fe01876f51b976af0dea6bc144eb')
#        else:
#            weights_path = keras_utils.get_file(
#                'resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5',
#                WEIGHTS_PATH_NO_TOP,
#                cache_subdir='models',
#                md5_hash='a268eb855778b3df3c7506639542a6af')
#        model.load_weights(weights_path)
#        if backend.backend() == 'theano':
#            keras_utils.convert_all_kernels_in_model(model)
#    elif weights is not None:
#        model.load_weights(weights)

    return model  

#model=HourGlassPose(include_top=False,input_shape=(320,320,3))
from classification_models.resnet import ResNet34

def GetHourGlassPose(include_top=False,
             weights='imagenet',
             input_tensor=None,
             input_shape=None,
             pooling=None,
             classes=1000):
    
    model = ResNet34(input_shape=input_shape, weights=weights,include_top=False)
    #model.summary()
    
    #layer_name = 'my_layer'
    #
    #stage2_unit1_relu1
    #stage3_unit1_relu1
    #stage4_unit1_relu1
    x = model.get_layer('relu1').output   
    x=Conv2DTranspose(256, kernel_size=4, strides=(2, 2), padding='same', output_padding=None, data_format=None, dilation_rate=(1, 1))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)  
    x3 = model.get_layer('stage4_unit1_relu1').output
    
    x = add([x, x3])
    x=Conv2DTranspose(128, kernel_size=4, strides=(2, 2), padding='same', output_padding=None, data_format=None, dilation_rate=(1, 1))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    x2 = model.get_layer('stage3_unit1_relu1').output   
    x = add([x, x2])
    x=Conv2DTranspose(64, kernel_size=4, strides=(2, 2), padding='same', output_padding=None, data_format=None, dilation_rate=(1, 1))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    x1 = model.get_layer('stage2_unit1_relu1').output    
    x = add([x, x1])
    x = Conv2D(32, (3, 3),strides=(1, 1),padding='same')(x)
    model = Model(model.input, x, name='HourglassPose')
    
    return model

#plot_model(model, to_file='Resnet34-Posetesting.png', show_shapes=True, show_layer_names=True)