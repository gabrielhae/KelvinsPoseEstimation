# -*- coding: utf-8 -*-

import tensorflow as tf
from keras import layers
from keras.models import Model
from keras.layers import Input, Multiply,GlobalAveragePooling2D, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D, Lambda
#import pydot
#from resnets_utils import *
from keras.initializers import glorot_uniform
# %matplotlib inline

import keras.backend as K


import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

def res_conv(X, filters, base, s):
    
    name_base = base + '/branch'
    
    F1, F2, F3 = filters

    ##### Branch1 is the main path and Branch2 is the shortcut path #####
    
    X_shortcut = X
    
    ##### Branch1 #####
    # First component of Branch1 
    X = BatchNormalization(axis=-1, name=name_base + '1/bn_1')(X)
    X= Activation('relu', name=name_base + '1/relu_1')(X)
    X = Conv2D(filters=F1, kernel_size=(1,1), strides=(1,1), padding='valid', name=name_base + '1/conv_1', kernel_initializer=glorot_uniform(seed=0))(X)

    # Second component of Branch1
    X = BatchNormalization(axis=-1, name=name_base + '1/bn_2')(X)
    X = Activation('relu', name=name_base + '1/relu_2')(X)
    X = Conv2D(filters=F2, kernel_size=(3,3), strides=(s,s), padding='same', name=name_base + '1/conv_2', kernel_initializer=glorot_uniform(seed=0))(X)
    
    # Third component of Branch1
    X = BatchNormalization(axis=-1, name=name_base + '1/bn_3')(X)
    X = Activation('relu', name=name_base + '1/relu_3')(X)
    X = Conv2D(filters=F3, kernel_size=(1,1), strides=(1,1), padding='valid', name=name_base + '1/conv_3', kernel_initializer=glorot_uniform(seed=0))(X)
    
    ##### Branch2 ####
    X_shortcut = BatchNormalization(axis=-1, name=name_base + '2/bn_1')(X_shortcut)
    X_shortcut= Activation('relu', name=name_base + '2/relu_1')(X_shortcut)
    X_shortcut = Conv2D(filters=F3, kernel_size=(1,1), strides=(s,s), padding='valid', name=name_base + '2/conv_1', kernel_initializer=glorot_uniform(seed=0))(X_shortcut)
    
    # Final step: Add Branch1 and Branch2
    X = Add(name=base + '/Add')([X, X_shortcut])

    return X

def res_identity(X, filters, base):
    
    name_base = base + '/branch'
    
    F1, F2, F3 = filters

    ##### Branch1 is the main path and Branch2 is the shortcut path #####
    
    X_shortcut = X
    
    ##### Branch1 #####
    # First component of Branch1 
    X = BatchNormalization(axis=-1, name=name_base + '1/bn_1')(X)
    Shortcut= Activation('relu', name=name_base + '1/relu_1')(X)
    X = Conv2D(filters=F1, kernel_size=(1,1), strides=(1,1), padding='valid', name=name_base + '1/conv_1', kernel_initializer=glorot_uniform(seed=0))(Shortcut)

    # Second component of Branch1
    X = BatchNormalization(axis=-1, name=name_base + '1/bn_2')(X)
    X = Activation('relu', name=name_base + '1/relu_2')(X)
    X = Conv2D(filters=F2, kernel_size=(3,3), strides=(1,1), padding='same', name=name_base + '1/conv_2', kernel_initializer=glorot_uniform(seed=0))(X)
    
    # Third component of Branch1
    X = BatchNormalization(axis=-1, name=name_base + '1/bn_3')(X)
    X = Activation('relu', name=name_base + '1/relu_3')(X)
    X = Conv2D(filters=F3, kernel_size=(1,1), strides=(1,1), padding='valid', name=name_base + '1/conv_3', kernel_initializer=glorot_uniform(seed=0))(X)    
    
    # Final step: Add Branch1 and the original Input itself
    X = Add(name=base + '/Add')([X, X_shortcut])

    return X

def Trunk_block(X, F, base):
    
    name_base = base
    
    X = res_identity(X, F, name_base + '/Residual_id_1')
    X = res_identity(X, F, name_base + '/Residual_id_2')
    
    return X

def interpolation(input_tensor, ref_tensor,name): # resizes input_tensor wrt. ref_tensor
    H, W = ref_tensor.get_shape()[1], ref_tensor.get_shape()[2]
    return tf.image.resize_nearest_neighbor(input_tensor, [H.value, W.value],name=name)

def Attention_1(X, filters, base):
    
    F1, F2, F3 = filters
    
    name_base = base
    
    X = res_identity(X, filters, name_base+ '/Pre_Residual_id')
    
    X_Trunk = Trunk_block(X, filters, name_base+ '/Trunk')
    
    X = MaxPooling2D((3,3), strides=(2,2), padding='same', name=name_base+ '/Mask/pool_3')(X)
    
    X = res_identity(X, filters, name_base+ '/Mask/Residual_id_3_Down')
    
    Residual_id_3_Down_shortcut = X
    
    Residual_id_3_Down_branched = res_identity(X, filters, name_base+ '/Mask/Residual_id_3_Down_branched')
    
    X = MaxPooling2D((3,3), strides=(2,2), padding='same', name=name_base+ '/Mask/pool_2')(X)
    
    X = res_identity(X, filters, name_base+ '/Mask/Residual_id_2_Down')
    
    Residual_id_2_Down_shortcut = X
    
    Residual_id_2_Down_branched = res_identity(X, filters, name_base+ '/Mask/Residual_id_2_Down_branched')
    
    X = MaxPooling2D((3,3), strides=(2,2), padding='same', name=name_base+ '/Mask/pool_1')(X)
    
    X = res_identity(X, filters, name_base+ '/Mask/Residual_id_1_Down')
    
    X = res_identity(X, filters, name_base+ '/Mask/Residual_id_1_Up')
    
    temp_name1 = name_base+ "/Mask/Interpool_1"
    
    X = Lambda(interpolation, arguments={'ref_tensor': Residual_id_2_Down_shortcut,'name':temp_name1})(X)
                                          
    X = Add(name=base + '/Mask/Add_after_Interpool_1')([X, Residual_id_2_Down_branched])
                                          
    X = res_identity(X, filters, name_base+ '/Mask/Residual_id_2_Up')
    
    temp_name2 = name_base+ "/Mask/Interpool_2"
    
    X = Lambda(interpolation, arguments={'ref_tensor': Residual_id_3_Down_shortcut,'name':temp_name2})(X)
                                          
    X = Add(name=base + '/Mask/Add_after_Interpool_2')([X, Residual_id_3_Down_branched])
                                          
    X = res_identity(X, filters, name_base+ '/Mask/Residual_id_3_Up')
    
    temp_name3 = name_base+ "/Mask/Interpool_3"
    
    X = Lambda(interpolation, arguments={'ref_tensor': X_Trunk,'name':temp_name3})(X)
                                          
    X = BatchNormalization(axis=-1, name=name_base + '/Mask/Interpool_3/bn_1')(X)
                                          
    X = Activation('relu', name=name_base + '/Mask/Interpool_3/relu_1')(X)
                                          
    X = Conv2D(F3, kernel_size=(1,1), strides=(1,1), padding='valid', name=name_base + '/Mask/Interpool_3/conv_1', kernel_initializer=glorot_uniform(seed=0))(X)
    
    X = BatchNormalization(axis=-1, name=name_base + '/Mask/Interpool_3/bn_2')(X)
                                          
    X = Activation('relu', name=name_base + '/Mask/Interpool_3/relu_2')(X)
                                          
    X = Conv2D(F3, kernel_size=(1,1), strides=(1,1), padding='valid', name=name_base + '/Mask/Interpool_3/conv_2', kernel_initializer=glorot_uniform(seed=0))(X)
    
    X = Activation('sigmoid', name=name_base+'/Mask/sigmoid')(X)
      
    X = Multiply(name=name_base+'/Mutiply')([X_Trunk,X])
    
    X = Add(name=name_base+'/Add')([X_Trunk,X])

    X = res_identity(X, filters, name_base+ '/Post_Residual_id')
    
    return X

def Attention_2(X, filters, base):
    
    F1, F2, F3 = filters
    
    name_base = base
    
    X = res_identity(X, filters, name_base+ '/Pre_Residual_id')
    
    X_Trunk = Trunk_block(X, filters, name_base+ '/Trunk')
    
    X = MaxPooling2D((3,3), strides=(2,2), padding='same', name=name_base+ '/Mask/pool_2')(X)
    
    X = res_identity(X, filters, name_base+ '/Mask/Residual_id_2_Down')
    
    Residual_id_2_Down_shortcut = X
    
    Residual_id_2_Down_branched = res_identity(X, filters, name_base+ '/Mask/Residual_id_2_Down_branched')
    
    X = MaxPooling2D((3,3), strides=(2,2), padding='same', name=name_base+ '/Mask/pool_1')(X)
    
    X = res_identity(X, filters, name_base+ '/Mask/Residual_id_1_Down')
                                          
    X = res_identity(X, filters, name_base+ '/Mask/Residual_id_1_Up')
    
    temp_name1 = name_base+ "/Mask/Interpool_1"
    
    X = Lambda(interpolation, arguments={'ref_tensor': Residual_id_2_Down_shortcut,'name':temp_name1})(X)
                                          
    X = Add(name=base + '/Mask/Add_after_Interpool_1')([X, Residual_id_2_Down_branched])
                                          
    X = res_identity(X, filters, name_base+ '/Mask/Residual_id_2_Up')
    
    temp_name2 = name_base+ "/Mask/Interpool_2"
    
    X = Lambda(interpolation, arguments={'ref_tensor': X_Trunk,'name':temp_name2})(X)
                                          
    X = BatchNormalization(axis=-1, name=name_base + '/Mask/Interpool_2/bn_1')(X)
                                          
    X = Activation('relu', name=name_base + '/Mask/Interpool_2/relu_1')(X)
                                          
    X = Conv2D(F3, kernel_size=(1,1), strides=(1,1), padding='valid', name=name_base + '/Mask/Interpool_2/conv_1', kernel_initializer=glorot_uniform(seed=0))(X)
    
    X = BatchNormalization(axis=-1, name=name_base + '/Mask/Interpool_2/bn_2')(X)
                                          
    X = Activation('relu', name=name_base + '/Mask/Interpool_2/relu_2')(X)
                                          
    X = Conv2D(F3, kernel_size=(1,1), strides=(1,1), padding='valid', name=name_base + '/Mask/Interpool_2/conv_2', kernel_initializer=glorot_uniform(seed=0))(X)
    
    X = Activation('sigmoid', name=name_base+'/Mask/sigmoid')(X)
      
    X = Multiply(name=name_base+'/Mutiply')([X_Trunk,X])
    
    X = Add(name=name_base+'/Add')([X_Trunk,X])

    X = res_identity(X, filters, name_base+ '/Post_Residual_id')
    
    return X

def Attention_3(X, filters, base):
    
    F1, F2, F3 = filters
    
    name_base = base
    
    X = res_identity(X, filters, name_base+ '/Pre_Residual_id')
    
    X_Trunk = Trunk_block(X, filters, name_base+ '/Trunk')
    
    X = MaxPooling2D((3,3), strides=(2,2), padding='same', name=name_base+ '/Mask/pool_1')(X)
    
    X = res_identity(X, filters, name_base+ '/Mask/Residual_id_1_Down')
                                          
    X = res_identity(X, filters, name_base+ '/Mask/Residual_id_1_Up')
    
    temp_name2 = name_base+ "/Mask/Interpool_1"
    
    X = Lambda(interpolation, arguments={'ref_tensor': X_Trunk,'name':temp_name2})(X)
                                          
    X = BatchNormalization(axis=-1, name=name_base + '/Mask/Interpool_2/bn_1')(X)
                                          
    X = Activation('relu', name=name_base + '/Mask/Interpool_2/relu_1')(X)
                                          
    X = Conv2D(F3, kernel_size=(1,1), strides=(1,1), padding='valid', name=name_base + '/Mask/Interpool_2/conv_1', kernel_initializer=glorot_uniform(seed=0))(X)
    
    X = BatchNormalization(axis=-1, name=name_base + '/Mask/Interpool_2/bn_2')(X)
                                          
    X = Activation('relu', name=name_base + '/Mask/Interpool_2/relu_2')(X)
                                          
    X = Conv2D(F3, kernel_size=(1,1), strides=(1,1), padding='valid', name=name_base + '/Mask/Interpool_2/conv_2', kernel_initializer=glorot_uniform(seed=0))(X)
    
    X = Activation('sigmoid', name=name_base+'/Mask/sigmoid')(X)
      
    X = Multiply(name=name_base+'/Mutiply')([X_Trunk,X])
    
    X = Add(name=name_base+'/Add')([X_Trunk,X])

    X = res_identity(X, filters, name_base+ '/Post_Residual_id')
    
    return X

#input_shape=(224, 224, 3)
#X_input = Input(input_shape)
#
#X = Conv2D(64, (7,7), strides=(2,2), padding='same', name='conv_1', kernel_initializer=glorot_uniform(seed=0))(X_input)
#X = BatchNormalization(axis=-1, name='bn_1')(X)
#X = Activation('relu', name='relu_1')(X)
#X = MaxPooling2D((3,3), strides=(2,2), padding='same' ,name='pool_1')(X)
#X = res_conv(X, [64,64,256], 'Residual_conv_1', 1)
#
#### Attention 1 Start
#X = Attention_1(X, [64,64,256], 'Attention_1')
#### Attention 1 End
#
#X = res_conv(X, [128,128,512], 'Residual_conv_2', 2)
#
#### Attention 2 Start
#X = Attention_2(X, [128,128,512], 'Attention_2')
#### Attention 2 End
#
#X = res_conv(X, [256,256,1024], 'Residual_conv_3', 2)
#
#### Attention 3 Start
#X = Attention_3(X, [256,256,1024], 'Attention_3')
#### Attention 3 End
#
#X = res_conv(X, [512,512,2048], 'Residual_conv_4', 2)
#
#X = res_identity(X, [512,512,2048], 'Residual_id_1')
#X = res_identity(X, [512,512,2048], 'Residual_id_2')
#X = BatchNormalization(axis=-1, name='bn_2')(X)
#X = Activation('relu', name='relu_2')(X)
#
#
##X = AveragePooling2D((7,7), strides=(1,1), name='avg_pool')(X)
##X = Dense(1000, name='Dense_1')(X)
##X = Activation('softmax', name='classifier')(X)
#
#
#model = Model(inputs=X_input, outputs=X, name='attention_56')
#
#model.summary()

def Attention56(include_top=True,
             input_tensor=None,
             input_shape=None,
             pooling=None,
             classes=1000):

#    global backend, layers, models, keras_utils
#    backend, layers, models, keras_utils = get_submodules_from_kwargs(kwargs)
#
#    if not (weights in {'imagenet', None} or os.path.exists(weights)):
#        raise ValueError('The `weights` argument should be either '
#                         '`None` (random initialization), `imagenet` '
#                         '(pre-training on ImageNet), '
#                         'or the path to the weights file to be loaded.')
#
#    if weights == 'imagenet' and include_top and classes != 1000:
#        raise ValueError('If using `weights` as `"imagenet"` with `include_top`'
#                         ' as true, `classes` should be 1000')
#
#    # Determine proper input shape
#    input_shape = _obtain_input_shape(input_shape,
#                                      default_size=224,
#                                      min_size=32,
#                                      data_format=backend.image_data_format(),
#                                      require_flatten=include_top,
#                                      weights=weights)

    if input_tensor is None:
        img_input = layers.Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = layers.Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor
#    if K.image_data_format() == 'channels_last':
#        bn_axis = 3
#    else:
#        bn_axis = 1
#    
    x = Conv2D(64, (7,7), strides=(2,2), padding='same', name='conv_1', kernel_initializer=glorot_uniform(seed=0))(img_input)
    x = BatchNormalization(axis=-1, name='bn_1')(x)
    x = Activation('relu', name='relu_1')(x)
    x = MaxPooling2D((3,3), strides=(2,2), padding='same' ,name='pool_1')(x)
    x = res_conv(x, [64,64,256], 'Residual_conv_1', 1)
    
    ### Attention 1 Start
    x = Attention_1(x, [64,64,256], 'Attention_1')
    ### Attention 1 End
    
    x = res_conv(x, [128,128,512], 'Residual_conv_2', 2)
    
    ### Attention 2 Start
    x = Attention_2(x, [128,128,512], 'Attention_2')
    ### Attention 2 End
    
    x = res_conv(x, [256,256,1024], 'Residual_conv_3', 2)
    
    ### Attention 3 Start
    x = Attention_3(x, [256,256,1024], 'Attention_3')
    ### Attention 3 End
    
    x = res_conv(x, [512,512,2048], 'Residual_conv_4', 2)
    
    x = res_identity(x, [512,512,2048], 'Residual_id_1')
    x = res_identity(x, [512,512,2048], 'Residual_id_2')
    x = BatchNormalization(axis=-1, name='bn_2')(x)
    x = Activation('relu', name='relu_2')(x)    

    if include_top:
        #x = layers.GlobalAveragePooling2D(name='avg_pool')(x)
        #x = layers.Dense(classes, activation='softmax', name='fc1000')(x)
        
        x = AveragePooling2D((7,7), strides=(1,1), name='avg_pool')(x)
        x = Dense(classes, name='Dense_1')(x)
        x = Activation('softmax', name='classifier')(x)       
    else:
        if pooling == 'avg':
            x = layers.GlobalAveragePooling2D(name='avg_pool')(x)
        elif pooling == 'max':
            x = layers.GlobalMaxPooling2D(name='max_pool')(x)
        else:
            warnings.warn('The output shape of `ResNet50(include_top=False)` '
                          'has been changed since Keras 2.2.0.')

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = img_input#keras_utils.get_source_inputs(input_tensor)
    else:
        inputs = img_input
    # Create model.
    model = Model(inputs=inputs, outputs=x, name='attention_56')

    return model