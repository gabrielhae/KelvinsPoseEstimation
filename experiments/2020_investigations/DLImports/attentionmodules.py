# -*- coding: utf-8 -*-
"""
Created on Mon Aug 12 21:58:52 2019

@author: Gabriel
"""

#CBAM Attention Block
from keras import backend as K
from keras.layers import Reshape, multiply, Permute, Add,Multiply, Lambda,BatchNormalization,GlobalAveragePooling2D,Dense,GlobalMaxPooling2D,Concatenate,Activation,Conv2D
from keras.activations import sigmoid

def attach_attention_module(net, attention_module,module_id):
  if attention_module == 'se_block': # SE_block
    net = se_block(net)
  elif attention_module == 'cbam_block': # CBAM_block
    net = cbam_block(net,module_id)
  else:
    raise Exception("'{}' is not supported attention module!".format(attention_module))

  return net


#https://www.kaggle.com/c/tgs-salt-identification-challenge/discussion/64367#latest-421628
def AttentionBlock(x,shortcut,i_filters):
    g1 = Conv2D(i_filters,kernel_size = 1)(shortcut) 
    g1 = BatchNormalization()(g1)
    x1 = Conv2D(i_filters,kernel_size = 1)(x) 
    x1 = BatchNormalization()(x1)

    g1_x1 = Add()([g1,x1])
    psi = Activation('relu')(g1_x1)
    psi = Conv2D(1,kernel_size = 1)(psi) 
    psi = BatchNormalization()(psi)
    psi = Activation('sigmoid')(psi)
    x = Multiply()([x,psi])
    return x

def se_block(input_feature, ratio=8):
	"""Contains the implementation of Squeeze-and-Excitation(SE) block.
	As described in https://arxiv.org/abs/1709.01507.
	"""
	
	channel_axis = 1 if K.image_data_format() == "channels_first" else -1
	channel = input_feature._keras_shape[channel_axis]

	se_feature = GlobalAveragePooling2D()(input_feature)
	se_feature = Reshape((1, 1, channel))(se_feature)
	assert se_feature._keras_shape[1:] == (1,1,channel)
	se_feature = Dense(channel // ratio,
					   activation='relu',
					   kernel_initializer='he_normal',
					   use_bias=True,
					   bias_initializer='zeros')(se_feature)
	assert se_feature._keras_shape[1:] == (1,1,channel//ratio)
	se_feature = Dense(channel,
					   activation='sigmoid',
					   kernel_initializer='he_normal',
					   use_bias=True,
					   bias_initializer='zeros')(se_feature)
	assert se_feature._keras_shape[1:] == (1,1,channel)
	if K.image_data_format() == 'channels_first':
		se_feature = Permute((3, 1, 2))(se_feature)

	se_feature = multiply([input_feature, se_feature])
	return se_feature

def cbam_block(cbam_feature,module_id, ratio=8):
	"""Contains the implementation of Convolutional Block Attention Module(CBAM) block.
	As described in https://arxiv.org/abs/1807.06521.
	"""
	
	cbam_feature = channel_attention(cbam_feature,module_id, ratio)
	cbam_feature = spatial_attention(cbam_feature,module_id)
	return cbam_feature

def channel_attention(input_feature,module_id, ratio=8):
	
	channel_axis = 1 if K.image_data_format() == "channels_first" else -1
	channel = input_feature._keras_shape[channel_axis]
	
	shared_layer_one = Dense(channel//ratio,
							 activation='relu',
							 kernel_initializer='he_normal',
							 use_bias=True,
							 bias_initializer='zeros',
							 name='ca_dense1_'+str(module_id))
	shared_layer_two = Dense(channel,
							 kernel_initializer='he_normal',
							 use_bias=True,
							 bias_initializer='zeros',
							 name='ca_dense2_'+str(module_id))
	
	avg_pool = GlobalAveragePooling2D(name='ca_GAP_'+str(module_id))(input_feature)    
	avg_pool = Reshape((1,1,channel),name='ca_reshape1_'+str(module_id))(avg_pool)
	assert avg_pool._keras_shape[1:] == (1,1,channel)
	avg_pool = shared_layer_one(avg_pool)
	#avg_pool.name = 'ca_dense1_'+str(module_id)
	assert avg_pool._keras_shape[1:] == (1,1,channel//ratio)
	avg_pool = shared_layer_two(avg_pool)
	#avg_pool.name = 'ca_dense2_'+str(module_id)
	assert avg_pool._keras_shape[1:] == (1,1,channel)
	
	max_pool = GlobalMaxPooling2D(name='ca_GMP_'+str(module_id))(input_feature)
	max_pool = Reshape((1,1,channel),name='ca_reshape2_'+str(module_id))(max_pool)
	assert max_pool._keras_shape[1:] == (1,1,channel)
	max_pool = shared_layer_one(max_pool)
	#max_pool.name = 'ca_dense3_'+str(module_id)
	assert max_pool._keras_shape[1:] == (1,1,channel//ratio)
	max_pool = shared_layer_two(max_pool)
	#max_pool.name = 'ca_dense4_'+str(module_id)
	assert max_pool._keras_shape[1:] == (1,1,channel)
	
	cbam_feature = Add(name='ca_Add_'+str(module_id))([avg_pool,max_pool])
	cbam_feature = Activation('sigmoid',name='ca_act_'+str(module_id))(cbam_feature)
	
	if K.image_data_format() == "channels_first":
		cbam_feature = Permute((3, 1, 2))(cbam_feature)
	
	return multiply([input_feature, cbam_feature])

def spatial_attention(input_feature,module_id):
	kernel_size = 7
	
	if K.image_data_format() == "channels_first":
		channel = input_feature._keras_shape[1]
		cbam_feature = Permute((2,3,1),name='sa_permute1_'+str(module_id))(input_feature)
	else:
		channel = input_feature._keras_shape[-1]
		cbam_feature = input_feature
	
	avg_pool = Lambda(lambda x: K.mean(x, axis=3, keepdims=True),name='sa_lambda1_'+str(module_id))(cbam_feature)
	assert avg_pool._keras_shape[-1] == 1
	max_pool = Lambda(lambda x: K.max(x, axis=3, keepdims=True),name='sa_lambda2_'+str(module_id))(cbam_feature)
	assert max_pool._keras_shape[-1] == 1
	concat = Concatenate(axis=3,name='sa_concat_'+str(module_id))([avg_pool, max_pool])
	assert concat._keras_shape[-1] == 2
	cbam_feature = Conv2D(filters = 1,
					kernel_size=kernel_size,
					strides=1,
					padding='same',
					activation='sigmoid',
					kernel_initializer='he_normal',
					use_bias=False,
                    name='sa_Conv2D_'+str(module_id))(concat)	
	assert cbam_feature._keras_shape[-1] == 1
	
	if K.image_data_format() == "channels_first":
		cbam_feature = Permute((3, 1, 2),name='sa_permute2_'+str(module_id))(cbam_feature)
		
	return multiply([input_feature, cbam_feature])