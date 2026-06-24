from keras.layers import *
from keras.models import Model
from keras.constraints import Constraint
from keras.initializers import Constant
from keras.layers import Layer
from keras.callbacks import Callback

from keras import backend as K
import numpy

class CurrentLossWeight(Callback):
    
    def __init__(self, model):
        self.model = model
        self.layer_dict = dict([(layer.name, layer) for layer in self.model.layers])
        
    def on_train_begin(self, epoch, logs=None):
         u_weights=self.layer_dict['weighted_loss'].get_weights()
         print('sx: ' + str("{:.3f}".format(u_weights[0][0]))+' sq: ' + str("{:.3f}".format(u_weights[1][0])))   

    def on_epoch_end(self, epoch, logs=None):
         u_weights=self.layer_dict['weighted_loss'].get_weights()
         print('sx: ' + str("{:.3f}".format(u_weights[0][0]))+' sq: ' + str("{:.3f}".format(u_weights[1][0])))                  
        
class Between(Constraint):
    def __init__(self,min_value,max_value):
        self.min_value = min_value
        self.max_value = max_value

    def __call__(self,w):
        return K.clip(w,self.min_value, self.max_value)

    def get_config(self):
        return {'min_value': self.min_value,
                'max_value': self.max_value}

class LossWeighter(Layer):
    def __init__(self, **kwargs): #kwargs can have 'name' and other things
        super(LossWeighter, self).__init__(**kwargs)

    #create the trainable weight here, notice the constraint between 0 and 1
    def build(self, inputShape):
        self.sx = self.add_weight(name='sx', 
                                     shape=(1,),
                                     initializer=Constant(-0.6), 
                                     constraint=Between(-100,100),
                                     trainable=True)  
        self.sq = self.add_weight(name='sq', 
                                     shape=(1,),
                                     initializer=Constant(-2.2), 
                                     constraint=Between(-100,100),
                                     trainable=True)      
        
        super(LossWeighter,self).build(inputShape)

    def call(self,inputs):
        qLoss, rLoss = inputs
        return K.exp(-self.sx)*rLoss + self.sx + K.exp(-self.sq)*qLoss + self.sq
    
    def compute_output_shape(self,inputShape):
        inputShape
        return inputShape[0]