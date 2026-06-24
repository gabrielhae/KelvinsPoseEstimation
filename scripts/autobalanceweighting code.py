# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 00:08:12 2019

@author: Gabriel
"""

alpha = K.variable(0.5)
beta = K.variable(0.5)
model.compile(..., loss_weights=[alpha, beta], ...)

model.fit( ..., callbacks=[MyCallback(alpha, beta)], ...)

class ReWeightLossCallback(Callback):
    def __init__(self, alpha, beta):
        self.alpha = 1.0
        self.beta = 1.0
    # customize your behavior
    def on_epoch_end(self, epoch, logs={}):
        Qloss=logs['q_loss'][-1]
        Rloss=logs['r_loss'][-1]
        
        
        


def on_epoch_end(self, epoch, logs={}):
        if epoch == 2:
            K.set_value(self.alpha, K.get_value(self.alpha) / 1.5)
            K.set_value(self.beta, K.get_value(self.beta) * 1.5)
        logger.info("epoch %s, alpha = %s, beta = %s" % (epoch, K.get_value(self.alpha), K.get_value(self.beta)))
        
def on_batch_end(self, batch, logs={}):
            