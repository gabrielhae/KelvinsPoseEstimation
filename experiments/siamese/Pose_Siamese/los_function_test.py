# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 18:10:23 2019

@author: Gabriel
"""
import tensorflow as tf
import numpy as np
from pyquaternion import Quaternion
from scipy.spatial import distance
from keras import backend as K

K.clear_session() 

sess = tf.InteractiveSession()

def geodesic_angular_loss(y_pred, y_true):
    #Validated as working Has max at 3.141 and min at 0.0
    y_pred=K.l2_normalize(y_pred,axis = -1)
    y_true=K.l2_normalize(y_true,axis = -1)
    dotprod=K.sum(y_true*y_pred,axis=-1,keepdims=True)
    theta=2.0*tf.math.acos(dotprod)
    return theta

def deltarloss(y_true, y_pred):

    delta=y_pred-y_true    
    delta_normsquared=K.sum(delta*delta,axis=-1,keepdims=True)    
    delta=K.l2_normalize(delta, axis=-1)
    y_true = K.l2_normalize(y_true, axis=-1)
    sbar=1.0 - K.sum(y_true*delta,axis=-1,keepdims=True)
    
    return delta_normsquared*sbar

def modified_geodesic_loss(y_pred, y_true):
    #Validated as working Has max at 1.0 and min at 0.0
    y_pred=K.l2_normalize(y_pred,axis = -1)
    y_true=K.l2_normalize(y_true,axis = -1)
    dotprod=K.sum(y_true*y_pred,axis=-1,keepdims=True)
    result = 1.0 - K.square(dotprod)
    return result

def numpy_modified_geodesic_loss(q1, q2):
    #Validated as working Has max at 1.0 and min at 0.0
    q1=q1/np.linalg.norm(q1,2)
    q2=q2/np.linalg.norm(q2,2)
    dotprod=np.dot(q2, q1)
    result = 1.0 - np.square(dotprod)
    return result

def norm_penalty(_, y_pred):
    norm = K.sqrt(K.sum(K.square(y_pred), axis=-1))
    penalty = K.square(1.0 - norm)
    return penalty

def numpy_norm_penalty(_, q_pred):
    norm = np.linalg.norm(q_pred,2)
    penalty = np.square(1.0 - norm)
    return penalty


def my_r2_score(y_true, y_pred):
    ssres = np.sum(np.square(y_true - y_pred))
    sstot = np.sum(np.square(y_true - np.mean(y_true)))
    return 1 - ssres / sstot

def r2_score(y_true, y_pred):
    SS_res =  K.sum(K.square(y_true - y_pred)) 
    SS_tot = K.sum(K.square(y_true - K.mean(y_true))) 
    return ( 1 - SS_res/(SS_tot + K.epsilon()) )

def explained_variance(y_true, y_pred):
    nom =  K.var(y_true - y_pred) 
    denom = K.var(y_true)
    return ( 1 - nom/(denom + K.epsilon()) )

def mse(y_true, y_pred):
    return K.mean(K.square(y_pred - y_true), axis=-1,keepdims=True)

from sklearn.metrics import explained_variance_score


posvec1=np.array([0.10024913,-0.23724233,7.1780357])
posvec2=np.array([-0.544427,0.73372984,22.013918])
posvec1 = tf.convert_to_tensor(posvec1, dtype=tf.float32)
posvec2 = tf.convert_to_tensor(posvec2, dtype=tf.float32)

print("Legeo Loss: "+str(deltarloss(posvec1,posvec1).eval()))

#q1=np.array([0.743431,-0.513377,-0.382707,0.193104])
##import os
##import random
##import json
##with open(os.path.join(r'C:/DeepLearningFolder/Pose Estimation/speed/', 'train' + '.json'), 'r') as f:
##    label_list = json.load(f)
##random.shuffle(label_list)#Shuffle the label list first
##train_labels = label_list[:int(len(label_list)*.7)]
##
###q_poses = [label['q_vbs2tango'] for label in train_labels]
##q_poses=np.array([label['q_vbs2tango'] for label in train_labels])
###q_distances=numpy_modified_geodesic_loss(q_poses,q1)
###q_distances=[numpy_modified_geodesic_loss(qposes, q1) for qposes in q_poses]
###q_distances=[distance.cdist(qmatch, q1, 'euclidean') for qmatch in q_poses]
##q_distances=distance.cdist(q_poses, q1, 'euclidean')
###q_distances=np.array([numpy_modified_geodesic_loss(qmatch, q) for qmatch in self.q_poses])
##max_value = max(q_distances)
##max_index = q_distances.index(max_value)
##min_value = min(q_distances)
##min_index = q_distances.index(min_value)   
###sorted_qdistances = [[index,num] for index, num in sorted(enumerate(q_distances), key=lambda x: x[-1])]
##
##shortest_distance=min_value
##farthest_distance=max_value
##
##shortest_imagename=train_labels[min_index]['filename']
##farthest_imagename=train_labels[max_index]['filename']
#
#
#
#q2=np.array([-0.285419,-0.21395,0.28536,-0.889568])
#
#q1=np.array([1.0,0.5,0.6,0.2])
#q2=np.array([0.71,0.71,0.0,0.0])
#
#q5=np.array([1.0,0.5,0.2,0.2])
#q6=np.array([0.5,0.71,0.0,0.0])
#
#q7stacked=np.stack((q5, q6), axis=0)
#
#
#
#print("nump r2 test: "+str(my_r2_score(q1,q1)))
#
#print("nump Geodesic Losses: "+str(numpy_modified_geodesic_loss(q1,q2)))
##from scipy.spatial import distance
#print("cosine distance: "+str(distance.cdist(np.reshape(np.array(q1), (1, 4)), np.reshape(np.array(q2), (1, 4)), 'cosine')))
##Y = distance.cdist(q1, q2, 'cosine')
#q3=np.stack((q1, q2), axis=0)
#
#q4=np.stack((q2, q1), axis=0)
#
#print("keras explained variance score stacked: "+str(explained_variance_score(q3,q7stacked)))
#
#print("nump r2 test array: "+str(my_r2_score(q3,q4)))
#
#q1pq = Quaternion(q1)
#q2pq = Quaternion(q2)
#
##print(Quaternion.distance(q1pq, q2pq))#Find the intrinsic geodesic distance between q0 and q1.
#
##print(Quaternion.sym_distance(q1pq, q2pq))
##
##print(Quaternion.absolute_distance(q1pq, q2pq))#not right
#
#
#q1 = tf.convert_to_tensor(q1, dtype=tf.float32)
#q2 = tf.convert_to_tensor(q2, dtype=tf.float32)
#check=geodesic_angular_loss(q1,q2)
#
##print(check.eval())
#
##print("keras r2 test: "+str(coeff_determination(q1,q1).eval()))
#
#
#
#print("Norm Penalty: "+str(norm_penalty(q2,q2).eval()))
#
#print("Angular Loss Identical q: "+str(geodesic_angular_loss(q1,q2).eval()))
#
#q3 = tf.convert_to_tensor(q3, dtype=tf.float32)
#q4 = tf.convert_to_tensor(q4, dtype=tf.float32)
#
#print("MSE test: "+str(mse(q3,q4).eval()))
#
##print("Angular Losses: "+str(geodesic_angular_loss(q3,q4).eval()))
#
#print("Geodesic Losses: "+str(modified_geodesic_loss(q1,q2).eval()))
#
#
##print("keras r2 test array: "+str(coeff_determination(q3,q4).eval()))
#
#print("keras explained variance score: "+str(explained_variance(q1,q2).eval()))
#
#q7stacked = tf.convert_to_tensor(q7stacked, dtype=tf.float32)
#
#print("keras explained variance score stacked: "+str(explained_variance(q3,q7stacked).eval()))

sess.close()