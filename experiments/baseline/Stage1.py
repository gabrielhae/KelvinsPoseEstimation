# -*- coding: utf-8 -*-
"""
Stage 1: Pre-Process Images
"""
import os
import json
from random import shuffle,seed

#Variables
speed_root = 'C:\DeepLearningFolder\Pose Estimation\speed/'
dataset='train'
with open(os.path.join(speed_root, dataset + '.json'), 'r') as f:
    label_list = json.load(f)

#seed(3)
#shuffled_label_list=label_list.copy()  
#shuffle(shuffled_label_list) 
#train_labels_set = shuffled_label_list[:int(len(shuffled_label_list)*.8)]
#validation_holdout_set = shuffled_label_list[int(len(shuffled_label_list)*.8):]
#
#shuffle(train_labels_set)
#train_labels = train_labels_set[:int(len(train_labels_set)*.8)]
#validation_labels = train_labels_set[int(len(train_labels_set)*.8):]

########    
#Proprocess images for low res trials
from PIL import Image
from tqdm import tqdm


def calculate_brightness(image):
    greyscale_image = image.convert('L')
    histogram = greyscale_image.histogram()
    pixels = sum(histogram)
    brightness = scale = len(histogram)

    for index in range(0, scale):
        ratio = histogram[index] / pixels
        brightness += ratio * (-scale + index)

    return 1 if brightness == 255 else brightness / scale
    
dataset_root=speed_root
dataset='train'
with open(os.path.join(dataset_root, dataset + '.json'), 'r') as f:
    image_list = json.load(f)

print('Running evaluation on {} set...'.format(dataset))

#brightness=[]
#i=0
#for img in tqdm(image_list):
#    i+=1
#    if i==6000:
#        break
#    img_path = os.path.join(dataset_root, 'images', dataset, img['filename'])
#    processed_img_path = os.path.join(dataset_root, 'preprocessed_images_1200', dataset, img['filename'])
#    
#    im = Image.open(processed_img_path)
#    
#    width, height = im.size   # Get dimensions
#    new_width=1200
#    new_height=1200
#    left = (width - new_width)/2
#    top = (height - new_height)/2
#    right = (width + new_width)/2
#    bottom = (height + new_height)/2
#    
#    croppped=im.crop((left, top, right, bottom))
#    converted=croppped.convert('RGB')
#    resized=converted.resize((600,600),Image.LANCZOS)
#    #converted.save(processed_img_path.replace('jpg', 'png'),format='png')
#    #resized.save(processed_img_path)
#    brightness.append(calculate_brightness(resized))
#######        
#import numpy as np
#from plotly import tools
#import plotly.graph_objs as go
#from plotly.offline import plot
#import plotly.figure_factory as ff
#
#brightnesses = np.array(brightness)  
#hist_data = [brightnesses]
#group_labels = ['Brightness train images 600']
#fig = ff.create_distplot(hist_data, group_labels, bin_size=[.01])
#plot(fig, filename='brightness_distribution6000.html')  
#
#print("Mean Brightness: "+str(np.mean(brightnesses)))
#print("Std Brightness: "+str(np.std(brightnesses)))

    
#"""
#Stage 2: Examine Dataset
#-Plot quaternion viewpoints on sphere
#-Plot position vectors
#-Run statistics on samples for balanced training data
#
#--outcome need to randomly select points uniformly over sphere for training set
#"""
#####Plot quaternions on sphere & Plot 
q=[]
r=[]
for image_ann in label_list:
    q.append(image_ann['q_vbs2tango'])
    r.append(image_ann['r_Vo2To_vbs_true'])
    
q
#    
###q=q[0:7500]
###r=r[0:7500] 
##    
##"""
##viewpoints_around_tango means: -0.0073374100758298285   0.018638749784794035    0.005818862838557613
##distances_to_tango means: -0.001989078533333334 -0.0014909761333333346  10.7930023552
##
##viewpoints_around_tango std: 0.5617402162567091 0.6076709669603496      0.5610247994593859
##distances_to_tango std: 0.3167724518573509      0.40632898049604893     5.963075357225127
##"""
##
##q=q[7500:12001]
##r=r[7500:12001] 
##"""
##viewpoints_around_tango means: 0.012718734705778875     0.015959296453086542    -0.008257695947071928
##distances_to_tango means: -0.0013905924444444445        -0.005670619777777779   11.073876306
##
##viewpoints_around_tango std: 0.5589995312263295 0.6126664067740417      0.55827837429116
##distances_to_tango std: 0.3293510709609005      0.41895338956372874     6.093170656977594
##"""
#    
#from mpl_toolkits import mplot3d
#import numpy as np
#import matplotlib.pyplot as plt
#
#
#positionvectors_to_tango = np.array(r)
#quaternions_around_tango = np.array(q)
#
#
#from pyquaternion import Quaternion
#listviewpoints=[]
#for qt in quaternions_around_tango:
#    q6 = Quaternion(qt)
#    
#    new_position=q6.rotate([0.0, 0.0, -1.0]) #camera is positioned 1 metre behind tango for imageing
#    listviewpoints.append(new_position)
#
#listviewpoints=np.array(listviewpoints)
#
#from plotly.offline import plot
#import plotly.graph_objs as go
#
#import numpy as np
#
#trace1 = go.Scatter3d(
#    x=positionvectors_to_tango[:,0],
#    y=positionvectors_to_tango[:,1],
#    z=positionvectors_to_tango[:,2],
#       
#    mode='markers',
#    marker=dict(
#        size=3,
#        line=dict(
#            color='rgba(217, 217, 217,0.0)',
#            width=0.5
#        ),
#        opacity=1.0
#    )
#)
#trace2 = go.Scatter3d(
#    x=listviewpoints[:,0],
#    y=listviewpoints[:,1],
#    z=listviewpoints[:,2],
#       
#    mode='markers',
#    marker=dict(
#        size=3,
#        line=dict(
#            color='rgba(255, 255, 255,0.0)',
#            width=0.5
#        ),
#        opacity=1.0
#    )
#)
#data = [trace1]
#data2 = [trace2]
#layout = go.Layout(
#    margin=dict(
#        l=0,
#        r=0,
#        b=0,
#        t=0
#    )
#)
#plot(data, filename='distances_to_tango.html')
#plot(data2, filename='viewpoints_around_tango.html')
#
#from plotly.graph_objs import Mesh3d
#from numpy import sin, cos, pi
#
## some math: generate points on the surface of ellipsoid
#
#phi = np.linspace(0, 2*pi)
#theta = np.linspace(-pi/2, pi/2)
#phi, theta=np.meshgrid(phi, theta)
#
#x = cos(theta) * sin(phi) * 0.9
#y = cos(theta) * cos(phi) * 0.9
#z = sin(theta)
#
## to use with Jupyter notebook
#
#plot([Mesh3d({
#                'x': x.flatten(), 
#                'y': y.flatten(), 
#                'z': z.flatten(), 
#                'alphahull': 0
#}),trace2],filename='sphere.html')
#    
#    
##stats
#print('viewpoints_around_tango means: '+str(np.mean(listviewpoints[:,0]))+'\t'+str(np.mean(listviewpoints[:,1]))+'\t'+str(np.mean(listviewpoints[:,2])))
#print('distances_to_tango means: '+str(np.mean(positionvectors_to_tango[:,0]))+'\t'+str(np.mean(positionvectors_to_tango[:,1]))+'\t'+str(np.mean(positionvectors_to_tango[:,2])))
#print('')
#print('viewpoints_around_tango std: '+str(np.std(listviewpoints[:,0]))+'\t'+str(np.std(listviewpoints[:,1]))+'\t'+str(np.std(listviewpoints[:,2])))
#print('distances_to_tango std: '+str(np.std(positionvectors_to_tango[:,0]))+'\t'+str(np.std(positionvectors_to_tango[:,1]))+'\t'+str(np.std(positionvectors_to_tango[:,2])))
#
#
##Stage 4 - test loss function
#
#import tensorflow as tf
#from keras import backend as K
#import numpy as np
#
#def geodesic_loss(y_pred, y_true):
#    y_pred=tf.nn.l2_normalize(y_pred,axis = 0)
#    tmp=tf.linalg.tensordot(y_true,y_pred,axes=1)
#    theta=2.0*tf.math.acos(tf.clip_by_value(tmp,-1.0+1e-12,1.0-1e-12))
#    return theta
#
#
#def normalization_loss(y_pred):
#    beta=10.0
#    norm=tf.linalg.global_norm(y_pred)
#    output=beta*tf.math.square(1.0-norm)
#    return output
#
###regularlizaton term to prevent overfitting
#
#
#        
#        return tf.reduce_mean(theta)