#https://www.learnopencv.com/head-pose-estimation-using-opencv-and-dlib/


import numpy as np
import json
import os
from PIL import Image
from PIL import ImageDraw
from matplotlib import pyplot as plt
import matplotlib.patches as patches
from random import seed,choice,random

import pyquaternion
from pyquaternion import Quaternion
from rotation import Rotation as R

import warnings
warnings.filterwarnings("ignore")

from skimage.transform import AffineTransform, warp,rotate,SimilarityTransform
from skimage import util

# deep learning framework imports
try:
    from tensorflow.keras.utils import Sequence
    from tensorflow.keras.preprocessing import image as keras_image
    has_tf = True
except ModuleNotFoundError:
    has_tf = False

try:
    import torch
    from torch.utils.data import Dataset
    from torchvision import transforms
    has_pytorch = True
except ImportError:
    has_pytorch = False
    

from cv2 import resize


from skimage.restoration import (denoise_tv_chambolle, denoise_bilateral,
                                 denoise_wavelet, estimate_sigma,denoise_nl_means)
from skimage import data, img_as_float,color
from skimage.util import random_noise
from skimage import transform
from scipy.ndimage import interpolation
#import cv2

#seed(11)
#np.random.seed(seed=5)

images='preprocessed_images_1200'#preprocessed_

angles=[0,45,90,135,180,225,270,315]
angle=np.random.choice(angles)
#angle=0

edgemethod='edge'
#‘constant’, ‘edge’, ‘symmetric’, ‘reflect’, ‘wrap’

#could probably be improved
#SatMarkers=np.array([[-.4,.375,0],[.4,.375,0],[.4,-.375,0],[-.4,-.375,0],[-.4,.375,.32],[.4,.375,.32],[.4,-.375,.32],[-.4,-.375,.32]]) #could probably scale this by factor, augentation?
SatMarkers=np.array([[-.6,.575,-.1],[.6,.575,-.1],[.6,-.575,-.1],[-.6,-.575,-.1],[-.6,.575,.42],[.6,.575,.42],[.6,-.575,.42],[-.6,-.575,.42]])


class Camera:

    """" Utility class for accessing camera parameters. """

    fx = 0.0176  # focal length[m]
    fy = 0.0176  # focal length[m]
    nu = 1200  # number of horizontal[pixels]
    nv = 1200  # number of vertical[pixels]
    ppx = 5.86e-6*1200/nu # horizontal pixel pitch[m / pixel]
    ppy = ppx  # vertical pixel pitch[m / pixel]
    fpx = fx / ppx  # horizontal focal length[pixels]
    fpy = fy / ppy  # vertical focal length[pixels]
    k = [[fpx,   0, nu / 2],
         [0,   fpy, nv / 2],
         [0,     0,      1]]
    
    K = np.array(k)


def process_json_dataset(root_dir):
    with open(os.path.join(root_dir, 'train.json'), 'r') as f:
        train_images_labels = json.load(f)

    with open(os.path.join(root_dir, 'test.json'), 'r') as f:
        test_image_list = json.load(f)

    with open(os.path.join(root_dir, 'real_test.json'), 'r') as f:
        real_test_image_list = json.load(f)

    partitions = {'test': [], 'train': [], 'real_test': []}
    labels = {}

    for image_ann in train_images_labels:
        partitions['train'].append(image_ann['filename'])
        labels[image_ann['filename']] = {'q': image_ann['q_vbs2tango'], 'r': image_ann['r_Vo2To_vbs_true']}

    for image in test_image_list:
        partitions['test'].append(image['filename'])

    for image in real_test_image_list:
        partitions['real_test'].append(image['filename'])

    return partitions, labels


def quat2dcm(q):

    """ Computing direction cosine matrix from quaternion, adapted from PyNav. """

    # normalizing quaternion
    q = q/np.linalg.norm(q)

    q0 = q[0]
    q1 = q[1]
    q2 = q[2]
    q3 = q[3]

    dcm = np.zeros((3, 3))

    dcm[0, 0] = 2 * q0 ** 2 - 1 + 2 * q1 ** 2
    dcm[1, 1] = 2 * q0 ** 2 - 1 + 2 * q2 ** 2
    dcm[2, 2] = 2 * q0 ** 2 - 1 + 2 * q3 ** 2

    dcm[0, 1] = 2 * q1 * q2 + 2 * q0 * q3
    dcm[0, 2] = 2 * q1 * q3 - 2 * q0 * q2

    dcm[1, 0] = 2 * q1 * q2 - 2 * q0 * q3
    dcm[1, 2] = 2 * q2 * q3 + 2 * q0 * q1

    dcm[2, 0] = 2 * q1 * q3 + 2 * q0 * q2
    dcm[2, 1] = 2 * q2 * q3 - 2 * q0 * q1

    return dcm


def project(q, r):

        """ Projecting points to image frame to draw axes """

        # reference points in satellite frame for drawing axes
        p_axes = np.array([[0, 0, 0, 1],
                           [1, 0, 0, 1],
                           [0, 1, 0, 1],
                           [0, 0, 1, 1]])
        points_body = np.transpose(p_axes)

        # transformation to camera frame
        pose_mat = np.hstack((np.transpose(quat2dcm(q)), np.expand_dims(r, 1)))
        p_cam = np.dot(pose_mat, points_body)

        # getting homogeneous coordinates
        points_camera_frame = p_cam / p_cam[2]

        # projection to image plane
        points_image_plane = Camera.K.dot(points_camera_frame)

        x, y = (points_image_plane[0], points_image_plane[1])
        return x, y
    
def projectMarkers(q, r, marker):

        """ Projecting points to image frame to draw axes """

        # reference points in tango satellite frame for drawing axes
        p_axes = np.array([[marker[0], marker[1], marker[2], 1],
                           [1, 0, 0, 1],
                           [0, 1, 0, 1],
                           [0, 0, 1, 1]])
        
        points_body = np.transpose(p_axes)

        # transformation to camera frame
        pose_mat = np.hstack((np.transpose(quat2dcm(q)), np.expand_dims(r, 1)))
        p_cam = np.dot(pose_mat, points_body)

        # getting homogeneous coordinates
        points_camera_frame = p_cam / p_cam[2]

        # projection to image plane
        points_image_plane = Camera.K.dot(points_camera_frame)

        x, y = (points_image_plane[0], points_image_plane[1])
        return x, y
    
def quaternion_mult(q,r):
    return [r[0]*q[0]-r[1]*q[1]-r[2]*q[2]-r[3]*q[3],
            r[0]*q[1]+r[1]*q[0]-r[2]*q[3]+r[3]*q[2],
            r[0]*q[2]+r[1]*q[3]+r[2]*q[0]-r[3]*q[1],
            r[0]*q[3]-r[1]*q[2]+r[2]*q[1]+r[3]*q[0]]

def point_rotation_by_quaternion(point,q):
    r = [0]+point
    q_conj = [q[0],-1*q[1],-1*q[2],-1*q[3]]
    return quaternion_mult(quaternion_mult(q,r),q_conj)[1:]

def getangle(v1, v2, acute):
# v1 is your firsr vector
# v2 is your second vector
    angle = np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
    if (acute == True):
        return angle
    else:
        return 2 * np.pi - angle
    
def make_q_between(angle,v1,v2):
    w=np.cross(v1, v2)
    w=w /np.linalg.norm(w)
    q=[angle,w[0],w[1],w[2]]
    q=q /np.linalg.norm(q)
    return q

def q_fromtwovectors(v1,v2):
    v1=v1 /np.linalg.norm(v1)
    v2=v2 /np.linalg.norm(v2)
    w = np.cross(v1, v2)
    ang=1.0 + np.dot(v1, v2)
    q=[ang, w[0], w[1], w[2]]
    q=q /np.linalg.norm(q)
    return q


class SatellitePoseEstimationDataset:

    """ Class for dataset inspection: easily accessing single images, and corresponding ground truth pose data. """

    def __init__(self, root_dir='/datasets/speed_debug'):
        self.partitions, self.labels = process_json_dataset(root_dir)
        self.root_dir = root_dir

    def get_image(self, i=0, split='train'):

        """ Loading image as PIL image. """

        img_name = self.partitions[split][i]
        img_name = os.path.join(self.root_dir, images, split, img_name)
        image = Image.open(img_name).convert('RGB')#.replace("jpg","png")
        
        return image

    def get_pose(self, i=0):

        """ Getting pose label for image. """

        img_id = self.partitions['train'][i]
        q, r = self.labels[img_id]['q'], self.labels[img_id]['r']
        return q, r
    

    def shift(self,image, vector):
        transform = SimilarityTransform(translation=vector)
        shifted = warp(image, transform, mode=edgemethod, preserve_range=True)
        return shifted.astype(image.dtype)
    
    def scaleit(self,image, factor, isseg=False):
        order = 0 if isseg == True else 3
    
        height, width, depth= image.shape
        zheight             = int(np.round(factor * height))
        zwidth              = int(np.round(factor * width))
        zdepth              = depth
    
        if factor < 1.0:
            newimg  = np.zeros_like(image)
            row     = (height - zheight) // 2
            col     = (width - zwidth) // 2
            layer   = (depth - zdepth) // 2
            newimg[row:row+zheight, col:col+zwidth, layer:layer+zdepth] = interpolation.zoom(image, (float(factor), float(factor), 1.0), order=order, mode='nearest')[0:zheight, 0:zwidth, 0:zdepth]
    
            return newimg
    
        elif factor > 1.0:
            row     = (zheight - height) // 2
            col     = (zwidth - width) // 2
            layer   = (zdepth - depth) // 2
    
            newimg = interpolation.zoom(image[row:row+zheight, col:col+zwidth, layer:layer+zdepth], (float(factor), float(factor), 1.0), order=order, mode='nearest')  
            
            extrah = (newimg.shape[0] - height) // 2
            extraw = (newimg.shape[1] - width) // 2
            extrad = (newimg.shape[2] - depth) // 2
            newimg = newimg[extrah:extrah+height, extraw:extraw+width, extrad:extrad+depth]
    
            return newimg
    
        else:
            return image    
        
    def clipped_zoom(self,img, zoom_factor):
        """
        Center zoom in/out of the given image and returning an enlarged/shrinked view of 
        the image without changing dimensions
        Args:
            img : Image array
            zoom_factor : amount of zoom as a ratio (0 to Inf)
        """
        height, width = img.shape[:2] # It's also the final desired shape
        new_height, new_width = int(height * zoom_factor), int(width * zoom_factor)
    
        ### Crop only the part that will remain in the result (more efficient)
        # Centered bbox of the final desired size in resized (larger/smaller) image coordinates
        y1, x1 = max(0, new_height - height) // 2, max(0, new_width - width) // 2
        y2, x2 = y1 + height, x1 + width
        bbox = np.array([y1,x1,y2,x2])
        # Map back to original image coordinates
        bbox = (bbox / zoom_factor).astype(np.int)
        y1, x1, y2, x2 = bbox
        cropped_img = img[y1:y2, x1:x2]
    
        # Handle padding when downscaling
        resize_height, resize_width = min(new_height, height), min(new_width, width)
        pad_height1, pad_width1 = (height - resize_height) // 2, (width - resize_width) //2
        pad_height2, pad_width2 = (height - resize_height) - pad_height1, (width - resize_width) - pad_width1
        pad_spec = [(pad_height1, pad_height2), (pad_width1, pad_width2)] + [(0,0)] * (img.ndim - 2)
    
        result = resize(cropped_img, (resize_width, resize_height))
        result = np.pad(result, pad_spec, mode=edgemethod)
        assert result.shape[0] == height and result.shape[1] == width
        
        return result        

    def visualize(self, i, partition='train', ax=None):

        """ Visualizing image, with ground truth pose with axes projected to training image. """

        if ax is None:
            ax = plt.gca()
        img = self.get_image(i)
        width, height = img.size
        q, r = self.get_pose(i)
        
        
                     
        numpy_img=np.array(img)#
        newz = r[2] + (random() * (40.0 - r[2]))
        scale_downsize=r[2]/newz
        numpy_img=self.clipped_zoom(numpy_img, scale_downsize)
        r[2]=newz    
        #new_x,new_y=project(q, r)
        #ax.scatter(new_x[0], new_y[0],color='b',s=8)      
        
        
        #
        scale=r[2]/Camera.fx
        
        #Uniform translation
        new_x,new_y=np.random.uniform(200,width-200),np.random.uniform(200,height-200)
        #new_x,new_y=100,1100#np.random.uniform(100,width-100),np.random.uniform(100,height-100)
        old_x,old_y=project(q, r)
        #print('Original Pixel Coord of r: '+str(old_x[0]),str(old_y[0]))
        #print('Translating to centre  Pixel Coord of r: '+str(old_x[0]-width/2),str(old_y[0]-height/2))
        print('Target Pixel Coord before projection and vector rotation: '+str(new_x),str(new_y))
        old_x,old_y=old_x[0],old_y[0]
        translate_x=old_x-new_x
        translate_y=old_y-new_y
        #translate_x=0
        #translate_y=0

        r_new=[(new_x-width/2)*Camera.ppx*scale,(new_y-height/2)*Camera.ppx*scale,Camera.fpx*Camera.ppx*scale]
        
        #translate_x=-0
        #translate_y=0
        #ax.scatter(width/2, height/2,color='w',s=2)
        #ax.scatter(new_x, new_y,color='r',s=8)#unrotated    
        r=r_new
        #ax.scatter(tnew_x[0], tnew_y[0],color='b',s=8)
        
        shifted_image=self.shift(numpy_img,(translate_x,translate_y))#working to here

        new_im = Image.fromarray(shifted_image)

        rotated = util.img_as_ubyte(rotate(shifted_image, -angle,mode=edgemethod,preserve_range=False))
        new_im = Image.fromarray(rotated)
#        
        ax.imshow(new_im)

        # no pose label for test
        if partition == 'train':
           
            q = self.q_rotation(q,angle)
            r = self.z_rotation(r,angle)
            xa, ya = project(q, r)##checking
            
            print('Pixel following vector rotation about camera axis: '+str(xa[0]), str(ya[0]))
            #print(xa[0], ya[0])
            
            ax.arrow(xa[0], ya[0], xa[1] - xa[0], ya[1] - ya[0], head_width=10, color='r')#i left camera direction
            ax.arrow(xa[0], ya[0], xa[2] - xa[0], ya[2] - ya[0], head_width=10, color='g')#j up camera direction
            ax.arrow(xa[0], ya[0], xa[3] - xa[0], ya[3] - ya[0], head_width=10, color='b')#k down camera lens            

            #######START CODE for Cropping and Shifting            
            xa1=np.zeros(shape=(8,1))
            ya1=np.zeros(shape=(8,1))
            i=0
            for marker in SatMarkers:
            #mrkr=np.add(np.array([0.4,0,0]),r)
                xa1q, ya1q = projectMarkers(q, r,marker)#output is pixel positions
                xa1[i],ya1[i]=xa1q[0], ya1q[0]
            #print(xa1, ya1)
                i+=1
            #print('max x: '+str(np.max(xa1)))
            #print('min x: '+str(np.min(xa1)))
            #print('max y: '+str(np.max(ya1)))
            #print('min y: '+str(np.min(ya1)))
            
            deltax=np.max(xa1)-np.min(xa1)
            deltay=np.max(ya1)-np.min(ya1)
            max_boxdim=np.ceil(max(deltax,deltay))
            #print(max_boxdim)
            #print(max_boxdim)#should multipy by 
            #ax.scatter(xa1, ya1,color='c',s=4)
            
            if deltax>deltay:
                #print('x>y')
                left=np.floor(np.min(xa1))
                right=np.floor(np.max(xa1))
                
                top=np.floor(np.min(ya1)+(deltay/2.)-(max_boxdim/2.))
                bottom=np.floor(np.max(ya1)-(deltay/2.)+(max_boxdim/2.))
                #modify ya1 values
                
                
            if deltay>deltax:
                
                #print('y>x')
                top=np.floor(np.min(ya1))
                bottom=np.floor(np.max(ya1))
                left=np.floor(np.min(xa1)+(deltax/2.)-(max_boxdim/2.))
                right=np.floor(np.max(xa1)-(deltax/2.)+(max_boxdim/2.))    
  
            #bounding box not restricted by image dims              
#            ax.scatter(left, top,color='r',s=2)
#            ax.scatter(right, top,color='r',s=2)
#            ax.scatter(left, bottom,color='r',s=2)
#            ax.scatter(right, bottom,color='r',s=2) 


            #rect = patches.Rectangle((left,top),max_boxdim,max_boxdim,linewidth=1,edgecolor='r',facecolor='none')
            #ax.add_patch(rect) 
                
            if left<0:
                #print('left<0')
                left=0
                right=max_boxdim
                
            if top<0:
                #print('top<0')
                top=0
                bottom=max_boxdim
                
            if right>width:
                #print('right>w')
                left=width-max_boxdim
                right=width
                
            if bottom>height:
                #print('bottom>h')
                top=height-max_boxdim
                bottom=height       
                
            if left<0 or bottom<0 or right<0 or top<0:
                left=0
                bottom=600
                right=600
                top=0                
#                
#            rect = patches.Rectangle((left,top),600,600,linewidth=1,edgecolor='b',facecolor='none')
#            ax.add_patch(rect)  
##                
#            ax.scatter(left, top,color='r',s=5)
#            ax.scatter(right, top,color='g',s=5)
#            ax.scatter(left, bottom,color='b',s=5)
#            ax.scatter(right, bottom,color='y',s=5)
##                
            #cropped_example = img.crop((left, top, right, bottom))
            #cropped_example.save("crop.jpg")
        
#            rect = patches.Rectangle((bottom,left),max_boxdim,max_boxdim,linewidth=1,edgecolor='r',facecolor='none')
#            ax.add_patch(rect)
            
            #ax.scatter(xa1, ya1,color='y',s=2)
            #######END CODE for Cropping and Shifting


        return
  
    def z_rotation(self, vector,angle):
        """Rotates 3-D vector around z-axis"""
        theta=np.radians(angle)
        R = np.array([[np.cos(theta), -np.sin(theta),0],[np.sin(theta), np.cos(theta),0],[0,0,1]])
        return np.dot(R,vector)    
    
    def q_rotation(self, quat,angle):
        """Rotates quaternion around the z-axis"""
        quaternion = R.from_quat(quat)
        q90r = R.from_euler('x',-angle, degrees=True)
        quaternion*=q90r
        qrotated=quaternion.as_quat()
        return qrotated      

from matplotlib import pyplot as plt
from random import randint
#

dataset_root_dir = r'C:\DeepLearningFolder\Pose Estimation\speed'
dataset = SatellitePoseEstimationDataset(root_dir=dataset_root_dir)
#######
#######For Bounding Box Points
partitions,labels=process_json_dataset(dataset_root_dir)

def getbbox(q,r,width,height):
    
    xa1=np.zeros(shape=(8,1))
    ya1=np.zeros(shape=(8,1))
    i=0
    for marker in SatMarkers:
        xa1q, ya1q = projectMarkers(q, r,marker)#output is pixel positions
        xa1[i],ya1[i]=xa1q[0], ya1q[0]
        i+=1
    
    deltax=np.max(xa1)-np.min(xa1)
    deltay=np.max(ya1)-np.min(ya1)
    max_boxdim=np.ceil(max(deltax,deltay))
    
    if deltax>deltay:
        left=np.floor(np.min(xa1))
        right=np.floor(np.max(xa1))
        
        top=np.floor(np.min(ya1)+(deltay/2.)-(max_boxdim/2.))
        bottom=np.floor(np.max(ya1)-(deltay/2.)+(max_boxdim/2.))                
        
    if deltay>deltax:
        
        top=np.floor(np.min(ya1))
        bottom=np.floor(np.max(ya1))
        left=np.floor(np.min(xa1)+(deltax/2.)-(max_boxdim/2.))
        right=np.floor(np.max(xa1)-(deltax/2.)+(max_boxdim/2.))    
        
    if left<0:
        #print('left<0')
        left=0
        right=max_boxdim
        
    if top<0:
        #print('top<0')
        top=0
        bottom=max_boxdim
        
    if right>width:
        #print('right>w')
        left=width-max_boxdim
        right=width
        
    if bottom>height:
        #print('bottom>h')
        top=height-max_boxdim
        bottom=height       
    
    if left<0 or bottom<0 or right<0 or top<0:
        left=0
        bottom=height
        right=width
        top=0
    
    
    return int(left),int(top),int(right),int(bottom)

#f = open('600_bbox_data.txt', 'w')
#import random
#bbox_list=[]
#sizeforhist=[]
#for i in range(0,12000):
#    img_id = partitions['train'][i]
#    q, r = labels[img_id]['q'], labels[img_id]['r']
#    labels[img_id]['bbox']=list(getbbox(q,r,600,600))
#    #print(i,getbbox(q,r,1200,1200))
#    bbox=labels[img_id]['bbox']#.replace("jpg","png")
#    classname=random.choice(["tango"])#,"random","odd","mouse","dog","cat"])
#    
#    #img_name = os.path.join(dataset_root_dir, images, 'train', img_id)
#    #img = Image.open(img_name).convert('RGB')
#    #cropped_example = img.crop((bbox[0], bbox[1], bbox[2], bbox[3]))
#    #cropped_example = cropped_example.resize((1200,1200), Image.ANTIALIAS)
#    #cropped_example.save("C:/DeepLearningFolder/Pose Estimation/speed/preprocessed_images_1200_cropped_resized_bbox/train/"+str(img_id))
#    #f.write("C:/DeepLearningFolder/Pose Estimation/speed/preprocessed_images_600/train/"+str(img_id)+","+str(bbox[0])+","+str(bbox[1])+","+str(bbox[2])+","+str(bbox[3])+","+str(r[0])+","+str(r[1])+","+str(r[2])+","+classname+"\n") 
#    bbox_list.append(labels[img_id]['bbox'])
#    sizeforhist.append(bbox[2]-bbox[0])
    
    
 # python will convert \n to os.linesep
#f.close()
#from plotly import tools
#import plotly.graph_objs as go
#from plotly.offline import plot
#import plotly.figure_factory as ff
#
#sizeforhist=np.array(sizeforhist)
#hist_data = [sizeforhist[:]]
#group_labels = ['Training Box Sizes']
#fig = ff.create_distplot(hist_data, group_labels, bin_size=[4])
#plot(fig, filename='600Box Sizes.html') 
 
######
##Augment quaternions for balancing
#partitions,labels=process_json_dataset(dataset_root_dir)
#
#def z_rotation(vector,theta):
#    """Rotates 3-D vector around z-axis"""
#    R = np.array([[np.cos(theta), -np.sin(theta),0],[np.sin(theta), np.cos(theta),0],[0,0,1]])
#    return np.dot(R,vector)
#
#q_list=[]
#r_list=[]
#for i in range(0,12000):
#    img_id = partitions['train'][i]
#    q, r = labels[img_id]['q'], labels[img_id]['r']
#    #q_list.append(q)
#    #r_list.append(r)
#    quaternion = R.from_quat(q)
#    for angle in [0,90,180,270]:
#        q90r = R.from_euler('x',angle, degrees=True)
#        rotated_quaternion=quaternion*q90r
#        r_q=rotated_quaternion.as_quat()
#        r_r = z_rotation(r,np.radians(-angle))
#        q_list.append(r_q.tolist())
#        r_list.append(r_r.tolist())
        
        

#import numpy as np
#
#positionvectors_to_tango = np.array(r_list)
#quaternions_around_tango = np.array(q_list)
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
#plot(data, filename='augmented_distances_to_tango.html')
#plot(data2, filename='augmented_viewpoints_around_tango.html')
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
#}),trace2],filename='augmented_sphere.html')
#    
#    
##stats
#print('viewpoints_around_tango means: '+str(np.mean(listviewpoints[:,0]))+'\t'+str(np.mean(listviewpoints[:,1]))+'\t'+str(np.mean(listviewpoints[:,2])))
#print('distances_to_tango means: '+str(np.mean(positionvectors_to_tango[:,0]))+'\t'+str(np.mean(positionvectors_to_tango[:,1]))+'\t'+str(np.mean(positionvectors_to_tango[:,2])))
#print('')
#print('viewpoints_around_tango std: '+str(np.std(listviewpoints[:,0]))+'\t'+str(np.std(listviewpoints[:,1]))+'\t'+str(np.std(listviewpoints[:,2])))
#print('distances_to_tango std: '+str(np.std(positionvectors_to_tango[:,0]))+'\t'+str(np.std(positionvectors_to_tango[:,1]))+'\t'+str(np.std(positionvectors_to_tango[:,2])))


rows = 2
cols = 2



#fig, axes = plt.subplots(rows, cols, figsize=(12, 6))
#for i in range(rows):
#    for j in range(cols):
#        img = dataset.get_image(randint(0, 12000))
#        axes[i][j].imshow(img)
#        axes[i][j].axis('off')
#fig.tight_layout()
#plt.show()
#
#rows = 2
#cols = 2

#fig, axes = plt.subplots(rows, cols, figsize=(12, 12))
#for i in range(rows):
#    for j in range(cols):
#        dataset.visualize(randint(0, 200), ax=axes[i][j])
#        axes[i][j].axis('off')
#fig.tight_layout()
#plt.show()


fig, axes = plt.subplots(1, 1, figsize=(7, 7))
dataset.visualize(85)
plt.axis('off')
plt.tight_layout()
plt.show()
#fig, axes = plt.subplots(1, 1, figsize=(7, 7))
#dataset.visualize(19)
#plt.axis('off')
#plt.tight_layout()
#plt.show()
#fig, axes = plt.subplots(1, 1, figsize=(7, 7))
#dataset.visualize(11958)
#plt.axis('off')
#plt.tight_layout()
#plt.show()

##########
#take crop but snap to edge of avaiable regions