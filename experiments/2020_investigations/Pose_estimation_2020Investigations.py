import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import pdb
import shutil
import json
import keras
#import dill as pickle
import pickle
from pickle import dump, load
import tensorflow as tf
from tensorflow import keras
from tensorflow.compat.v1 import set_random_seed
from keras import backend as K

from logging import ERROR
tf.compat.v1.logging.set_verbosity(ERROR)
tf.compat.v1.disable_eager_execution()
import threading

import random
from random import seed,choice
from PIL import Image
from pickle import dump,load

from skimage.transform import AffineTransform, warp,rotate,SimilarityTransform
from skimage import util,img_as_ubyte,img_as_float,exposure
from matplotlib import pyplot as plt


from keras.applications.inception_v3 import InceptionV3,preprocess_input

#from inception_v3_bnf16 import InceptionV3

#from keras.applications.inception_v3 import preprocess_input
#from CBAMinception_v3 import InceptionV3

#from keras.applications.efficientnet import EfficientNetB0,EfficientNetB3,EfficientNetB4,EfficientNetB5,EfficientNetB6,EfficientNetB7,preprocess_input

from keras.models import Model,model_from_json
from keras.preprocessing import image
from tensorflow.keras.utils import Sequence,plot_model
from keras.utils.vis_utils import model_to_dot
from IPython.display import SVG
from keras.preprocessing import image as keras_image
from keras.layers import Lambda,Input
from keras.layers.advanced_activations import PReLU
from keras.callbacks import LambdaCallback,Callback,CSVLogger,History,ModelCheckpoint,ProgbarLogger,ReduceLROnPlateau,TensorBoard,EarlyStopping
from keras.regularizers import l1,l2

from uncertainty_layer_two import LossWeighter,CurrentLossWeight

from main_lr_finder import LRFinder
from cyclical_learning_rate_withRLRoP import CyclicLR

import numpy as np
from scipy import linalg
from submission import SubmissionWriter

from PlotPredictions import PlotinPLotly

from rotation import Rotation as R
from tqdm import tqdm
import os



from PIL import ImageEnhance

#config = tf.ConfigProto()
#config.gpu_options.allow_growth = True
#session = tf.Session(config=config)

selection=3

if selection==1:
    resolution='images'
elif selection==2:
    resolution='preprocessed_images' #224
elif selection==3:
    resolution='preprocessed_images_1200'
elif selection==4:
    resolution='preprocessed_images_600'    
elif selection==5:
    resolution='preprocessed_images_300'      
else:
    resolution='images'
    
edgemethod='edge'
#‘constant’, ‘edge’, ‘symmetric’, ‘reflect’, ‘wrap’

#file_prefix='2020_Incepv3_1200_bottleneckresolution_'
file_prefix='2020_Iv3_1200_bnatend_'


f16=False
if f16:
    K.set_floatx('float16')
    # default is 1e-7 which is too small for float16.  Without adjusting the epsilon, we will get NaN predictions because of divide by zero problems
    K.set_epsilon(1e-4)#loss function breaking

img_size=600

seed_int=50

checkLR=False
restart_checkpoint_path=None#'2020_Iv3_1200_bnatend_600weights.42-4.32.hdf5'#None#'2020_Incepv3_1200_attmpt2_600weights.44--0.96.hdf5'#None#'2020_Incepv3_1200_attmpt2_600weights.25-0.43.hdf5'#None#'2020_Inceptionv3_1200-600epoch_600weights.63-0.00.hdf5'#'2020_Inceptionv3_1200-600epoch_600weights.28-4.89.hdf5'#None#'2020_Inceptionv3_200epoch_600weights.66-1.57.hdf5'
Training=True
submit=False
predict=False
evaluatemodel=False

#Custom imports
import importlib.machinery

# loader = importlib.machinery.SourceFileLoader('net', 'C:\DeepLearningFolder\Pose Estimation\UrsoNet-master\net.py')
# mod = loader.load_module()
# from net import UrsoNet

# =============================================================================
# loader = importlib.machinery.SourceFileLoader('dl_bot', './DLImports/dl_bot.py')
# mod = loader.load_module()
# from dl_bot import DLBot
# 
# loader = importlib.machinery.SourceFileLoader('telegram_bot_callback', './DLImports/telegram_bot_callback.py')
# mod = loader.load_module()
# from telegram_bot_callback import TelegramBotCallback
# 
# telegram_token = "1207870703:AAEk_5xHqkOEtZ9iPXhbsg970ulf-6zbEtU"  # replace TOKEN with your bot's token
# 
# #  user id is optional, however highly recommended as it limits the access to you alone.
# telegram_user_id = 928643452  # replace None with your telegram user id (integer):
# 
# # Create a DLBot instance
# bot = DLBot(token=telegram_token, user_id=telegram_user_id)
# =============================================================================


#Define Optimizer
loader = importlib.machinery.SourceFileLoader('myoptimizers', './DLImports/myoptimizers.py')
mod = loader.load_module()
from myoptimizers import Yogi,FTML,Padam,AdamW,AccumOptimizer,RAdam,AdamAccumulate

#Define Attention Module
loader = importlib.machinery.SourceFileLoader('attentionmodules', './DLImports/attentionmodules.py')
mod = loader.load_module()
from attentionmodules import attach_attention_module

class Camera:

    """" Utility class for accessing camera parameters. """

    fx = 0.0176  # focal length[m]
    fy = 0.0176  # focal length[m]
    nu = img_size#224  # number of horizontal[pixels]
    nv = img_size#224  # number of vertical[pixels]
    ppx = 5.86e-6*1200/nu # horizontal pixel pitch[m / pixel]
    ppy = ppx  # vertical pixel pitch[m / pixel]
    fpx = fx / ppx  # horizontal focal length[pixels]
    fpy = fy / ppy  # vertical focal length[pixels]
    k = [[fpx,   0, nu / 2],
         [0,   fpy, nv / 2],
         [0,     0,      1]]
    
    K = np.array(k)

def evaluate(model, dataset, append_submission, dataset_root):

    """ Running evaluation on test set, appending results to a submission. """

    with open(os.path.join(dataset_root, dataset + '.json'), 'r') as f:
        image_list = json.load(f)

    print('Running evaluation on {} set...'.format(dataset))

    for img in tqdm(image_list):
        img_path = os.path.join(dataset_root,resolution, dataset, img['filename'])
        if selection==1:
            pil_img = image.load_img(img_path, target_size=(img_size,img_size))
        elif selection==2:
            pil_img = image.load_img(img_path, target_size=(img_size,img_size))
        else:
            pil_img = image.load_img(img_path, target_size=(img_size,img_size),interpolation='lanczos')
  
        x = image.img_to_array(pil_img)
        x = preprocess_input(x,backend = K)
        x = np.expand_dims(x, 0)
        output = model.predict(x)
        test=np.concatenate((output[0],output[1]),axis=None)
        append_submission(img['filename'], test[0:4], test[4:7])
        #append_submission(img['filename'], output[0][0:4], output[1][0:3])
        
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

def project(k, q, r):

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
        points_image_plane = k.dot(points_camera_frame)

        x, y = (points_image_plane[0], points_image_plane[1])
        return x, y  
    
class threadsafe_iter:
    """Takes an iterator/generator and makes it thread-safe by
    serializing call to the `next` method of given iterator/generator.
    """
    def __init__(self, it):
        self.it = it
        self.lock = threading.Lock()

    def __iter__(self):
        return self
    
    def length(self):
        return len(self.it)    
    
    def next(self):
        with self.lock:
            return self.it.next()    
        
class KerasDataGenerator(Sequence):#Sequence

        """ DataGenerator for Keras to be used with fit_generator (https://keras.io/models/sequential/#fit_generator)"""

        def __init__(self, 
                     preprocessor, 
                     label_list, 
                     speed_root, 
                     batch_size=32, 
                     dim=(224, 224), 
                     n_channels=3, 
                     shuffle=False,
                     augment=False,
                     featurewise_center=False,
                     samplewise_center=False,
                     featurewise_std_normalization=False,
                     samplewise_std_normalization=False,
                     zca_whitening=False,
                     data_format=None):

            # loading dataset
            self.image_root = os.path.join(speed_root,resolution, 'train')

            # Initialization
            self.preprocessor = preprocessor
            self.dim = dim
            self.batch_size = batch_size
            self.labels = self.labels = {label['filename']: {'q': label['q_vbs2tango'], 'r': label['r_Vo2To_vbs_true']}
                                         for label in label_list}
            self.list_IDs = [label['filename'] for label in label_list]
            self.n_channels = n_channels
            self.shuffle = shuffle
            self.augment = augment
            self.indexes = None
            self.on_epoch_end()
            self.width = dim[0]
            self.height = dim[1]
            
            #Camera Parameters
            self.fx = 0.0176  # focal length[m]
            fy = 0.0176  # focal length[m]
            nu = self.width  # number of horizontal[pixels]
            nv = self.height  # number of vertical[pixels]
            self.ppx = 5.86e-6*1200/nu # horizontal pixel pitch[m / pixel]
            ppy = self.ppx  # vertical pixel pitch[m / pixel]
            self.fpx = self.fx / self.ppx  # horizontal focal length[pixels]
            fpy = fy / ppy  # vertical focal length[pixels]
            k = [[self.fpx,   0, nu / 2],
                 [0,   fpy, nv / 2],
                 [0,     0,      1]]
            
            self.K = np.array(k)      
            
            self.featurewise_center = featurewise_center
            self.samplewise_center = samplewise_center
            self.featurewise_std_normalization = featurewise_std_normalization
            self.samplewise_std_normalization = samplewise_std_normalization
            self.zca_whitening = zca_whitening
            
            if ((self.featurewise_center) or (self.samplewise_center) or (self.featurewise_std_normalization) or (self.samplewise_std_normalization) or (self.zca_whitening)):
                self.standardize_enabled=True
            else:
                self.standardize_enabled=False
            
            self.mean = None
            self.std = None
            self.principal_components = None       
            
            if data_format is None:
                data_format = K.image_data_format()
            
            if data_format == 'channels_first':
                self.channel_axis = 1
                self.row_axis = 2
                self.col_axis = 3
            if data_format == 'channels_last':
                self.channel_axis = 3
                self.row_axis = 1
                self.col_axis = 2            

        def __len__(self):

            """ Denotes the number of batches per epoch. """

            return int(np.ceil(len(self.list_IDs) / self.batch_size))#do i need to add ceil here?

        def __getitem__(self, index):

            """ Generate one batch of data """

            # Generate indexes of the batch
            indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

            # Find list of IDs
            list_IDs_temp = [self.list_IDs[k] for k in indexes]

            # Generate data
            X, Y = self.__data_generation(list_IDs_temp)

            return X, Y

        def on_epoch_end(self):

            """ Updates indexes after each epoch """

            self.indexes = np.arange(len(self.list_IDs))
            if self.shuffle:
                np.random.shuffle(self.indexes)
                
        def hemisphere_q(self,q):
            if q[0] <=0:
                return [qval*-1 for qval in q]
            else:
                return q

        def __data_generation(self, list_IDs_temp):

            """ Generates data containing batch_size samples """

            # Initialization
            X = np.empty((self.batch_size, *self.dim, self.n_channels), dtype=np.float32)
            y1 = np.empty((self.batch_size, 4), dtype=np.float32)
            y2 = np.empty((self.batch_size, 3), dtype=np.float32)#change to 4
            dummy = np.empty((self.batch_size, 3), dtype=np.float32)
            
            # Generate data
            for i, ID in enumerate(list_IDs_temp):

                q, r = self.labels[ID]['q'], self.labels[ID]['r']
                q=self.hemisphere_q(q)
                
                img_path = os.path.join(self.image_root, ID)
                #img = keras_image.load_img(img_path.replace('jpg','png'), target_size=self.dim)
                img = keras_image.load_img(img_path, target_size=self.dim,interpolation='lanczos')
                
                #img = Image.open(img_path).resize(self.dim,Image.LANCZOS).convert('RGB')
                #img=np.array(img)#Image type to unit8
                #pdb.set_trace()
                x = keras_image.img_to_array(img)#Input as nfloat64 ->output as float32
                if self.augment:
                    #pdb.set_trace()
                    q,r,x = self._augment(q,r,x)#Input as numpy array of uint8 ->output as float64
                else:
                    pass
                #pdb.set_trace()
                #if self.standardize_enabled:
                #    x = self.standardize(x)
                #pdb.set_trace()
                x = self.preprocessor(x)#will scale pixels between -1 and 1, sample-wise 
                #x = self.preprocessor(x,backend=K)#resnext  
                #pdb.set_trace()
                X[i,] = x 
                y1[i],y2[i] = q,r#np.concatenate([q, r])
            
            #Y=np.concatenate((y1, y2), axis=1)
            if Training:
                return [X, y1,y2], dummy
            else:
                return X, [y1,y2]#, dummy

        def _augment(self, q,r,image):
            
            rotate_aug_enabled=True
            
            #Random translation
            if bool(random.getrandbits(1)):
                #grid_translation_choice=np.array([200,400,600,800,1000])*self.width/1200#for 1200 resolution
                grid_translation_choice=np.array([200,300,400,500,600,700,800,900,1000])*self.width/1200
                #grid_translation_choice=np.array([100,150,200,250,300,350,400,450,500])*self.width/600

                new_x=np.random.choice(grid_translation_choice)
                new_y=np.random.choice(grid_translation_choice)
                #new_x,new_y=np.random.uniform(100,self.width-100),np.random.uniform(100,self.height-100)#To adjust limits
                old_x,old_y=project(self.K,q, r)
                old_x,old_y=old_x[0],old_y[0]
                translate_x=old_x-new_x
                translate_y=old_y-new_y
                
                scale=r[2]/self.fx #Calculate scale of camera line translation
                r=[(new_x-self.width/2)*self.ppx*scale,(new_y-self.height/2)*self.ppx*scale,self.fpx*self.ppx*scale]#compute new position vector
                image=self.shift(image,(translate_x,translate_y))#Input as numpy array of uint8 - > Output as float64
    
            #Random rotation
            if rotate_aug_enabled:
                angles=[0,45,90,135,180,225,270,315]
                #angles=[0,90,180,270]
                angle=np.random.choice(angles)
                if angle!=0:
                    image=rotate(image,-angle,mode=edgemethod,preserve_range=True) #Input as numpy array of float64 - > Output as float64
        
                    q = self.q_rotation(q,angle)
                    r = self.z_rotation(r,angle)
                else:
                    pass
            
            return q,r,image
        
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
        
        def shift(self,image, vector):
            transform = AffineTransform(translation=vector)
            shifted = warp(image, transform, mode=edgemethod, preserve_range=True)
            return shifted

        def standardize(self, x):
            """Apply the normalization configuration to a batch of inputs.
            # Arguments
                x: batch of inputs to be normalized.
            # Returns
                The inputs, normalized.
            """
            # x is a single image, so it doesn't have image number at index 0
            img_channel_axis = self.channel_axis - 1
            if self.samplewise_center:
                x -= np.mean(x, axis=img_channel_axis, keepdims=True)
            if self.samplewise_std_normalization:
                x /= (np.std(x, axis=img_channel_axis, keepdims=True) + 1e-7)
    
            if self.featurewise_center:
                if self.mean is not None:
                    x -= self.mean
                else:
                    warnings.warn('This ImageDataGenerator specifies '
                                  '`featurewise_center`, but it hasn\'t'
                                  'been fit on any training data. Fit it '
                                  'first by calling `.fit(numpy_data)`.')
            if self.featurewise_std_normalization:
                if self.std is not None:
                    x /= (self.std + 1e-7)
                else:
                    warnings.warn('This ImageDataGenerator specifies '
                                  '`featurewise_std_normalization`, but it hasn\'t'
                                  'been fit on any training data. Fit it '
                                  'first by calling `.fit(numpy_data)`.')
            if self.zca_whitening:
                if self.principal_components is not None:
                    flatx = np.reshape(x, (x.size))
                    whitex = np.dot(flatx, self.principal_components)
                    x = np.reshape(whitex, (x.shape[0], x.shape[1], x.shape[2]))
                else:
                    warnings.warn('This ImageDataGenerator specifies '
                                  '`zca_whitening`, but it hasn\'t'
                                  'been fit on any training data. Fit it '
                                  'first by calling `.fit(numpy_data)`.')
            return x    
        
        def fit(self, x, seed=None):
            """Fits internal statistics to some sample data.
            Required for featurewise_center, featurewise_std_normalization
            and zca_whitening.
            # Arguments
                x: Numpy array, the data to fit on. Should have rank 4.
                    In case of grayscale data,
                    the channels axis should have value 1, and in case
                    of RGB data, it should have value 3.
                augment: Whether to fit on randomly augmented samples
                rounds: If `augment`,
                    how many augmentation passes to do over the data
                seed: random seed.
            # Raises
                ValueError: in case of invalid input `x`.
            """
            x = np.asarray(x, dtype=K.floatx())
            if x.ndim != 4:
                raise ValueError('Input to `.fit()` should have rank 4. '
                                 'Got array with shape: ' + str(x.shape))
            if x.shape[self.channel_axis] not in {1, 3, 4}:
                raise ValueError(
                    'Expected input to be images (as Numpy array) '
                    'following the data format convention "' + self.data_format + '" '
                    '(channels on axis ' + str(self.channel_axis) + '), i.e. expected '
                    'either 1, 3 or 4 channels on axis ' + str(self.channel_axis) + '. '
                    'However, it was passed an array with shape ' + str(x.shape) +
                    ' (' + str(x.shape[self.channel_axis]) + ' channels).')
    
            if seed is not None:
                np.random.seed(seed)
    
            x = np.copy(x)
    
            if self.featurewise_center:
                self.mean = np.mean(x, axis=(0, self.row_axis, self.col_axis))
                broadcast_shape = [1, 1, 1]
                broadcast_shape[self.channel_axis - 1] = x.shape[self.channel_axis]
                self.mean = np.reshape(self.mean, broadcast_shape)
                x -= self.mean
    
            if self.featurewise_std_normalization:
                self.std = np.std(x, axis=(0, self.row_axis, self.col_axis))
                broadcast_shape = [1, 1, 1]
                broadcast_shape[self.channel_axis - 1] = x.shape[self.channel_axis]
                self.std = np.reshape(self.std, broadcast_shape)
                x /= (self.std + K.epsilon())
    
            if self.zca_whitening:
                flat_x = np.reshape(x, (x.shape[0], x.shape[1] * x.shape[2] * x.shape[3]))
                sigma = np.dot(flat_x.T, flat_x) / flat_x.shape[0]
                u, s, _ = linalg.svd(sigma)
                self.principal_components = np.dot(np.dot(u, np.diag(1. / np.sqrt(s + 10e-7))), u.T)        
   
#Losses

def geodesic_angular_loss(y_true,y_pred):
    #Validated as working Has max at 3.141 and min at 0.0
    y_pred=K.l2_normalize(y_pred,axis = -1)
    y_true=K.l2_normalize(y_true,axis = -1)
    dotprod=K.sum(y_true*y_pred,axis=-1,keepdims=True)
    theta=2.0*tf.math.acos(dotprod)
    return theta

def norm_penalty(y_true,y_pred):
    norm = K.sqrt(K.sum(K.square(y_pred), axis=-1))
    penalty = K.square(1.0 - norm)
    return penalty

def my_loss(y_true, y_pred):

    mse=K.mean(K.square(y_pred - y_true), axis=-1)


    y_true = K.l2_normalize(y_true, axis=-1)
    y_pred = K.l2_normalize(y_pred, axis=-1)

    cosine_prox=-K.sum(y_true * y_pred, axis=-1)

    mainloss=mse+cosine_prox
    return mainloss

def quaternion_distance_loss(y_pred, y_true):
    norm=K.sqrt(K.sum(y_pred*y_pred,axis=-1,keepdims=True))
    penalty = K.square(1.0 - norm)   
    
    #y_pred=K.l2_normalize(y_pred,axis = -1)
    dotprod=K.sum(y_pred*y_true,axis=-1,keepdims=True)
    d = K.square(dotprod)

    loss=d + 10.0*penalty
    return K.maximum(loss,K.epsilon())

def unit_norm_loss(y_pred, y_true):
    norm=K.sum(y_pred*y_pred,axis=-1,keepdims=True)
    penalty = K.pow(1.0 - K.minimum(norm,1.0-K.epsilon()),2)
    return K.sum(penalty, axis=-1)

def euclidean_distance_loss(y_true, y_pred):
    return K.sqrt(K.sum(K.square(y_pred - y_true), axis=-1))

def geodesic_loss(y_true,y_pred):
    #Validated as working Has max at 1.0 and min at 0.0
    y_pred=K.l2_normalize(y_pred,axis = -1)
    y_true=K.l2_normalize(y_true,axis = -1)
    dotprod=K.sum(y_true*y_pred,axis=-1,keepdims=True)
    result = 1.0 - K.square(dotprod)
    return result

def calc_q_loss(x):
    true,pred=x
    return geodesic_loss(true,pred)

def r2_loss(y_true, y_pred):
    SS_res =  K.sum(K.square(y_true - y_pred)) 
    SS_tot = K.sum(K.square(y_true - K.mean(y_true))) 
    #return ( 1 - SS_res/(SS_tot + K.epsilon()) )
    return (SS_res/(SS_tot + K.epsilon()) )#loss conversion

def ev(y_true, y_pred):#explained_variance
    nom =  K.var(y_true - y_pred) 
    denom = K.var(y_true)
    return ( 1 - nom/(denom + K.epsilon()) )

def ev_loss(y_true, y_pred):#explained_variance
    nom =  K.var(y_true - y_pred) 
    denom = K.var(y_true)
    return (nom/(denom + K.epsilon()) )

def mse(y_true, y_pred):
    #return K.mean(K.square(K.clip(y_pred, 0., 100.) - K.clip(y_true, 0., 100.)), axis=-1,keepdims=True)

    return K.mean(K.square(y_pred - y_true), axis=-1,keepdims=True)

def deltarloss(y_true, y_pred):

    delta=y_pred-y_true   
    #delta=K.clip(y_pred, 0., 100.) - K.clip(y_true, 0., 100.) 
    delta_normsquared=K.sum(delta*delta,axis=-1,keepdims=True)    
    delta=K.l2_normalize(delta, axis=-1)
    y_true = K.l2_normalize(y_true, axis=-1)
    sbar=1.0 - K.sum(y_true*delta,axis=-1,keepdims=True)
    
    return delta_normsquared*sbar

def calc_r_loss(x):
    true,pred=x
    return mse(true,pred)

def calc_deltar_loss(x):
    true,pred=x
    return deltarloss(true,pred)

def MainLoss(y_true,y_pred):
    return y_pred[:,0]

def QLoss(y_true,y_pred):
    return y_pred[:,1]

def RLoss(y_true,y_pred):
    return y_pred[:,2]

def RTwoLoss(y_true,y_pred):
    return y_pred[:,3]

def build_model(Training=True,opt=None):
    print('Creating Model...')     
        
    K.set_learning_phase(Training)
    input_shape=(img_size, img_size, 3)
        
    ####InceptionV3
    pretrained_model = InceptionV3(weights="imagenet", include_top=False,input_shape=input_shape,pooling='avg')       
    #pretrained_model = InceptionV3(weights="imagenet", include_top=False,input_shape=input_shape,pooling='avg',attention_module='cbam_block')       

    # #EfficeintNetB0
    #pretrained_model = EfficientNetB0(weights="imagenet", include_top=False,input_shape=input_shape,pooling='avg',backend = K,layers = keras.layers, models = keras.models, utils = keras.utils)
    
    # #EfficeintNetB3
    #pretrained_model = EfficientNetB3(weights="imagenet", include_top=False,input_shape=input_shape,pooling='avg',backend = K,layers = keras.layers, models = keras.models, utils = keras.utils)
    
    # #EfficeintNetB4
    #pretrained_model = EfficientNetB4(weights="imagenet", include_top=False,input_shape=input_shape,pooling='avg',backend = K,layers = keras.layers, models = keras.models, utils = keras.utils)
    
    # #EfficeintNetB5
    # pretrained_model = EfficientNetB5(weights="imagenet", include_top=False,input_shape=input_shape,pooling='avg',backend = K,layers = keras.layers, models = keras.models, utils = keras.utils)
    
    # #EfficeintNetB6
    # pretrained_model = EfficientNetB6(weights="imagenet", include_top=False,input_shape=input_shape,pooling='avg',backend = K,layers = keras.layers, models = keras.models, utils = keras.utils)
    
    #EfficeintNetB7
    #pretrained_model = EfficientNetB7(weights="imagenet", include_top=False,input_shape=input_shape,pooling='avg',backend = K,layers = keras.layers, models = keras.models, utils = keras.utils)
    
    x = pretrained_model.output  
    
    #x = attach_attention_module(x, 'cbam_block',module_id=11)
    #x = keras.layers.GlobalAveragePooling2D(name='avg_pool')(x)

    #x = keras.layers.Conv2D(128, (3, 3), padding='SAME', strides=(2, 2), name='bottleneck_layer')(x)
    
    #nr_features = int(128 * 9* 9)
    #x = keras.layers.Reshape((nr_features,))(x)



    # Adding new trainable hidden and output layers to the model
    #rvecfork = keras.layers.Dense(2048, activation="relu")(x)

    rvecfork = keras.layers.Dense(2048)(x)
    rvecfork = keras.layers.BatchNormalization(name ='head_bn')(rvecfork)
    rvecfork = keras.layers.Activation('relu')(rvecfork)   


    rvecfork = keras.layers.Dropout(0.2)(rvecfork)
    qvec=keras.layers.Dense(4, activation="tanh",name='q1',kernel_regularizer=l2(0.01))(rvecfork)
    qvec = Lambda(lambda  x: K.l2_normalize(x,axis=-1),name='q')(qvec)#,name="q_pred"
    rvec=keras.layers.Dense(3, activation="linear",name='r')(rvecfork)
    
    pred_model_final = keras.models.Model(inputs=pretrained_model.input, outputs=[qvec,rvec])
    q_pred=pred_model_final.outputs[0]
    r_pred=pred_model_final.outputs[1]
    
    q_true=Input((4,))
    r_true=Input((3,))
    
    qLoss = Lambda(calc_q_loss, name='loss_q')([q_true, q_pred])
    rLoss = Lambda(calc_r_loss, name='loss_r')([r_true, r_pred])
    deltarLoss = Lambda(calc_deltar_loss, name='deltaloss_r')([r_true, r_pred])
    weightedLoss = LossWeighter(name='weighted_loss')([qLoss,rLoss,deltarLoss])
    
    merged_output = keras.layers.concatenate([weightedLoss,qLoss,rLoss,deltarLoss], axis=-1)
    model_final = Model([pred_model_final.input, q_true, r_true], merged_output)

    if Training:
        model_final.compile(optimizer=opt,loss=MainLoss,metrics=[QLoss,RLoss,RTwoLoss])
    else:
        model_final=pred_model_final
        model_final.compile(loss={'q': geodesic_loss, 'r': mse},loss_weights=[2.0,1.0], optimizer=opt,metrics={'q': [geodesic_angular_loss], 'r': [r2_loss,ev_loss]})#,metrics=['accuracy','cosine_proximity','mean_squared_error',lr_metric])
        
    return model_final



def main(speed_root, epochs, batch_size):

    # Loading and splitting dataset
    with open(os.path.join(speed_root, 'train' + '.json'), 'r') as f:
        label_list = json.load(f)
        
    # with open(os.path.join(speed_root, 'real' + '.json'), 'r') as f:
    #     label_list_real = json.load(f)

    # label_list=label_list+label_list_real
    
    random.shuffle(label_list)#Shuffle the label list first
    train_labels = label_list[:int(len(label_list)*.7)]
    validation_labels = label_list[int(len(label_list)*.7):]

    # Data generators for training and validation
    training_generator = KerasDataGenerator(preprocess_input, 
                                            train_labels, 
                                            speed_root, 
                                            dim=(img_size,img_size),
                                            batch_size=batch_size,
                                            n_channels=3,
                                            shuffle=True,#should probably make this true as last batch always has a lower mean
                                            augment=True,
                                            samplewise_center=False,
                                            featurewise_center=False,#If featurewise images will be quite faint due to background majority
                                            samplewise_std_normalization=False,
                                            featurewise_std_normalization=False,
                                            zca_whitening=False)#Memory will run out during dot product
   
    validation_generator = KerasDataGenerator(preprocess_input, 
                                              validation_labels, 
                                              speed_root, 
                                              dim=(img_size,img_size),
                                              batch_size=batch_size,
                                              n_channels=3,
                                              shuffle=False,
                                              augment=False,
                                              samplewise_center=False,
                                              featurewise_center=False,#If featurewise images will be quite faint due to background majority
                                              samplewise_std_normalization=False,
                                              featurewise_std_normalization=False,)   
    
    opt = AdamAccumulate(learning_rate=1e-6, beta_1=0.9, beta_2=0.999,epsilon=None, decay=0., amsgrad=False, accum_iters=5)
        
    def get_lr_metric(optimizer):
        def lr(y_true, y_pred):
            return optimizer.learning_rate
        return lr
        
    def get_init_epoch(restart_checkpoint_path=None):
        return int(restart_checkpoint_path.split('.')[1].split('-')[0])
    
    #Build Model
    model_final=build_model(Training,opt)

    # Load checkpoint:
    if restart_checkpoint_path is not None:
        print('Loading Model...') 
        # Load model:
        # json_file = open(file_prefix+str(img_size)+'model_final.json', 'r')
        # loaded_model_json = json_file.read()
        # json_file.close()
        # keras.backend.set_learning_phase(Training)#for training all
        # model_final = model_from_json(loaded_model_json,custom_objects={'LossWeighter': LossWeighter,
        #                                                                 'geodesic_loss':geodesic_loss,
        #                                                                 'mse':mse,
        #                                                                 'deltarloss':deltarloss})

        model_final.load_weights(restart_checkpoint_path, by_name=True)

        # Finding the epoch index from which we are resuming
        initial_epoch = get_init_epoch(restart_checkpoint_path)

        with open(file_prefix+str(img_size)+'random_gen_state.obj', 'rb') as f:
            np.random.set_state(load(f))        

    else:
        
        # print("Saving model to disk")
        # model_json = model_final.to_json()
        # with open(file_prefix+str(img_size)+"model_final.json", "w") as json_file:
        #     json_file.write(model_json)    

        initial_epoch = 0        

    #Plot the model
    plot_model(model_final, to_file=file_prefix+str(img_size)+'model_final.png', show_shapes=True, show_layer_names=True)
    #model_final.summary()        
        

    # # check the layers by name
    # startindex=312
    # endindex=323
    # for i,layer in enumerate(model_final.layers):
    #     layer.trainable=False
    #     if i >= startindex and i<=endindex:
    #         layer.trainable=True
    #     print(i,layer.name,layer.trainable)

    # model_final.compile(optimizer=opt,loss=MainLoss,metrics=[QLoss,RLoss,RTwoLoss])

    if checkLR:

        lr_finder = LRFinder(model_final)
        lr_finder.find_generator(training_generator, start_lr=1e-9, end_lr=1e-6, epochs=1, steps_per_epoch=None)
        lr_finder.plot_loss(n_skip_beginning=5, n_skip_end=1)

#Main model training
    if Training:

        #Define Callbacks
        BestModel = ModelCheckpoint(filepath=file_prefix+str(img_size)+'weights.{epoch:02d}-{val_loss:.2f}.hdf5', monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=True, mode='min', period=1)
        
        csv_logger = CSVLogger(file_prefix+str(img_size)+'History_log.csv', append=True, separator=',')

        early_stopping = EarlyStopping(patience=10, verbose=1)

        current_loss=CurrentLossWeight(model_final)

        class myTerminateOnNaN(Callback):
            """Callback that terminates training when a NaN loss is encountered.
            """

            def on_batch_end(self, batch, logs=None):
                logs = logs or {}
                loss = logs.get('loss')
                if loss is not None:
                    if np.isnan(loss) or np.isinf(loss):
                        print('Batch %d: Invalid loss, terminating training' % (batch))
                        self.model.stop_training = True
        tnan=myTerminateOnNaN()

# =============================================================================
#         tboard_epoch=TensorBoard(log_dir='./logs_'+file_prefix+str(img_size), 
#                        histogram_freq=0, 
#                        batch_size=32, 
#                        write_graph=False, 
#                        write_grads=False, 
#                        write_images=False, 
#                        embeddings_freq=0, 
#                        embeddings_layer_names=None, 
#                        embeddings_metadata=None, 
#                        embeddings_data=None, 
#                        update_freq='epoch')   
# =============================================================================
        
        clr = CyclicLR(base_lr=1e-4,max_lr=1e-3,#1e-5 to 2e-5
                       step_size=1050,
                       mode='triangular2',
                       gamma=1.,
                       scale_fn=None,
                       scale_mode='cycle',
                       monitor='val_loss',### 
                       factor=0.5, 
                       patience=6,
                       verbose=1, 
                       reduction_mode='min', 
                       min_delta=1e-4, 
                       cooldown=2, 
                       min_lr=1e-6)

        # Create a TelegramBotCallback instance
        #telegram_callback = TelegramBotCallback(bot)

        if restart_checkpoint_path is not None and os.path.exists(file_prefix+str(img_size)+'best_valloss.pickle'):
            with open(file_prefix+str(img_size)+'best_valloss.pickle', 'rb') as f:
                best = pickle.load(f)
                BestModel.best = best

        def save_chkpt_cb():
            with open(file_prefix+str(img_size)+'best_valloss.pickle', 'wb') as f:
                pickle.dump(BestModel.best, f, protocol=pickle.HIGHEST_PROTOCOL)

        save_chkpt_cb_callback = LambdaCallback(on_epoch_end=lambda epoch, logs: save_chkpt_cb())
            

        # # check the layers by name
        # startindex=312
        # endindex=323
        # for i,layer in enumerate(model_final.layers):
        #     layer.trainable=False
        #     if i >= startindex and i<=endindex:
        #         layer.trainable=True
        #     print(i,layer.name,layer.trainable)

        # model_final.compile(optimizer=opt,loss=MainLoss,metrics=[QLoss,RLoss,RTwoLoss])

#        # Training the model (transfer learning)
# =============================================================================
#         history = model_final.fit_generator(
#                                 training_generator,
#                                 epochs=epochs,
#                                 validation_data=validation_generator,
#                                 workers=1,
#                                 max_queue_size=40,
#                                 initial_epoch=initial_epoch,
#                                 callbacks=[tnan,current_loss,clr,BestModel,csv_logger,early_stopping,save_chkpt_cb_callback])
# =============================================================================
        history = model_final.fit(x=training_generator, 
                                  epochs=epochs, 
                                  verbose=2, 
                                  callbacks=[tnan,current_loss,clr,BestModel,early_stopping,save_chkpt_cb_callback],
                                  validation_data=validation_generator, 
                                  initial_epoch=initial_epoch, 
                                  steps_per_epoch=None, 
                                  validation_steps=None, 
                                  validation_freq=1, 
                                  max_queue_size=40, 
                                  workers=4, 
                                  use_multiprocessing=False)
        with open(file_prefix+str(img_size)+'random_gen_state.obj', 'wb') as f:
            dump(np.random.get_state(), f)
    
    if evaluatemodel:
        evaluations = model_final.evaluate_generator(validation_generator, max_queue_size=40, workers=4, use_multiprocessing=False, verbose=1)
        print('Validation Losses: '+ str(evaluations))
####Predict on Validation Data  
    if predict:
        predictions = model_final.predict_generator(generator=validation_generator,
                                                    workers=1,
                                                    max_queue_size=40,
                                                    verbose=1)
        #np.save('predictions', predictions)
        
        #fileObject = open('predictions.pkl', 'wb')
        #dump(predictions,fileObject)
        
        PlotinPLotly.send_to_plotly_results(train_labels,validation_labels,predictions)

#####Generating submission
    if submit:
        submission = SubmissionWriter()
        evaluate(model_final, 'test', submission.append_test, speed_root)
        evaluate(model_final, 'real_test', submission.append_real_test, speed_root)
        submission.export(suffix=file_prefix+str(img_size)+'results')      

    return model_final


def seed_everything(SEED=seed_int):
    random.seed(SEED)
    os.environ['PYTHONHASHSEED'] = str(SEED)
    np.random.seed(SEED)
    #ia.seed(SEED)#must be commented otherwise not reproducable
    tf.random.set_seed(SEED)
    
seed_everything()

dataset=r'C:/DeepLearningFolder/Pose Estimation/speed/'

model_final=main(dataset, int(200), int(16))