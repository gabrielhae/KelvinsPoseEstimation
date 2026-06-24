import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'

import json
import keras
import tensorflow as tf
from tensorflow import set_random_seed
from keras import backend as K
keras.backend.clear_session() 

import threading

import random
from random import seed,choice
from PIL import Image
from pickle import dump,load
#from utils import KerasDataGenerator

from skimage.transform import AffineTransform, warp,rotate,SimilarityTransform
from skimage import util,img_as_ubyte,img_as_float,exposure
from matplotlib import pyplot as plt

from keras.applications.resnet50 import ResNet50,preprocess_input
from keras.models import Model,model_from_json
from keras.preprocessing import image
from keras.utils import Sequence,plot_model
from keras.utils.vis_utils import model_to_dot
from IPython.display import SVG
from keras.preprocessing import image as keras_image
from keras.layers import Lambda,Input
from keras.layers.advanced_activations import PReLU
from keras.callbacks import Callback,CSVLogger,History,ModelCheckpoint,ProgbarLogger,ReduceLROnPlateau,TensorBoard,EarlyStopping

from main_lr_finder import LRFinder
from cyclical_learning_rate_withRLRoP import CyclicLR

#from tensorflow.keras.applications.resnet50 import preprocess_input
#from tensorflow.keras.preprocessing import image
import numpy as np
from scipy import linalg
from submission import SubmissionWriter

from plotlyplotting import PlotinPLotly

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

from rotation import Rotation as R
from tqdm import tqdm
import os

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

selection=2

if selection==1:
    resolution='images'
elif selection==2:
    resolution='preprocessed_images_1200'
else:
    resolution='images'
    
edgemethod='edge'
#‘constant’, ‘edge’, ‘symmetric’, ‘reflect’, ‘wrap’

input_img_size=(224,224)

seed_int=5

""" 
    Example script demonstrating training on the SPEED dataset using Keras.
    Usage example: python keras_example.py --dataset [path to speed] --epochs [num epochs] --batch [batch size]
"""
class Camera:

    """" Utility class for accessing camera parameters. """

    fx = 0.0176  # focal length[m]
    fy = 0.0176  # focal length[m]
    nu = input_img_size[0]#224  # number of horizontal[pixels]
    nv = input_img_size[0]#224  # number of vertical[pixels]
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
            pil_img = image.load_img(img_path, target_size=input_img_size)
        elif selection==2:
            pil_img = image.load_img(img_path, target_size=input_img_size)
        else:
            pil_img = image.load_img(img_path, target_size=input_img_size)
            
        x = image.img_to_array(pil_img)
        x = preprocess_input(x)
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
            X, y = self.__data_generation(list_IDs_temp)

            return X, y

        def on_epoch_end(self):

            """ Updates indexes after each epoch """

            self.indexes = np.arange(len(self.list_IDs))
            if self.shuffle:
                np.random.shuffle(self.indexes)

        def __data_generation(self, list_IDs_temp):

            """ Generates data containing batch_size samples """

            # Initialization
            X = np.empty((self.batch_size, *self.dim, self.n_channels), dtype=np.float32)
            y1 = np.empty((self.batch_size, 4), dtype=float)
            y2 = np.empty((self.batch_size, 3), dtype=float)#change to 4

            # Generate data
            for i, ID in enumerate(list_IDs_temp):

                q, r = self.labels[ID]['q'], self.labels[ID]['r']
                
                img_path = os.path.join(self.image_root, ID)
                #img = keras_image.load_img(img_path, target_size=self.dim)
                
                img = Image.open(img_path).resize(self.dim,Image.LANCZOS).convert('RGB')
                img=np.array(img)#Image type to unit8
                if self.augment:
                    q,r,img = self._augment(q,r,img)#Input as numpy array of uint8 ->output as float64
                else:
                    pass
                
                x = keras_image.img_to_array(img)#Input as nfloat64 ->output as float32
                if self.standardize_enabled:
                    x = self.standardize(x)
                x = self.preprocessor(x)#will scale pixels between -1 and 1, sample-wise    
                X[i,] = x 
                y1[i],y2[i] = q,r#np.concatenate([q, r])

            return X, [y1, y2] 

        def _augment(self, q,r,image):
            
            #Random translation
            if bool(random.getrandbits(1)):
                grid_translation_choice=np.array([200,400,600,800,1000])*self.width/1200#for 1200 resolution
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
#            
            #Random rotation
            if bool(random.getrandbits(1)):
                angles=[45,90,135,180,225,270,315]
                #angles=[90,180,270]
                angle=np.random.choice(angles)
                image=rotate(image,-angle,mode=edgemethod,preserve_range=True) #Input as numpy array of float64 - > Output as float64
                
    #           Random Brightness, Gamma, Contrast            
                
                #if self.contrast_stretching: #####
                #if True:#np.random.random() < 0.5: #####
                    #p2, p98 = np.percentile(image, (1, 99)) #####
                    #image = exposure.rescale_intensity(image, in_range=(p2, p98))
                
                q = self.q_rotation(q,angle)
                r = self.z_rotation(r,angle)
            
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
            #transform = SimilarityTransform(translation=vector)#double check what is the correct transform to appy, identical
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
def geodesic_loss(y_pred, y_true):
    y_pred=K.l2_normalize(y_pred,axis = 1)
    tmp=K.sum(y_true*y_pred,axis=-1,keepdims=True)
    theta=2.0*tf.math.acos(K.clip(K.abs(tmp),-1.0+K.epsilon(),1.0-K.epsilon()))
    return theta

def euclidean_distance_loss(y_true, y_pred):
    return K.sqrt(K.sum(K.square(y_pred - y_true), axis=-1))

def main(speed_root, epochs, batch_size):

    """ Setting up data generators and model, training, and evaluating model on test and real_test sets. """

    # Loading and splitting dataset
    with open(os.path.join(speed_root, 'train' + '.json'), 'r') as f:
        label_list = json.load(f)
        
    #Determine Bounds
    
    
        
    random.shuffle(label_list)#Shuffle the label list first
    train_labels = label_list[:int(len(label_list)*.7)]
    validation_labels = label_list[int(len(label_list)*.7):]

    # Data generators for training and validation
    training_generator = KerasDataGenerator(preprocess_input, 
                                            train_labels, 
                                            speed_root, 
                                            dim=input_img_size,
                                            batch_size=batch_size,
                                            n_channels=3,
                                            shuffle=True,#should probably make this true as last batch always has a lower mean
                                            augment=False,
                                            samplewise_center=False,
                                            featurewise_center=False,#If featurewise images will be quite faint due to background majority
                                            samplewise_std_normalization=False,
                                            featurewise_std_normalization=False,
                                            zca_whitening=False)#Memory will run out during dot product
    
    #training_generator=threadsafe_iter(training_generator)
    
#    X,Y=training_generator[5]
#    training_generator.fit(X)
    
    validation_generator = KerasDataGenerator(preprocess_input, 
                                              validation_labels, 
                                              speed_root, 
                                              dim=input_img_size,
                                              batch_size=batch_size,
                                              n_channels=3,
                                              shuffle=False,
                                              augment=False,
                                              samplewise_center=True,
                                              featurewise_center=False,#If featurewise images will be quite faint due to background majority
                                              samplewise_std_normalization=True,
                                              featurewise_std_normalization=False,)

##   Test Augmentation View
#    X,Y=training_generator[0]
#    image_nparray=X[5]
#    q=Y[0][5]
#    r=Y[1][5]
#    fig, axes = plt.subplots(1, 1, figsize=(7, 7))
#    new_im = Image.fromarray(img_as_ubyte(np.divide(image_nparray, 255)))
#    axes.imshow(new_im)
#    xa, ya = project(Camera.K,q, r)
#    axes.arrow(xa[0], ya[0], xa[1] - xa[0], ya[1] - ya[0], head_width=10, color='r')#i left camera direction
#    axes.arrow(xa[0], ya[0], xa[2] - xa[0], ya[2] - ya[0], head_width=10, color='g')#j up camera direction
#    axes.arrow(xa[0], ya[0], xa[3] - xa[0], ya[3] - ya[0], head_width=10, color='b')#k down camera lens     
#    plt.axis('off')
#    plt.tight_layout()
#    plt.show()    
    
    
      
    from keras import optimizers

    learning_rate=0.00015
    optimizer_option=6
    
    if optimizer_option==1:
        opt = optimizers.SGD(lr=learning_rate, decay=1e-6, momentum=0.9, nesterov=True)
    elif optimizer_option==2:
        opt = optimizers.RMSprop(lr=learning_rate, rho=0.9, epsilon=None, decay=0.0)
    elif optimizer_option==3:
        opt = optimizers.Adagrad(lr=learning_rate, epsilon=None, decay=0.0)
    elif optimizer_option==4:
        opt = optimizers.Adagrad(lr=learning_rate, epsilon=None, decay=0.0)
    elif optimizer_option==5:
        opt = optimizers.Adadelta(lr=learning_rate, rho=0.95, epsilon=None, decay=0.0)
    elif optimizer_option==6:
        opt = optimizers.Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    elif optimizer_option==7:
        opt = optimizers.Adamax(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0)    
    elif optimizer_option==8:
        opt = optimizers.Nadam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)
    else:
        opt = optimizers.Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
        
    def get_lr_metric(optimizer):
        def lr(y_true, y_pred):
            return optimizer.lr
        return lr
    
    lr_metric = get_lr_metric(opt)
    
    Restart=True
    Training=False
    Inference =not Training
    submit=False
    predict=True
    evaluatemodel=False
    
    if Restart:
    
        print('Loading Model...') 
        # load json and create model
        json_file = open(str(input_img_size[0])+'model_final.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        keras.backend.set_learning_phase(Training)#for training all
        model_final = model_from_json(loaded_model_json)
        
        #for layer in model_final.layers:
        #    print(layer, layer.trainable)
        # load weights into new model
        model_final.load_weights(str(input_img_size[0])+"_pose_bestmodel.hdf5")
        print("Loaded model from disk")
        
        #pop input layer
#        model_final.layers.pop(0)        
#        newInput = Input(batch_shape=(0,512,512,3))    # let us say this new InputLayer
#        newOutputs = model_final(newInput)
#        model_final = Model(newInput, newOutputs)

        with open(str(input_img_size[0])+'random_gen_state.obj', 'rb') as f:
            np.random.set_state(load(f))
    
    elif Restart==False:
        print('Creating Model...')     
        # Loading and freezing pre-trained model
        K.set_learning_phase(0)
        input_shape=(input_img_size[0], input_img_size[1], 3)
        #input_tensor = Input(shape =input_shape)
        pretrained_model = ResNet50(weights="imagenet", include_top=False,input_shape=input_shape)
        
        #batch_norm = BatchNormalization()(input_tensor)
        #x = pretrained_model(batch_norm)
        
        #pretrained_model
        # Adding new trainable hidden and output layers to the model
        K.set_learning_phase(1)
        x = pretrained_model.output
        x = keras.layers.Flatten()(x) #could get out of memory
        #x = keras.layers.GlobalAveragePooling2D(name='avg_pool')(x)
        #x = keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
        #x = keras.layers.Dropout(0.2)(x)
        x = keras.layers.Dense(1024, activation="relu")(x)
        #x = keras.layers.Dense(64, activation="relu")(x)
        
        #x = keras.layers.Dropout(0.4)(x)
        #x = keras.layers.Dense(256, activation="relu",kernel_regularizer=keras.regularizers.l2(0.01))(x)
        #x = keras.layers.Dropout(0.2)(x)
        #x = keras.layers.Dense(64, activation="relu",kernel_regularizer=keras.regularizers.l2(0.01))(x)#Fork Here
        #predictions = keras.layers.Dense(7, activation="linear")(x)
        #model_final = keras.models.Model(inputs=pretrained_model.input, outputs=predictions)
        #model_final.compile(loss="mean_squared_error", optimizer='adam',metrics=['accuracy',lr_metric])   
        
        
        #mysection
        #qvec = keras.layers.Dense(1024, activation="relu")(x)
        qvec=keras.layers.Dense(4, activation="linear",name='q')(x)
        #qvec = Lambda(lambda  x: K.l2_normalize(x,axis=1),name='q')(qvec)#,name="q_pred"
        
        rvec=keras.layers.Dense(3, activation="linear",name='r')(x)
        
        model_final = keras.models.Model(inputs=pretrained_model.input, outputs=[qvec,rvec])
        
        # i.e. freeze all pretrained_model.layers
        for layer in pretrained_model.layers:
            layer.trainable = False
            
        #for layer in pretrained_model.layers[-1]:#activation_49 
            #print(layer, layer.trainable) 
        pretrained_model.layers[-1].trainable = False   
            
#        for layer in model_final.layers:
#            print(layer, layer.trainable)            
        
        #from keras.utils import plot_model
        #plot_model(model_final, show_shapes=True, expand_nested=True,to_file='model_final.png')
        
        with open(str(input_img_size[0])+'random_gen_state.obj', 'wb') as f:
            dump(np.random.get_state(), f)
            
        # serialize model to JSON
        print("Saving model to disk")
        model_json = model_final.to_json()
        with open(str(input_img_size[0])+"model_final.json", "w") as json_file:
            json_file.write(model_json)    

    model_final.compile(loss=['mean_squared_error','mean_squared_error'],loss_weights=[1.0,1.0], optimizer=opt)#,metrics=['accuracy','cosine_proximity','mean_squared_error',lr_metric])
    
    ##Loss must also include positional error
    ##activation function should be relu for positional vector as it ranges from zero to infinity
    ##activation function should be tanh for quaternion vector
    model_final.summary()
#    def myprint(s):
#        with open('modelsummary.txt','w+') as f:
#            print(s, file=f)
#    
#    model_final.summary(print_fn=myprint)
#    
    from contextlib import redirect_stdout

    with open('modelsummary.txt', 'w') as f:
        with redirect_stdout(f):
            model_final.summary()

    #SVG(model_to_dot(model_final).create(prog='dot', format='svg'))
    #plot_model(model_final, to_file='model_plot.png', show_shapes=True, show_layer_names=True)#conda install -c conda-forge pydot 
    
#    lr_finder = LRFinder(model_final)
###    #lr_finder.find(x_train, y_train, start_lr=0.0001, end_lr=0.01, batch_size=batch_size, epochs=epochs)
#    lr_finder.find_generator(training_generator, start_lr=0.0001, end_lr=0.01, epochs=1, steps_per_epoch=None)
#    lr_finder.plot_loss(n_skip_beginning=5, n_skip_end=1)
#
####Main model training    
    if Training:
        BestModel = ModelCheckpoint(filepath=str(input_img_size[0])+'_pose_bestmodel.hdf5', monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=True, mode='min', period=1)
        csv_logger = CSVLogger(str(input_img_size[0])+'History_log.csv', append=True, separator=',')
        early_stopping = EarlyStopping(patience=15, verbose=1)
        #reduceLROnPlato = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, verbose=1, mode='min')
        tboard_epoch=TensorBoard(log_dir='./logs', 
                       histogram_freq=0, 
                       batch_size=32, 
                       write_graph=True, 
                       write_grads=False, 
                       write_images=False, 
                       embeddings_freq=0, 
                       embeddings_layer_names=['lambda_1','dense_5'], 
                       embeddings_metadata=None, 
                       embeddings_data=None, 
                       update_freq='epoch')   
        
    #    tboard_batch=TensorBoard(log_dir='./logsbatch', 
    #                   histogram_freq=0, 
    #                   batch_size=50, 
    #                   write_graph=True, 
    #                   write_grads=False, 
    #                   write_images=False, 
    #                   embeddings_freq=0, 
    #                   embeddings_layer_names=['lambda_1','dense_5'], 
    #                   embeddings_metadata=None, 
    #                   embeddings_data=None, 
    #                   update_freq='batch')        
        
        clr = CyclicLR(base_lr=0.0005, 
                       max_lr=0.001,
                       step_size=960,#1050, #change to 1050
                       mode='triangular2',
                       gamma=1.,
                       scale_fn=None,
                       scale_mode='cycle',
                       monitor='val_loss',### 
                       factor=0.5, 
                       patience=10,
                       verbose=1, 
                       reduction_mode='min', 
                       min_delta=1e-4, 
                       cooldown=4, 
                       min_lr=1e-6) 
    
        # Training the model (transfer learning)
        history = model_final.fit_generator(
                                training_generator,
                                epochs=epochs,
                                validation_data=validation_generator,
                                workers=4,
                                max_queue_size=40,
                                callbacks=[BestModel,csv_logger,clr,early_stopping,tboard_epoch])
        
        with open(str(input_img_size[0])+'random_gen_state.obj', 'wb') as f:
            dump(np.random.get_state(), f)    
    
        print('Training losses: ', history.history['loss'])
        print('Validation losses: ', history.history['val_loss'])
        
        model_final.load_weights(filepath=str(input_img_size[0])+'_pose_bestmodel.hdf5')

    if evaluatemodel:
        evaluations = model_final.evaluate_generator(validation_generator, max_queue_size=40, workers=4, use_multiprocessing=False, verbose=1)
        print('Validation Losses: '+ str(evaluations))
####Predict on Validation Data  
    if predict:
        predictions = model_final.predict_generator(generator=validation_generator,
                                                    workers=4,
                                                    max_queue_size=40,
                                                    verbose=1)
        fileObject = open('predictions.pkl', 'wb')
        dump(predictions,fileObject)
        
        PlotinPLotly.send_to_plotly_results(train_labels,validation_labels,predictions)

#####Generating submission
    if submit:
        submission = SubmissionWriter()
        evaluate(model_final, 'test', submission.append_test, speed_root)
        evaluate(model_final, 'real_test', submission.append_real_test, speed_root)
        submission.export(suffix=str(input_img_size[0])+'results')      
    

    return model_final

seed(seed_int)
np.random.seed(seed=seed_int)
dataset=r'C:/DeepLearningFolder/Pose Estimation/speed/'

#with open(os.path.join(dataset, 'train' + '.json'), 'r') as f:
#    label_list = json.load(f)
#
#q=[]
#r=[]
#xa=[]
#ya=[]
#za=[]
#for image_ann in label_list:
#    qi=image_ann['q_vbs2tango']
#    ri=image_ann['r_Vo2To_vbs_true']
#    q.append(qi)
#    r.append(ri)
#    xai, yai = project(Camera.K,qi, ri)
#    xa.append(xai[0])
#    ya.append(yai[0])
#    za.append(ri[2])
#    
#import pandas as pd    
#data=pd.DataFrame(
#    {'x_pixel': xa,
#     'y_pixel': ya,
#     'z_m': za
#    })  
#print('X: Min='+str(np.min(xa))+' Max='+str(np.max(xa)))                
#print('Y: Min='+str(np.min(ya))+' Max='+str(np.max(ya)))
#
#minval=np.min([np.min(xa),np.min(ya)])
#maxval=np.max([np.max(xa),np.max(ya)])
#
#print(str(minval)+' '+str(maxval))
#                
#data.sort_values(by=['z_m'],inplace=True)

model_final=main(dataset, int(20), int(35))

#if __name__ == "__main__":
#    import argparse
#    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
#    parser.add_argument('--dataset', help='Path to the downloaded speed dataset.', default='')
#    parser.add_argument('--epochs', help='Number of epochs for training.', default=20)
#    parser.add_argument('--batch', help='number of samples in a batch.', default=32)
#    args = parser.parse_args()
#
#    main(args.dataset, int(args.epochs), int(args.batch))
#    history=0
#    model_final=0