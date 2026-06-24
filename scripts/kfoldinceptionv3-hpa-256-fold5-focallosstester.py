import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


import numpy as np
import pandas as pd

from datetime import datetime
import time
#import seaborn as sns
import matplotlib.pyplot as plt
from pickle import dump,load
from sklearn import metrics
#import skimage.io
#from skimage.transform import resize
from imgaug import augmenters as iaa
import cv2
from PIL import Image

import keras
#from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Model,model_from_json
from keras.layers import Activation, ZeroPadding2D, Convolution2D, GlobalAveragePooling2D, Dropout, Flatten, Dense,Input, Conv2D, MaxPooling2D, BatchNormalization, Concatenate, ReLU, LeakyReLU
from keras.regularizers import l2

from keras.applications.inception_v3 import InceptionV3
from keras.callbacks import Callback,CSVLogger,History,ModelCheckpoint,ProgbarLogger,ReduceLROnPlateau,TensorBoard,EarlyStopping
#from keras import metrics
#from keras.optimizers import Adam 
from keras.initializers import glorot_normal, RandomNormal, Zeros
from tqdm import tqdm

PATH = '../input/human-protein-atlas-image-classification'
TRAIN = '../input/human-protein-atlas-image-classification/train/'
TEST = '../input/human-protein-atlas-image-classification/test/'
LABELS = '../input/human-protein-atlas-image-classification/train.csv'
HPA_LABELS='../input/hpaclasslabels/hpa_specific_class_labels.csv'
HPATRAIN = '../input/hpa-rareclassimages512/grayscale_hpa_data_512/'

SAMPLE = '../input/human-protein-atlas-image-classification/sample_submission.csv'

Restart_Dir= '../input/modelforrestart/'

label_names = {
    0:  "Nucleoplasm",  
    1:  "Nuclear membrane",   
    2:  "Nucleoli",   
    3:  "Nucleoli fibrillar center",   
    4:  "Nuclear speckles",
    5:  "Nuclear bodies",   
    6:  "Endoplasmic reticulum",   
    7:  "Golgi apparatus",   
    8:  "Peroxisomes",   
    9:  "Endosomes",   
    10:  "Lysosomes",   
    11:  "Intermediate filaments",   
    12:  "Actin filaments",   
    13:  "Focal adhesion sites",   
    14:  "Microtubules",   
    15:  "Microtubule ends",   
    16:  "Cytokinetic bridge",   
    17:  "Mitotic spindle",   
    18:  "Microtubule organizing center",   
    19:  "Centrosome",   
    20:  "Lipid droplets",   
    21:  "Plasma membrane",   
    22:  "Cell junctions",   
    23:  "Mitochondria",   
    24:  "Aggresome",   
    25:  "Cytosol",   
    26:  "Cytoplasmic bodies",   
    27:  "Rods & rings"
}

#Functions
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

def getTrainDataset():
    
    path_to_train = TRAIN
    data = pd.read_csv(LABELS)
    #HPA Rare Class Insertion
    HPA_labels = pd.read_csv(HPA_LABELS)

    paths = []
    labels = []
    
    for name, lbl in zip(data['Id'], data['Target'].str.split(' ')):
        #print (lbl)
        y = np.zeros(28)
        for key in lbl:
            y[int(key)] = 1
        paths.append(os.path.join(TRAIN, name))
        labels.append(y)
        
    for name, lbl in zip(HPA_labels['Id'], HPA_labels['Target'].str.split(' ')):
        #print (lbl)
        y = np.zeros(28)
        for key in lbl:
            #print(key)
            y[int(key)] = 1
        paths.append(os.path.join(HPATRAIN, name))
        labels.append(y)

    return np.array(paths), np.array(labels)

def getTestDataset():
    
    path_to_test = TEST
    data = pd.read_csv(SAMPLE)

    paths = []
    labels = []
    
    for name in data['Id']:
        y = np.ones(28)
        paths.append(os.path.join(path_to_test, name))
        labels.append(y)

    return np.array(paths), np.array(labels)

class ProteinDataGenerator(keras.utils.Sequence):
    #Generates data for keras
    def __init__(self, paths, labels, batch_size, shape, n_classes, shuffle = False, augment=False ,use_cache = False):
        self.paths, self.labels = paths,labels
        self.batch_size = batch_size
        self.shape = shape
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.augment = augment
        self.use_cache = use_cache
        if use_cache == True:
            self.cache = np.zeros((paths.shape[0], shape[0], shape[1], shape[2]))
            self.is_cached = np.zeros((paths.shape[0]))
        self.on_epoch_end()
    
    def __len__(self):#Checked OK
        #number of batches per epoch
        return int(np.ceil(len(self.paths) / float(self.batch_size)))
      
    def __getitem__(self, idx):
        #generate one batch of data
        
        start_index=idx*self.batch_size
        end_index=(idx+1)*self.batch_size
        indexes = self.indexes[start_index:end_index]
        paths = self.paths[indexes]
        
        
        X = np.zeros((paths.shape[0], self.shape[0], self.shape[1], self.shape[2]))
        #y = np.empty((self.batch_size), dtype=int)
        # Generate data
        if self.use_cache == True:
            X = self.cache[indexes]
            for i, ID in enumerate(paths[np.where(self.is_cached[indexes] == 0)]):
                image = self.__load_image(ID)
                self.is_cached[indexes[i]] = 1
                self.cache[indexes[i]] = image
                #augment entry
                if self.augment:
                    image = self._augment(image)
                X[i] = image
        else:
            for i, ID in enumerate(paths):
                #augment entry
                image = self.__load_image(ID)
                if self.augment:
                    image = self._augment(image)
                X[i] = image
        
        y = self.labels[indexes]
                
        return X,y

    
    def on_epoch_end(self):#OK
        
        # Updates indexes after each epoch
        self.indexes = np.arange(len(self.paths))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __iter__(self):
        """Create a generator that iterate over the Sequence."""
        for item in (self[i] for i in range(len(self))):
            yield item
            
    def __load_image(self, path):
#        R = Image.open(path + '_red.png')
#        G = Image.open(path + '_green.png')
#        B = Image.open(path + '_blue.png')
#        Y = Image.open(path + '_yellow.png')
#
#        im = np.stack((
#            np.array(R), 
#            np.array(G), 
#            np.array(B),
#            np.array(Y)), -1)
        
        R = Image.open(path + '_red.png')
        G = Image.open(path + '_green.png')
        B = Image.open(path + '_blue.png')
        
        tempR=np.zeros(shape=(512,512,3)).astype('uint8')
        tempG=np.zeros(shape=(512,512,3)).astype('uint8')
        tempB=np.zeros(shape=(512,512,3)).astype('uint8')

        tempR[:,:,0]=np.array(R)
        tempG[:,:,1]=np.array(G)
        tempB[:,:,2]=np.array(B)   
        
        tempR=np.mean(tempR, axis=2)
        tempG=np.mean(tempG, axis=2)
        tempB=np.mean(tempB, axis=2)
        
        tempR=tempR/np.max(tempR)
        tempG=tempG/np.max(tempG)
        tempB=tempB/np.max(tempB)       
            
        im = np.stack((
            tempR, 
            tempG, 
            tempB), -1)        
            
        im = cv2.resize(im, (self.shape[0], self.shape[1]))
        #im = np.divide(im, 255)#this is to normalize data
        return im
    
    def _augment(self, image):
        augment_img = iaa.Sequential([
            iaa.OneOf([
                iaa.Affine(rotate=0),
                iaa.Affine(rotate=90),
                iaa.Affine(rotate=180),
                iaa.Affine(rotate=270),
                iaa.Fliplr(0.5),
                iaa.Flipud(0.5),
            ])], random_order=True)
        
        image_aug = augment_img.augment_image(image)
        return image_aug

class TimedStopping(Callback):
    '''Stop training when enough time has passed.
    # Arguments
        seconds: maximum time before stopping.
        safety_factor: stop safety_factor * average_time_per_epoch earlier
        verbose: verbosity mode.
    '''
    def __init__(self, seconds=None, safety_factor=1, verbose=0):
        super(Callback, self).__init__()

        self.start_time = 0
        self.safety_factor = safety_factor
        self.seconds = seconds
        self.verbose = verbose
        self.time_logs = []

    def on_train_begin(self, logs={}):
        self.start_time = time.time()

    def on_epoch_end(self, epoch, logs={}):
        elapsed_time = time.time() - self.start_time
        self.time_logs.append(elapsed_time)

        avg_elapsed_time = float(sum(self.time_logs)) / \
            max(len(self.time_logs), 1)

        print(" ", self.seconds - self.safety_factor * avg_elapsed_time)
        if elapsed_time > self.seconds - self.safety_factor * avg_elapsed_time:
            self.model.stop_training = True
            if self.verbose:
                print('Stopping after %s seconds.' % self.seconds)

import tensorflow as tf
from tensorflow import set_random_seed

from keras import backend as K

keras.backend.clear_session() 
####################################
## TensorFlow wizardry
#config = tf.ConfigProto()
# 
## Don't pre-allocate memory; allocate as-needed
#config.gpu_options.allow_growth = True
# 
## Only allow a total of half the GPU memory to be allocated
#config.gpu_options.per_process_gpu_memory_fraction = 0.6
# 
## Create a session with the above options specified.
#K.tensorflow_backend.set_session(tf.Session(config=config))
####################################

#input params to generator
batch_size = 16
dim = (299, 299, 3) # Xpixels, Ypixels, Channels
n_classes=28
SEED=444
VAL_RATIO = 0.10 # 10 % as validation
THRESHOLD = 0.2 # due to different cost of True Positive vs False Positive, this is the probability threshold to predict the class as 'yes'
set_random_seed(SEED)


def f1(y_true, y_pred):
    #y_pred = K.round(y_pred)
    y_pred = K.cast(K.greater(K.clip(y_pred, 0, 1), THRESHOLD), K.floatx())
    tp = K.sum(K.cast(y_true*y_pred, 'float'), axis=0)
    tn = K.sum(K.cast((1-y_true)*(1-y_pred), 'float'), axis=0)
    fp = K.sum(K.cast((1-y_true)*y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true*(1-y_pred), 'float'), axis=0)

    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())

    f1 = 2*p*r / (p+r+K.epsilon())
    f1 = tf.where(tf.is_nan(f1), tf.zeros_like(f1), f1)
    return K.mean(f1)

def f1_loss(y_true, y_pred):
    
    #y_pred = K.cast(K.greater(K.clip(y_pred, 0, 1), THRESHOLD), K.floatx())
    tp = K.sum(K.cast(y_true*y_pred, 'float'), axis=0)
    tn = K.sum(K.cast((1-y_true)*(1-y_pred), 'float'), axis=0)
    fp = K.sum(K.cast((1-y_true)*y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true*(1-y_pred), 'float'), axis=0)

    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())

    f1 = 2*p*r / (p+r+K.epsilon())
    f1 = tf.where(tf.is_nan(f1), tf.zeros_like(f1), f1)
    return 1-K.mean(f1)
    
def focal_loss(gamma=2., alpha=.25):
    def focalloss_fixed(y_true, y_pred):
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))

        pt_1 = K.clip(pt_1, 1e-3, .999)
        pt_0 = K.clip(pt_0, 1e-3, .999)

        return -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1))-K.sum((1-alpha) * K.pow( pt_0, gamma) * K.log(1. - pt_0))
    return focalloss_fixed
    
def create_model(input_shape, n_out):
    input_tensor = Input(shape=input_shape)
    base_model = InceptionV3(include_top=False,
                   weights='imagenet',
                   input_shape=input_shape)
    #bn = BatchNormalization()(input_tensor)
    #x = base_model(bn)
    #x = GlobalAveragePooling2D()(x)#added
    #x = Dropout(0.5)(x)
    #output = Dense(n_out, activation='sigmoid')(x)
    #model = Model(input_tensor, output)
    bn = BatchNormalization()(input_tensor)
    x = base_model(bn)
    x = Conv2D(32, kernel_size=(1,1), activation='relu')(x)
    x = Flatten()(x)
    x = Dropout(0.5)(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.5)(x)
    output = Dense(n_out, activation='sigmoid')(x)
    model = Model(input_tensor, output)    
    
    
    return model  

Restart=False

if Restart:

    print('Loading Model...') 
    # load json and create model
    json_file = open(Restart_Dir+'model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    # load weights into new model
    model.load_weights(Restart_Dir+"model.h5")
    print("Loaded model from disk")
    
    with open(Restart_Dir+'random_gen_state.obj', 'rb') as f:
        np.random.set_state(load(f))

elif Restart==False:
    print('Creating Model...')     
    model = create_model(
        input_shape=dim, 
        n_out=28)
    
    with open('random_gen_state.obj', 'wb') as f:
        dump(np.random.get_state(), f)
        
    # serialize model to JSON
    print("Saving model to disk")
    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)
 
from keras import optimizers

learning_rate=0.0001
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


class PredictGenerator(keras.utils.Sequence):
    #Generates data for keras
    def __init__(self, paths, batch_size, shape, use_cache = False):
        self.paths = paths
        self.indexes = np.arange(len(self.paths))
        self.batch_size = batch_size
        self.shape = shape
        self.use_cache = use_cache
        if use_cache == True:
            self.cache = np.zeros((paths.shape[0], shape[0], shape[1], shape[2]))
            self.is_cached = np.zeros((paths.shape[0]))
    
    def __len__(self):#Checked OK
        #number of batches per epoch
        return int(np.ceil(len(self.paths) / float(self.batch_size)))
      
    def __getitem__(self, idx):
        #generate one batch of data
        
        start_index=idx*self.batch_size
        end_index=(idx+1)*self.batch_size
        indexes = self.indexes[start_index:end_index]
        paths = self.paths[indexes]
        
        
        X = np.zeros((paths.shape[0], self.shape[0], self.shape[1], self.shape[2]))
        if self.use_cache == True:
            X = self.cache[indexes]
            for i, ID in enumerate(paths[np.where(self.is_cached[indexes] == 0)]):
                image = self.__load_image(ID)
                self.is_cached[indexes[i]] = 1
                self.cache[indexes[i]] = image
                X[i] = image
        else:
            for i, ID in enumerate(paths):
                image = self.__load_image(ID)
                X[i] = image
                
        return X

    def __iter__(self):
        """Create a generator that iterate over the Sequence."""
        for item in (self[i] for i in range(len(self))):
            yield item
            
    def __load_image(self, path):
        
        #R = Image.open(path + '_red.png')
        #G = Image.open(path + '_green.png')
        #B = Image.open(path + '_blue.png')
            
        #im = np.stack((
        #    np.array(R), 
        #    np.array(G), 
        #    np.array(B)), -1)       
            
        #im = cv2.resize(im, (self.shape[0], self.shape[1]))
        #im = np.divide(im, 255)#this is to normalize data
        
        R = Image.open(path + '_red.png')
        G = Image.open(path + '_green.png')
        B = Image.open(path + '_blue.png')
        
        tempR=np.zeros(shape=(512,512,3)).astype('uint8')
        tempG=np.zeros(shape=(512,512,3)).astype('uint8')
        tempB=np.zeros(shape=(512,512,3)).astype('uint8')

        tempR[:,:,0]=np.array(R)
        tempG[:,:,1]=np.array(G)
        tempB[:,:,2]=np.array(B)   
        
        tempR=np.mean(tempR, axis=2)
        tempG=np.mean(tempG, axis=2)
        tempB=np.mean(tempB, axis=2)
        
        tempR=tempR/np.max(tempR)
        tempG=tempG/np.max(tempG)
        tempB=tempB/np.max(tempB)       
            
        im = np.stack((
            tempR, 
            tempG, 
            tempB), -1)        
            
        im = cv2.resize(im, (self.shape[0], self.shape[1]))
        
        return im
    
def train_model(model, batch_size, epochs, dim, x, y, pathsTest, n_fold, kf,restart_fold):
    sk_f1 = metrics.f1_score
    preds_train = np.zeros((len(x),28), dtype = np.float)
    preds_test = np.zeros((len(pathsTest),28), dtype = np.float)
    train_scores = []; valid_scores = []

    i = 1

    for train_index, test_index in kf.split(x, y):
        print("TRAIN:", train_index, "TEST:", test_index)
        X_train, x_valid = x[train_index], x[test_index]
        y_train, y_valid = y[train_index], y[test_index]
        
        if restart_fold>i:
            i+=1
            print('Skipping this fold {}\n\n'.format(i))
            continue
        else:
            print('Beginning training for fold {}\n\n'.format(i))
        
        train_generator = ProteinDataGenerator(X_train, y_train, batch_size, dim, n_classes=28,use_cache=False, augment = True, shuffle = True)
        valid_generator = ProteinDataGenerator(x_valid, y_valid, batch_size, dim, n_classes=28,use_cache=False, shuffle = False)
        
        test_generator = PredictGenerator(pathsTest, batch_size, dim)

        csv_logger = CSVLogger('History_log.csv', append=True, separator=',')
        BestModel = ModelCheckpoint(filepath='inception.fold_' + str(i) + '.hdf5', monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=True, mode='min', period=1)
        reduceLROnPlato = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=1, cooldown=1, verbose=1, min_lr=1e-7)
        early_stop = EarlyStopping(monitor='val_loss', patience=3, verbose=1, min_delta=1e-4)
        callbacks = [csv_logger,BestModel,reduceLROnPlato,early_stop]
        
        train_steps = len(train_generator)
        valid_steps = len(valid_generator)
        test_steps = len(test_generator)
        
        model = model
        
        for layer in model.layers:
            layer.trainable = False
            
        model.layers[-1].trainable = True
        model.layers[-2].trainable = True
        model.layers[-3].trainable = True
        model.layers[-4].trainable = True
        model.layers[-5].trainable = True
        model.layers[-6].trainable = True        

        opt = optimizers.Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
        lr_metric = get_lr_metric(opt)
        model.compile(loss=[focal_loss(alpha=.25, gamma=2)],
                    optimizer=opt,
                    metrics=['acc',f1,lr_metric])
            

        model.fit_generator(
            train_generator,
            steps_per_epoch=train_steps,
            validation_data=valid_generator,
            validation_steps=valid_steps,
            epochs=1,
            verbose=1)
            
        for layer in model.layers:
            layer.trainable = True
        # we need to recompile the model for these modifications to take effect
        # we use SGD with a low learning rate
        opt = optimizers.Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
        lr_metric = get_lr_metric(opt)
        model.compile(loss=[focal_loss(alpha=.25, gamma=2)],
                    optimizer=opt,
                    metrics=['acc',f1,lr_metric])
            
        print('Training Bottom Layers of Model...')
        print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
         # train model
        csv_logger = CSVLogger('History_log_bottom.csv', append=True, separator=',') 
        history = model.fit_generator(
                train_generator,
                steps_per_epoch=train_steps,
                validation_data=valid_generator,
                validation_steps=valid_steps,
                epochs=epochs,
                verbose=1,
                callbacks=callbacks)
                
        print('Bottom Layer Training Finished...')        
        print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))


        model.load_weights(filepath='inception.fold_' + str(i) + '.hdf5')

        print('Running validation predictions on fold {}'.format(i))
        preds_valid = model.predict_generator(generator=valid_generator,
                                      steps=valid_steps, verbose=1)#valid_steps
        print('Running train predictions on fold {}'.format(i))
        preds_train = model.predict_generator(generator=train_generator,
                                      steps=train_steps, verbose=1)#train_steps
        #print(preds_train.shape)
        #print(y_train.shape)
        #valid_score = f1(y_valid, preds_valid)
        print(y_valid[:,0])
        print(preds_valid[:,0])
        rng = np.arange(0, 1, 0.001)
        f1s_valid = np.zeros((rng.shape[0], 28))
        f1s_train = np.zeros((rng.shape[0], 28))
        for j,t in enumerate(tqdm(rng)):
            for ilabel in range(28):
                p_valid = np.array(preds_valid[:,ilabel]>t, dtype=np.int8)
                p_train = np.array(preds_train[:,ilabel]>t, dtype=np.int8)
                scoref1_valid= sk_f1(y_valid[:,ilabel], p_valid, average='binary')
                scoref1_train= sk_f1(y_train[:,ilabel], p_train, average='binary')
                f1s_valid[j,ilabel] = scoref1_valid       
                f1s_train[j,ilabel] = scoref1_train  
        #print('max Validation f1'+np.max(f1s_valid, axis=0))
        #print('mean of max Validation f1'+np.mean(np.max(f1s_valid, axis=0)))
        #print('max Validation f1'+np.max(f1s_train, axis=0))
        #print('mean of max Validation f1'+np.mean(np.max(f1s_train, axis=0)))
        np.save('f1_valid_score_fold_' + str(i), f1s_valid)
        
        valid_score=np.mean(np.max(f1s_valid, axis=0))
        train_score=np.mean(np.max(f1s_train, axis=0))
        #print(valid_score)
        #train_score = f1(y_train, preds_train)      
        print('Val Score:{} for fold {}'.format(valid_score, i))
        print('Train Score: {} for fold {}'.format(train_score, i))

        valid_scores.append(valid_score)
        train_scores.append(train_score)
        print('Avg Train Score:{0:0.5f}, Val Score:{1:0.5f} after {2:0.5f} folds'.format
              (np.mean(train_scores), np.mean(valid_scores), i))
              
        #1 use https://www.kaggle.com/rejpalcz/cnn-128x128x4-keras-from-scratch-lb-0-328 to determine thresholdprobability vs f1 score
        #  will need to average over folds to determine optimal range for given model
        #  will then need to ensemble over models to determine optimal range for submission

        print('Running test predictions with fold {}'.format(i))

        preds_test_fold = model.predict_generator(generator=test_generator,
                                              steps=test_steps, verbose=1)#test_steps
        
        
        np.save('test_predictions_fold_' + str(i), preds_test_fold)
		
        preds_test += preds_test_fold

        print('\n\n')

        i += 1

        if i <= n_fold:
            print('Now beginning training for fold {}\n\n'.format(i))
        else:
            print('Finished training!')

    preds_test /= n_fold


    return preds_test
    
batch_size = 16
epochs = 20
n_fold = 5
restart_fold=5

import importlib.machinery
loader = importlib.machinery.SourceFileLoader('ml_stratifiers', '../input/moduleload/ml_stratifiers.py')
mod = loader.load_module()
from ml_stratifiers import MultilabelStratifiedShuffleSplit
msss = MultilabelStratifiedShuffleSplit(n_splits=n_fold, test_size=0.1, random_state=42)

print("loading test dataset paths...")
testpaths, testlabels = getTestDataset()

print("loading train dataset paths...")
trainpaths, trainlabels = getTrainDataset()

test_pred = train_model(model, batch_size, epochs, dim, trainpaths, 
                        trainlabels, testpaths, n_fold, msss,restart_fold)