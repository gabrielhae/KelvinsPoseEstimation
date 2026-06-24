import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import json
import keras
import tensorflow as tf
from tensorflow import set_random_seed
from keras import backend as K
keras.backend.clear_session() 

import random
from imgaug import augmenters as iaa
from PIL import Image
from pickle import dump,load
#from utils import KerasDataGenerator

from keras.applications.resnet50 import ResNet50,preprocess_input
from keras.models import Model,model_from_json
from keras.preprocessing import image
from keras.utils import Sequence
from keras.preprocessing import image as keras_image
from keras.layers import Lambda,Input
from keras.callbacks import Callback,CSVLogger,History,ModelCheckpoint,ProgbarLogger,ReduceLROnPlateau,TensorBoard,EarlyStopping

from main_lr_finder import LRFinder
from cyclical_learning_rate_withRLRoP import CyclicLR

#from tensorflow.keras.applications.resnet50 import preprocess_input
#from tensorflow.keras.preprocessing import image
import numpy as np
from submission import SubmissionWriter

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


""" 
    Example script demonstrating training on the SPEED dataset using Keras.
    Usage example: python keras_example.py --dataset [path to speed] --epochs [num epochs] --batch [batch size]
"""


def evaluate(model, dataset, append_submission, dataset_root):

    """ Running evaluation on test set, appending results to a submission. """

    with open(os.path.join(dataset_root, dataset + '.json'), 'r') as f:
        image_list = json.load(f)

    print('Running evaluation on {} set...'.format(dataset))

    for img in tqdm(image_list):
        img_path = os.path.join(dataset_root,resolution, dataset, img['filename'])
        if selection==1:
            pil_img = image.load_img(img_path, target_size=(512, 512))
        elif selection==2:
            pil_img = image.load_img(img_path, target_size=(512, 512))
        else:
            pil_img = image.load_img(img_path, target_size=(512, 512))
            
        x = image.img_to_array(pil_img)
        x = preprocess_input(x)
        x = np.expand_dims(x, 0)
        output = model.predict(x)
        test=np.concatenate((output[0],output[1]),axis=None)
        append_submission(img['filename'], test[0:4], test[4:7])
        #append_submission(img['filename'], output[0][0:4], output[1][0:3])
        
class KerasDataGenerator(Sequence):

        """ DataGenerator for Keras to be used with fit_generator (https://keras.io/models/sequential/#fit_generator)"""

        def __init__(self, preprocessor, label_list, speed_root, batch_size=32, dim=(224, 224), n_channels=3, shuffle=False,augment=False):

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

        def __len__(self):

            """ Denotes the number of batches per epoch. """

            return int(np.floor(len(self.list_IDs) / self.batch_size))

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
            X = np.empty((self.batch_size, *self.dim, self.n_channels))
            y1 = np.empty((self.batch_size, 4), dtype=float)
            y2 = np.empty((self.batch_size, 3), dtype=float)#change to 4

            # Generate data
            for i, ID in enumerate(list_IDs_temp):

                q, r = self.labels[ID]['q'], self.labels[ID]['r']
                
                img_path = os.path.join(self.image_root, ID)
                img = keras_image.load_img(img_path, target_size=(224, 224))
                if self.augment:
                    q,r,img = self._augment(q,r,img)
                else:
                    pass
                
                x = keras_image.img_to_array(img)
                x = self.preprocessor(x)
                X[i,] = x 
                y1[i],y2[i] = q,r#np.concatenate([q, r])

            return X, [y1, y2] 

        def _augment(self, q,r,image):
            
            #Random translation
            
            
            #random from list 0,90,180,270
            angle=random.choice([0,90,180,270])
            if angle==0:
                augment_img=iaa.Affine(rotate=0)
            elif angle==90:
                augment_img=iaa.Affine(rotate=90)
            elif angle==180:
                augment_img=iaa.Affine(rotate=180)
            elif angle==270:
                augment_img=iaa.Affine(rotate=270)    
            else:
                pass
            
            np_im = np.array(image)
            image_aug = augment_img.augment_image(np_im)
            
            q = self.q_rotation(q,angle)
            r = self.z_rotation(r,angle)
            
            new_im = Image.fromarray(image_aug)
            return q,r,new_im
        
        def z_rotation(self, vector,angle):
            """Rotates 3-D vector around z-axis"""
            theta=np.radians(-angle)
            R = np.array([[np.cos(theta), -np.sin(theta),0],[np.sin(theta), np.cos(theta),0],[0,0,1]])
            return np.dot(R,vector)    
        
        def q_rotation(self, quat,angle):
            """Rotates quaternion around the z-axis"""
            quaternion = R.from_quat(quat)
            q90r = R.from_euler('x',angle, degrees=True)
            quaternion*=q90r
            qrotated=quaternion.as_quat()
            return qrotated        
   
    #Geodesic Loss
def geodesic_loss(y_pred, y_true):
    y_pred=K.l2_normalize(y_pred,axis = 0)
    tmp=K.sum(y_true*y_pred,axis=-1,keepdims=True)
    theta=2.0*tf.math.acos(tf.clip_by_value(tmp,-1.0+1e-12,1.0-1e-12))
    return theta

def euclidean_distance_loss(y_true, y_pred):
    """
    Euclidean distance loss
    https://en.wikipedia.org/wiki/Euclidean_distance
    :param y_true: TensorFlow/Theano tensor
    :param y_pred: TensorFlow/Theano tensor of the same shape as y_true
    :return: float
    """
    return K.sqrt(K.sum(K.square(y_pred - y_true), axis=-1))

def main(speed_root, epochs, batch_size):

    """ Setting up data generators and model, training, and evaluating model on test and real_test sets. """

    # Setting up parameters
    params = {'dim': (224, 224),
              'batch_size': batch_size,
              'n_channels': 3,
              'shuffle': True}

    # Loading and splitting dataset
    with open(os.path.join(speed_root, 'train' + '.json'), 'r') as f:
        label_list = json.load(f)
    train_labels = label_list[:int(len(label_list)*.05)]
    validation_labels = label_list[int(len(label_list)*.05):]

    # Data generators for training and validation
    training_generator = KerasDataGenerator(preprocess_input, train_labels, speed_root, **params)
    validation_generator = KerasDataGenerator(preprocess_input, validation_labels, speed_root, **params)
    
    trickval_generator = KerasDataGenerator(preprocess_input, train_labels, speed_root, shuffle=True)

    
    
  
    
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
    
    Restart=False
    
    if Restart:
    
        print('Loading Model...') 
        # load json and create model
        json_file = open('model_final.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        keras.backend.set_learning_phase(1)#for training all
        model_final = model_from_json(loaded_model_json)
        
        #for layer in model_final.layers:
        #    print(layer, layer.trainable)
        # load weights into new model
        model_final.load_weights("pose_bestmodel.hdf5")
        print("Loaded model from disk")
        
        #pop input layer
#        model_final.layers.pop(0)        
#        newInput = Input(batch_shape=(0,512,512,3))    # let us say this new InputLayer
#        newOutputs = model_final(newInput)
#        model_final = Model(newInput, newOutputs)

        with open('random_gen_state.obj', 'rb') as f:
            np.random.set_state(load(f))
    
    elif Restart==False:
        print('Creating Model...')     
        # Loading and freezing pre-trained model
        keras.backend.set_learning_phase(0)
        input_shape=(224, 224, 3)
        #input_tensor = Input(shape =input_shape)
        pretrained_model = ResNet50(weights="imagenet", include_top=False,input_shape=input_shape)
        
        #batch_norm = BatchNormalization()(input_tensor)
        #x = pretrained_model(batch_norm)
        
        #pretrained_model
        # Adding new trainable hidden and output layers to the model
        keras.backend.set_learning_phase(1)
        x = pretrained_model.output
        x = keras.layers.Flatten()(x)
        #x = keras.layers.GlobalAveragePooling2D(name='avg_pool')(x)
        x = keras.layers.Dense(1024, activation="relu")(x)#Fork Here
        #predictions = keras.layers.Dense(7, activation="linear")(x)
        #model_final = keras.models.Model(inputs=pretrained_model.input, outputs=predictions)
        #model_final.compile(loss="mean_squared_error", optimizer='adam',metrics=['accuracy',lr_metric])   
        
        
    #    #mysection
        qvec=keras.layers.Dense(4, activation="tanh")(x)
        qvec = Lambda(lambda  x: K.l2_normalize(x,axis=1))(qvec)
        
        rvec=keras.layers.Dense(3, activation="linear")(x)
        #predictions = [qvec,rvec]
    #    
        model_final = keras.models.Model(inputs=pretrained_model.input, outputs=[qvec,rvec])
        
        #from keras.utils import plot_model
        #plot_model(model_final, show_shapes=True, expand_nested=True,to_file='model_final.png')
        
        with open('random_gen_state.obj', 'wb') as f:
            dump(np.random.get_state(), f)
            
        # serialize model to JSON
        print("Saving model to disk")
        model_json = model_final.to_json()
        with open("model_final.json", "w") as json_file:
            json_file.write(model_json)    

    model_final.compile(loss=[geodesic_loss,euclidean_distance_loss],loss_weights=[1.0,1.0], optimizer=opt,metrics=['accuracy',lr_metric])
    
    ##Loss must also include positional error
    ##activation function should be relu for positional vector as it ranges from zero to infinity
    ##activation function should be tanh for quaternion vector
    model_final.summary()
    
    
#    lr_finder = LRFinder(model_final)
###    #lr_finder.find(x_train, y_train, start_lr=0.0001, end_lr=0.01, batch_size=batch_size, epochs=epochs)
#    lr_finder.find_generator(training_generator, start_lr=0.0001, end_lr=0.002, epochs=1, steps_per_epoch=None)
#    lr_finder.plot_loss(n_skip_beginning=5, n_skip_end=1)
    
    BestModel = ModelCheckpoint(filepath='testpose_bestmodel.hdf5', monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=True, mode='min', period=1)
    csv_logger = CSVLogger('History_log.csv', append=True, separator=',')
    early_stopping = EarlyStopping(patience=4, verbose=1)
    #reduceLROnPlato = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, verbose=1, mode='min')
    clr = CyclicLR(base_lr=0.0001, 
                   max_lr=0.001,
                   step_size=1200, 
                   mode='triangular2',
                   gamma=1.,
                   scale_fn=None,
                   scale_mode='cycle',
                   monitor='val_loss',### 
                   factor=0.5, 
                   patience=2,
                   verbose=1, 
                   reduction_mode='min', 
                   min_delta=1e-4, 
                   cooldown=1, 
                   min_lr=1e-6) 

    # Training the model (transfer learning)
    history = model_final.fit_generator(
        training_generator,
        epochs=epochs,
        validation_data=trickval_generator,
        callbacks=[BestModel,early_stopping])#BestModel,csv_logger,clr,early_stopping])
    
    with open('random_gen_state.obj', 'wb') as f:
        dump(np.random.get_state(), f)    

    print('Training losses: ', history.history['loss'])
    print('Validation losses: ', history.history['val_loss'])
    
#    model_final.load_weights(filepath='pose_bestmodel.hdf5')
#
#    # Generating submission
#    submission = SubmissionWriter()
#    evaluate(model_final, 'test', submission.append_test, speed_root)
#    evaluate(model_final, 'real_test', submission.append_real_test, speed_root)
#    submission.export(suffix='keras_example')


#if __name__ == "__main__":
#    import argparse
#    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
#    parser.add_argument('--dataset', help='Path to the downloaded speed dataset.', default='')
#    parser.add_argument('--epochs', help='Number of epochs for training.', default=20)
#    parser.add_argument('--batch', help='number of samples in a batch.', default=32)
#    args = parser.parse_args()
#
#    main(args.dataset, int(args.epochs), int(args.batch))
    return history,model_final
dataset=r'C:/DeepLearningFolder/Pose Estimation/speed/'
history,model_final=main(dataset, int(50), int(16))

