import numpy as np
import json
import os
from PIL import Image
from matplotlib import pyplot as plt

from random import seed

#import pyquaternion
from rotation import Rotation as R

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

images='preprocessed_images_1200'#preprocessed_

angle=90

class Camera:

    """" Utility class for accessing camera parameters. """

    fx = 0.0176  # focal length[m]
    fy = 0.0176  # focal length[m]
    nu = 1200  # number of horizontal[pixels]
    nv = 1200  # number of vertical[pixels]
    ppx = 5.86e-6#*1920/224 # horizontal pixel pitch[m / pixel]
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


class SatellitePoseEstimationDataset:

    """ Class for dataset inspection: easily accessing single images, and corresponding ground truth pose data. """

    def __init__(self, root_dir='/datasets/speed_debug'):
        self.partitions, self.labels = process_json_dataset(root_dir)
        self.root_dir = root_dir

    def get_image(self, i=0, split='train'):

        """ Loading image as PIL image. """

        img_name = self.partitions[split][i]
        img_name = os.path.join(self.root_dir, images, split, img_name)
        image = Image.open(img_name).convert('RGB').rotate(angle)
        
        return image

    def get_pose(self, i=0):

        """ Getting pose label for image. """

        img_id = self.partitions['train'][i]
        q, r = self.labels[img_id]['q'], self.labels[img_id]['r']
        return q, r

    def visualize(self, i, partition='train', ax=None):

        """ Visualizing image, with ground truth pose with axes projected to training image. """

        if ax is None:
            ax = plt.gca()
        img = self.get_image(i)
        ax.imshow(img)

        # no pose label for test
        if partition == 'train':
            q, r = self.get_pose(i)
            #print(q)
            #quaternion=pyquaternion.Quaternion(np.array(q))
            #print(quaternion)
            #inv_quaternion = quaternion.inverse
            print(q)
            quaternion = R.from_quat(q)
            #inv_quaternion = quaternion.inv()
            #print(inv_quaternion.as_quat())
            #r3 = quaternion * inv_quaternion
            #q=r3.as_quat()
            
            q90r = R.from_euler('x',angle, degrees=True)
            quaternion*=q90r
            q=quaternion.as_quat()
            r = self.z_rotation(r,np.radians(-angle))
            xa, ya = project(q, r)
            ax.arrow(xa[0], ya[0], xa[1] - xa[0], ya[1] - ya[0], head_width=10, color='r')#i left camera direction
            ax.arrow(xa[0], ya[0], xa[2] - xa[0], ya[2] - ya[0], head_width=10, color='g')#j up camera direction
            ax.arrow(xa[0], ya[0], xa[3] - xa[0], ya[3] - ya[0], head_width=10, color='b')#k down camera lens

        return
    
    def z_rotation(self, vector,theta):
        """Rotates 3-D vector around z-axis"""
        R = np.array([[np.cos(theta), -np.sin(theta),0],[np.sin(theta), np.cos(theta),0],[0,0,1]])
        return np.dot(R,vector)


from matplotlib import pyplot as plt
from random import randint

dataset_root_dir = r'C:\DeepLearningFolder\Pose Estimation\speed'
dataset = SatellitePoseEstimationDataset(root_dir=dataset_root_dir)

#Augment quaternions for balancing
partitions,labels=process_json_dataset(dataset_root_dir)


#=======
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
#        
#        
#
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

#=======
rows = 2
cols = 2

seed(11)

fig, axes = plt.subplots(rows, cols, figsize=(12, 6))
for i in range(rows):
    for j in range(cols):
        img = dataset.get_image(randint(0, 12000))
        axes[i][j].imshow(img)
        axes[i][j].axis('off')
fig.tight_layout()
plt.show()

rows = 2
cols = 2

fig, axes = plt.subplots(rows, cols, figsize=(12, 12))
for i in range(rows):
    for j in range(cols):
        dataset.visualize(randint(0, 12000), ax=axes[i][j])
        axes[i][j].axis('off')
fig.tight_layout()
plt.show()
#=======


#fig, axes = plt.subplots(rows, cols, figsize=(12, 12))
#dataset.visualize(123)
#plt.axis('off')
#plt.tight_layout()
#plt.show()

#                iaa.Affine(rotate=0),
#                iaa.Affine(rotate=90),
#                iaa.Affine(rotate=180),
#                iaa.Affine(rotate=270),
#                iaa.Fliplr(0.5),
#                iaa.Flipud(0.5),
#if has_pytorch:
#    class PyTorchSatellitePoseEstimationDataset(Dataset):
#
#        """ SPEED dataset that can be used with DataLoader for PyTorch training. """
#
#        def __init__(self, split='train', speed_root='', transform=None):
#
#            if not has_pytorch:
#                raise ImportError('Pytorch was not imported successfully!')
#
#            if split not in {'train', 'test', 'real_test'}:
#                raise ValueError('Invalid split, has to be either \'train\', \'test\' or \'real_test\'')
#
#            with open(os.path.join(speed_root, split + '.json'), 'r') as f:
#                label_list = json.load(f)
#
#            self.sample_ids = [label['filename'] for label in label_list]
#            self.train = split == 'train'
#
#            if self.train:
#                self.labels = {label['filename']: {'q': label['q_vbs2tango'], 'r': label['r_Vo2To_vbs_true']}
#                               for label in label_list}
#            self.image_root = os.path.join(speed_root, images, split)
#
#            self.transform = transform
#
#        def __len__(self):
#            return len(self.sample_ids)
#
#        def __getitem__(self, idx):
#            sample_id = self.sample_ids[idx]
#            img_name = os.path.join(self.image_root, sample_id)
#
#            # note: despite grayscale images, we are converting to 3 channels here,
#            # since most pre-trained networks expect 3 channel input
#            pil_image = Image.open(img_name.replace('jpg', 'png')).convert('RGB')
#
#            if self.train:
#                q, r = self.labels[sample_id]['q'], self.labels[sample_id]['r']
#                y = np.concatenate([q, r])
#            else:
#                y = sample_id
#
#            if self.transform is not None:
#                torch_image = self.transform(pil_image)
#            else:
#                torch_image = pil_image
#
#            return torch_image, y
#else:
#    class PyTorchSatellitePoseEstimationDataset:
#        def __init__(self, *args, **kwargs):
#            raise ImportError('Pytorch is not available!')
#
#if has_tf:
#    class KerasDataGenerator(Sequence):
#
#        """ DataGenerator for Keras to be used with fit_generator (https://keras.io/models/sequential/#fit_generator)"""
#
#        def __init__(self, preprocessor, label_list, speed_root, batch_size=32, dim=(224, 224), n_channels=3, shuffle=True):
#
#            # loading dataset
#            self.image_root = os.path.join(speed_root, images, 'train')
#
#            # Initialization
#            self.preprocessor = preprocessor
#            self.dim = dim
#            self.batch_size = batch_size
#            self.labels = self.labels = {label['filename']: {'q': label['q_vbs2tango'], 'r': label['r_Vo2To_vbs_true']}
#                                         for label in label_list}
#            self.list_IDs = [label['filename'] for label in label_list]
#            self.n_channels = n_channels
#            self.shuffle = shuffle
#            self.indexes = None
#            self.on_epoch_end()
#
#        def __len__(self):
#
#            """ Denotes the number of batches per epoch. """
#
#            return int(np.floor(len(self.list_IDs) / self.batch_size))
#
#        def __getitem__(self, index):
#
#            """ Generate one batch of data """
#
#            # Generate indexes of the batch
#            indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
#
#            # Find list of IDs
#            list_IDs_temp = [self.list_IDs[k] for k in indexes]
#
#            # Generate data
#            X, y = self.__data_generation(list_IDs_temp)
#
#            return X, y
#
#        def on_epoch_end(self):
#
#            """ Updates indexes after each epoch """
#
#            self.indexes = np.arange(len(self.list_IDs))
#            if self.shuffle:
#                np.random.shuffle(self.indexes)
#
#        def __data_generation(self, list_IDs_temp):
#
#            """ Generates data containing batch_size samples """
#
#            # Initialization
#            X = np.empty((self.batch_size, *self.dim, self.n_channels))
#            y = np.empty((self.batch_size, 7), dtype=float)
#
#            # Generate data
#            for i, ID in enumerate(list_IDs_temp):
#                img_path = os.path.join(self.image_root, ID)
#                img = keras_image.load_img(img_path, target_size=(224, 224))
#                x = keras_image.img_to_array(img)
#                x = self.preprocessor(x)
#                X[i,] = x
#
#                q, r = self.labels[ID]['q'], self.labels[ID]['r']
#                y[i] = np.concatenate([q, r])
#
#            return X, y
#else:
#    class KerasDataGenerator:
#        def __init__(self, *args, **kwargs):
#            raise ImportError('tensorflow.keras is not available! Please install tensorflow.')
