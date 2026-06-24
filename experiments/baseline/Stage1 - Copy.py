# -*- coding: utf-8 -*-
"""
Stage 1: Pre-Process Images
"""
import os
import json
from random import shuffle,seed

#Variables
speed_root = 'C:\DeepLearningFolder\Pose Estimation\speed/'

with open(os.path.join(speed_root, 'test' + '.json'), 'r') as f:
    label_list = json.load(f)

size=300

seed(3)
shuffled_label_list=label_list.copy()  
shuffle(shuffled_label_list) 
train_labels_set = shuffled_label_list[:int(len(shuffled_label_list)*.8)]
validation_holdout_set = shuffled_label_list[int(len(shuffled_label_list)*.8):]

shuffle(train_labels_set)
train_labels = train_labels_set[:int(len(train_labels_set)*.8)]
validation_labels = train_labels_set[int(len(train_labels_set)*.8):]

#######    
#Proprocess images for low res trials
from PIL import Image
from tqdm import tqdm
    
dataset_root=speed_root
dataset='train'
with open(os.path.join(dataset_root, dataset + '.json'), 'r') as f:
    image_list = json.load(f)

print('Running evaluation on {} set...'.format(dataset))

for img in tqdm(image_list):
    img_path = os.path.join(dataset_root, 'images', dataset, img['filename'])
    processed_img_path = os.path.join(dataset_root, 'preprocessed_images_'+str(size), dataset, img['filename'])
    
    im = Image.open(img_path)
    
    width, height = im.size   # Get dimensions
    new_width=1200
    new_height=1200
    left = (width - new_width)/2
    top = (height - new_height)/2
    right = (width + new_width)/2
    bottom = (height + new_height)/2
    
    croppped=im.crop((left, top, right, bottom))
    converted=croppped.convert('RGB')
    resized=converted.resize((size,size),Image.LANCZOS)
    resized.save(processed_img_path)
    #croppped.save(processed_img_path)
#######        .replace('jpg', 'png'),format='png'