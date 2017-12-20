# because I made a mistake

import os
import random
import time
import itertools

import cv2
import numpy as np
import pandas as pd
import scipy.io as spio

from sklearn.utils import shuffle
from six.moves import cPickle as pickle


def resize_volume(image, target_dims):
    """Crops image to bounding box then resizes to target volume"""
    # TODO: using numpy `apply_along_axis` to (potentially) speed up
    
    # interpolate along depth (z-axis)
    xy_resized = np.zeros(shape=[target_dims[0], target_dims[1], image.shape[2]],
                         dtype=np.float32)
    for i in range(image.shape[2]):
        xy_resized[:,:,i] = cv2.resize(image[:,:,i], 
                                       dsize=(target_dims[0], target_dims[1]), 
                                       interpolation=cv2.INTER_AREA)
    # interpolate along x-axis
    resized = np.zeros(shape=[target_dims[0], target_dims[1], target_dims[2]],
                       dtype=np.float32)
    for i in range(target_dims[0]):
        resized[i,:,:] = cv2.resize(xy_resized[i,:,:], 
                                       dsize=(target_dims[2], target_dims[1]), 
                                       interpolation=cv2.INTER_AREA)
    return resized


def _pickle_load(filename):
    """Load each image from the filename"""
    with open(filename, 'rb') as f:
        save = pickle.load(f)
        image = save['image'].astype(np.float32)
        label = np.float32(save['label'])
    return image, label


def relabel_pickled_images(df_samples):
    """Re-label pickle files from error in earlier script"""
          
    num_rows = df_samples.shape[0]
    
    for i, sample in enumerate(df_samples.iterrows()):
        
        worms = sample[1]['worms']
        filename = sample[1]['sample_path']
        
        with open(filename, 'rb') as f:
            data = pickle.load(f)
        
        assert data['worms'] == worms
        
#         try:
#             with open(filename, 'wb') as f:
#                 pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
#         except Exception as e:
#             print("Unable to save pickle file: ", filename, ": ", e)
#             raise
              
        
        print('\tImage {} of {} saved      '.\
                                           format(i, 
                                                  num_rows), end='\r')
    print('')

    
def normalize_pickled_images(df_samples):
    """Re-label pickle files from error in earlier script"""
          
    num_rows = df_samples.shape[0]
    
    for i, sample in enumerate(df_samples.iterrows()):
        
        filename = sample[1]['sample_path']
        
        with open(filename, 'rb') as f:
            data = pickle.load(f)
        
        image = data['image']
#         data['image'] = (image - image.min())/image.ptp()
        
        assert image.max() == 1.0
        assert image.min() == 0.0
    
#         try:
#             with open(filename, 'wb') as f:
#                 pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
#         except Exception as e:
#             print("Unable to save pickle file: ", filename, ": ", e)
#             raise
        
#         print('\tImage {} of {} saved      '.\
#                                            format(i,
#                                                   num_rows), end='\r')
    print('')   
    
    
def resize_pickled_images(df_samples, dims):
    """Re-label pickle files from error in earlier script"""
          
    num_rows = df_samples.shape[0]
    
    for i, sample in enumerate(df_samples.iterrows()):
        
        filename = sample[1]['sample_path']
        
        with open(filename, 'rb') as f:
            data = pickle.load(f)
        
        image = data['image']
#         data['image'] = (image - image.min())/image.ptp()
        data['image'] = resize_volume(image, dims)
        
        try:
            with open(filename, 'wb') as f:
                pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
        except Exception as e:
            print("Unable to save pickle file: ", filename, ": ", e)
            raise
        
        print('\tImage {} of {} saved      '.\
                                           format(i,
                                                  num_rows), end='\r')
    print('')   
    
    
# define root directory
data_root = '/data/bigbone4/ciriondo/'

# find all pickle files of interest
save_dir = os.path.join(data_root, 'pickle_data_combined')
full_sample_list_csv = os.path.join(data_root, 'sample_lists/full_sample_list_1.csv')
df_samples = pd.read_csv(full_sample_list_csv)  

resize_pickled_images(df_samples, [80, 80, 40])