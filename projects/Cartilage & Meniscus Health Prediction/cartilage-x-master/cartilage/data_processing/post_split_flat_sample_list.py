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

random.seed(100)

data_root = '/data/bigbone4/ciriondo/'

sample_list_root = os.path.join(data_root, 'sample_lists')
flat_sample_list_csv = os.path.join(sample_list_root, 'flat_full_sample_list_1.csv')
df = pd.read_csv(flat_sample_list_csv)

# drop any remaining null rows
df = df.dropna()

# setting validation size
valid_size = 0.20

# classify patients into those who have had WORMS > 1 (ever) and those who havent
df_sum = df.groupby(by=['patient'])['worms'].agg(['sum', 'count']) # removed 'type'
df_sum['label'] = (df_sum['sum'] > 1.0)*1 + 0*(df_sum['sum'] <= 0)
df_sum = df_sum.reset_index()[['patient', 'count', 'label']]  # removed 'type'

# indexing by labels and sample count per patient
df_sum = df_sum.sort_values(['label', 'count']).set_index(['label', 'count'])
df_sum['dataset'] = 0

# loop through each partition of (label, sample count per patient) 
# and assign train/valid labels in 60/20
for label, df_part in df_sum.groupby([df_sum.index.get_level_values(0), df_sum.index.get_level_values(1)]):
    # get dataset sizes for partition
    m_part = df_part.shape[0]
    m_valid = int(m_part * valid_size)    # validation: 1
    m_train = m_part - m_valid            # train: 0
    
    # assigning numbers 0, 1, 2 to indicate the sample will go to each set
    idx = np.array([0]*m_train + [1]*m_valid)
    np.random.shuffle(idx)
    
    df_sum.loc[label, 'dataset'] = idx
    
# join dataset labels onto main table
df_sum = df_sum.reset_index()
df = shuffle(pd.merge(df, df_sum, how='inner', on=['patient'])).drop(['count', 'label'], axis=1).reset_index(drop=True)
        
# create dataframes with the locations and labels of all image for each dataset
df_train = df[df['dataset'] == 0].drop('dataset', axis=1)
df_valid = df[df['dataset'] == 1].drop('dataset', axis=1)

print('\nTrain data size: ', df_train.shape[0])
print('Validation data size: ', df_valid.shape[0])

pos_train = (df_train['worms'] >= 2).mean()
pos_valid = (df_valid['worms'] >= 2).mean()

print('\nTraining positives: {:.4f}'.format(pos_train))
print('Vaild positives: {:.4f}'.format(pos_valid))

df_train.to_csv(os.path.join(sample_list_root, 'flat_train_sample_list_1.csv'), index=False)
df_valid.to_csv(os.path.join(sample_list_root, 'flat_valid_sample_list_1.csv'), index=False)


