'''
    Inception V3 model for Keras, modified this code: 
    https://github.com/fchollet/deep-learning-models/blob/master/inception_v3.py
'''

from __future__ import print_function  
import os  
import numpy as np
import warnings
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential 
from keras.models import Model
from keras.layers import Flatten, Dense, Input, BatchNormalization, merge, Dropout
from keras.layers import Convolution2D, MaxPooling2D, AveragePooling2D
from keras.utils.layer_utils import convert_all_kernels_in_model
from keras.utils.data_utils import get_file
from keras import backend as K
from imagenet_utils import decode_predictions

# Inception V3 weights - get this file from the URL below 
# TF_WEIGHTS_PATH_NO_TOP = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.2/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'
TF_WEIGHTS_PATH_NO_TOP = '/Users/gerrypesavento/Documents/recycle_ai/datax/gh_train/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'

# Train images - these have to be organized in numbered folders
# Example for object 19, the images are in train/19/19a.jpg, train/19/19b.jpg, etc 
train_dir = '/Users/gerrypesavento/Documents/recycle_ai/datax_train_data/train_light'
nb_train_images = 0
nb_train_classes = 0 
for dir,subdir,files in os.walk(train_dir):
    nb_train_classes += len(subdir) # counts the number of object folders 
    for file in files:
        if file == '.DS_Store':
            badfile = dir + '/' + file
            print ('deleted .DS_Store file ', badfile)
            os.remove(badfile)
            nb_train_images -= 1
        elif ".jpg" not in file: 
            badfile = dir + '/' + file
            print ('deleted non-jpg file ', badfile)
            os.remove(badfile)
            nb_train_images -= 1
    nb_train_images += len(files)
print ('number of train classes ' + str(nb_train_classes))
print ('number of train images ' + str(nb_train_images))

# Validation images - these have to be organized in numbered folders
# Example for object 19, the images are in valid/19/19a.jpg, valid/19/19b.jpg, etc 
valid_dir = '/Users/gerrypesavento/Documents/recycle_ai/datax_train_data/valid_light'
nb_valid_images = 0
nb_valid_classes = 0 
for dir,subdir,files in os.walk(valid_dir):
    nb_valid_classes += len(subdir) # counts the number of object folders 
    for file in files:
        if file == '.DS_Store':
            badfile = dir + '/' + file
            print ('deleted .DS_Store file ', badfile)
            os.remove(badfile)
            nb_valid_images -= 1
        elif ".jpg" not in file: 
            badfile = dir + '/' + file
            print ('deleted non-jpg file ', badfile)
            os.remove(badfile)
            nb_valid_images -= 1
    nb_valid_images += len(files)
print ('number of valid classes ' + str(nb_valid_classes))
print ('number of valid images ' + str(nb_valid_images))

# default was 299; using 256x256 images 
# make sure all images are 256x256 px 
img_width, img_height = 256, 256

def conv2d_bn(x, nb_filter, nb_row, nb_col,
              border_mode='same', subsample=(1, 1),
              name=None):
    # Utility function to apply conv + BN

    if name is not None:
        bn_name = name + '_bn'
        conv_name = name + '_conv'
    else:
        bn_name = None
        conv_name = None
    
    bn_axis = 3  # for tf 

    x = Convolution2D(nb_filter, nb_row, nb_col,
                      subsample=subsample,
                      activation='relu',
                      border_mode=border_mode,
                      name=conv_name)(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name)(x)
    return x

def InceptionV3(include_top=True, weights='imagenet',
                input_tensor=None):
    '''
        Inception v3 architecture,
        returns a Keras model instance
    '''
    print ('running InceptionV3')

    if weights not in {'imagenet', None}:
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization) or `imagenet` '
                         '(pre-training on ImageNet).')
    
    # Determine proper input shape
    input_shape = (img_height, img_width, 3)  
    img_input = Input(shape=input_shape) # no tensor img_input
    channel_axis = 3

    x = conv2d_bn(img_input, 32, 3, 3, subsample=(2, 2), border_mode='valid')
    x = conv2d_bn(x, 32, 3, 3, border_mode='valid')
    x = conv2d_bn(x, 64, 3, 3)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = conv2d_bn(x, 80, 1, 1, border_mode='valid')
    x = conv2d_bn(x, 192, 3, 3, border_mode='valid')
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    # mixed 0, 1, 2: 35 x 35 x 256
    for i in range(3):
        branch1x1 = conv2d_bn(x, 64, 1, 1)

        branch5x5 = conv2d_bn(x, 48, 1, 1)
        branch5x5 = conv2d_bn(branch5x5, 64, 5, 5)

        branch3x3dbl = conv2d_bn(x, 64, 1, 1)
        branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)
        branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)

        branch_pool = AveragePooling2D(
            (3, 3), strides=(1, 1), border_mode='same')(x)
        branch_pool = conv2d_bn(branch_pool, 32, 1, 1)
        x = merge([branch1x1, branch5x5, branch3x3dbl, branch_pool],
                  mode='concat', concat_axis=channel_axis,
                  name='mixed' + str(i))

    # mixed 3: 17 x 17 x 768
    branch3x3 = conv2d_bn(x, 384, 3, 3, subsample=(2, 2), border_mode='valid')

    branch3x3dbl = conv2d_bn(x, 64, 1, 1)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3,
                             subsample=(2, 2), border_mode='valid')

    branch_pool = MaxPooling2D((3, 3), strides=(2, 2))(x)
    x = merge([branch3x3, branch3x3dbl, branch_pool],
              mode='concat', concat_axis=channel_axis,
              name='mixed3')

    # mixed 4: 17 x 17 x 768
    branch1x1 = conv2d_bn(x, 192, 1, 1)

    branch7x7 = conv2d_bn(x, 128, 1, 1)
    branch7x7 = conv2d_bn(branch7x7, 128, 1, 7)
    branch7x7 = conv2d_bn(branch7x7, 192, 7, 1)

    branch7x7dbl = conv2d_bn(x, 128, 1, 1)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 128, 7, 1)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 128, 1, 7)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 128, 7, 1)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 1, 7)

    branch_pool = AveragePooling2D((3, 3), strides=(1, 1), border_mode='same')(x)
    branch_pool = conv2d_bn(branch_pool, 192, 1, 1)
    x = merge([branch1x1, branch7x7, branch7x7dbl, branch_pool],
              mode='concat', concat_axis=channel_axis,
              name='mixed4')

    # mixed 5, 6: 17 x 17 x 768
    for i in range(2):
        branch1x1 = conv2d_bn(x, 192, 1, 1)

        branch7x7 = conv2d_bn(x, 160, 1, 1)
        branch7x7 = conv2d_bn(branch7x7, 160, 1, 7)
        branch7x7 = conv2d_bn(branch7x7, 192, 7, 1)

        branch7x7dbl = conv2d_bn(x, 160, 1, 1)
        branch7x7dbl = conv2d_bn(branch7x7dbl, 160, 7, 1)
        branch7x7dbl = conv2d_bn(branch7x7dbl, 160, 1, 7)
        branch7x7dbl = conv2d_bn(branch7x7dbl, 160, 7, 1)
        branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 1, 7)

        branch_pool = AveragePooling2D(
            (3, 3), strides=(1, 1), border_mode='same')(x)
        branch_pool = conv2d_bn(branch_pool, 192, 1, 1)
        x = merge([branch1x1, branch7x7, branch7x7dbl, branch_pool],
                  mode='concat', concat_axis=channel_axis,
                  name='mixed' + str(5 + i))

    # mixed 7: 17 x 17 x 768
    branch1x1 = conv2d_bn(x, 192, 1, 1)

    branch7x7 = conv2d_bn(x, 192, 1, 1)
    branch7x7 = conv2d_bn(branch7x7, 192, 1, 7)
    branch7x7 = conv2d_bn(branch7x7, 192, 7, 1)

    branch7x7dbl = conv2d_bn(x, 160, 1, 1)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 7, 1)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 1, 7)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 7, 1)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 1, 7)

    branch_pool = AveragePooling2D((3, 3), strides=(1, 1), border_mode='same')(x)
    branch_pool = conv2d_bn(branch_pool, 192, 1, 1)
    x = merge([branch1x1, branch7x7, branch7x7dbl, branch_pool],
              mode='concat', concat_axis=channel_axis,
              name='mixed7')

    # mixed 8: 8 x 8 x 1280
    branch3x3 = conv2d_bn(x, 192, 1, 1)
    branch3x3 = conv2d_bn(branch3x3, 320, 3, 3,
                          subsample=(2, 2), border_mode='valid')

    branch7x7x3 = conv2d_bn(x, 192, 1, 1)
    branch7x7x3 = conv2d_bn(branch7x7x3, 192, 1, 7)
    branch7x7x3 = conv2d_bn(branch7x7x3, 192, 7, 1)
    branch7x7x3 = conv2d_bn(branch7x7x3, 192, 3, 3,
                            subsample=(2, 2), border_mode='valid')

    branch_pool = AveragePooling2D((3, 3), strides=(2, 2))(x)
    x = merge([branch3x3, branch7x7x3, branch_pool],
              mode='concat', concat_axis=channel_axis,
              name='mixed8')

    # mixed 9: 8 x 8 x 2048
    for i in range(2):
        branch1x1 = conv2d_bn(x, 320, 1, 1)

        branch3x3 = conv2d_bn(x, 384, 1, 1)
        branch3x3_1 = conv2d_bn(branch3x3, 384, 1, 3)
        branch3x3_2 = conv2d_bn(branch3x3, 384, 3, 1)
        branch3x3 = merge([branch3x3_1, branch3x3_2],
                          mode='concat', concat_axis=channel_axis,
                          name='mixed9_' + str(i))

        branch3x3dbl = conv2d_bn(x, 448, 1, 1)
        branch3x3dbl = conv2d_bn(branch3x3dbl, 384, 3, 3)
        branch3x3dbl_1 = conv2d_bn(branch3x3dbl, 384, 1, 3)
        branch3x3dbl_2 = conv2d_bn(branch3x3dbl, 384, 3, 1)
        branch3x3dbl = merge([branch3x3dbl_1, branch3x3dbl_2],
                             mode='concat', concat_axis=channel_axis)

        branch_pool = AveragePooling2D(
            (3, 3), strides=(1, 1), border_mode='same')(x)
        branch_pool = conv2d_bn(branch_pool, 192, 1, 1)
        x = merge([branch1x1, branch3x3, branch3x3dbl, branch_pool],
                  mode='concat', concat_axis=channel_axis,
                  name='mixed' + str(9 + i))
    '''
    if include_top:
        # Classification block
        x = AveragePooling2D((8, 8), strides=(8, 8), name='avg_pool')(x)
        x = Flatten(name='flatten')(x)
        x = Dense(1000, activation='softmax', name='predictions')(x)
    '''

    # Create model
    model = Model(img_input, x)

    # load weights
    if weights == 'imagenet':
        weights_path = get_file('/Users/gerrypesavento/Documents/recycle_ai/datax/gh_train/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5',
                                TF_WEIGHTS_PATH_NO_TOP,
                                cache_subdir='models',
                                md5_hash='2f3609166de1d967d1a481094754f691')
        model.load_weights(weights_path)

    print ('InceptionV3 model loaded')

    return model

# used for prediction 
def preprocess_input(x):
    x /= 255.
    x -= 0.5
    x *= 2.
    return x

def extract_features(model):
    
    print ('--- running extract_features')

    datagen = ImageDataGenerator(rescale=1./255)

    # Train generator - generates batched of augmented normalized data 
    generator = datagen.flow_from_directory(
            train_dir,
            target_size=(img_width, img_height),
            batch_size=32,
            class_mode="categorical",
            shuffle=False)

    # Train labels 
    _, y = datagen.flow_from_directory(
            train_dir,
            target_size=(img_width, img_height),
            batch_size=nb_train_images,   
            class_mode="categorical",  
            shuffle=False).next()
    np.save(open('train_labels.npy', 'w'), y)

    # class_labels.json 
    import json
    json.dump(generator.class_indices, open("class_labels.json", "w"))
    print ("labels.json created")
    print ("creating the bottleneck_features_train.npy ...")

    # Train bottleneck features 
    bottleneck_features_train = model.predict_generator(generator, nb_train_images)
    np.save(open('bottleneck_features_train.npy', 'w'), bottleneck_features_train)
    print ('bottleneck_features_train.npy saved')

    # Validation generator - generates batched of augmented normalized data 
    generator = datagen.flow_from_directory(
            valid_dir,
            target_size=(img_width, img_height),
            batch_size=32,
            class_mode="categorical",
            shuffle=False)

    # Validation labels 
    _, y = datagen.flow_from_directory(
            valid_dir,  
            target_size=(img_width, img_height),
            batch_size=nb_valid_images,
            class_mode="categorical",
            shuffle=False).next()
    np.save(open('valid_labels.npy', 'w'), y)

    # Validation bottleneck features 
    bottleneck_features_validation = model.predict_generator(generator, nb_valid_images)
    np.save(open('bottleneck_features_valid.npy', 'w'), bottleneck_features_validation)
    print ('bottleneck_features_valid.npy saved')

def build_top_model(input_shape):

    print ('running build_top_model')
    model = Sequential()
    model.add(Convolution2D(32, 3, 3, input_shape=input_shape, border_mode='same', activation='relu'))  
    model.add(Dropout(0.2))
    model.add(Convolution2D(32, 3, 3, activation='relu', border_mode='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(BatchNormalization()) 
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nb_train_classes, activation='softmax'))
    return model

def train_top_model(model):

    print ('running train_top_model')

    # Train and Validation features and labels
    train_data = np.load(open('bottleneck_features_train.npy'))
    train_labels = np.load(open('train_labels.npy'))
    valid_data = np.load(open('bottleneck_features_valid.npy'))
    valid_labels = np.load(open('valid_labels.npy'))

    # compile model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(train_data, train_labels, nb_epoch=30, batch_size=32, validation_data=(valid_data, valid_labels))
    
    # save rai top model 
    top_model_weights_path = 'rai_top_model.h5'
    model.save_weights(top_model_weights_path)

    # print model data 
    print ("success, rai_top_model.h5 saved")
    print (model.summary())   


if __name__ == '__main__':
    print ('--- running main')
    model = InceptionV3(include_top=False, weights='imagenet')
    extract_features(model)   
    top_model = build_top_model(input_shape=model.output_shape[1:]) # model.output_shape[1:] = (1, 1, 2048)
    train_top_model(top_model)


'''
#  note - try this decode_predictions method 
if __name__ == '__main__':
    model = InceptionV3(include_top=True, weights='imagenet')

    img_path = 'elephant.jpg'
    img = image.load_img(img_path, target_size=(299, 299))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)

    x = preprocess_input(x)

    preds = model.predict(x)
    print('Predicted:', decode_predictions(preds))

''' 