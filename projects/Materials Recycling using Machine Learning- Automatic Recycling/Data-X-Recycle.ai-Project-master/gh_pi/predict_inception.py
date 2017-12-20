'''
    Build inception models 
'''
import numpy as np
import json

from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.models import Model
from keras.layers import Flatten, Dense, Input
from keras.layers import BatchNormalization, merge, Dropout
from keras.layers import Convolution2D, MaxPooling2D, AveragePooling2D
from keras.utils.layer_utils import convert_all_kernels_in_model
from keras.utils.data_utils import get_file
from keras import backend as K


# InceptionV3 model
def conv2d_bn(x, nb_filter, nb_row, nb_col,
              border_mode='same', subsample=(1, 1), name=None):
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


def InceptionV3(include_top=True, weights='imagenet', input_tensor=None):
    '''
        Inception v3 architecture,
        returns a Keras model instance
    '''
    # print (round((time() - t0), 2), 'InceptionV3 function: start ')

    # Determine proper input shape
    input_shape = (256, 256, 3)
    img_input = Input(shape=input_shape)  # no tensor img_input
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

    # Create model - runs fast
    model = Model(img_input, x)

    # path to TF weights 
    TF_WEIGHTS_PATH_NO_TOP = 'file:/home/pi/Desktop/inception/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'

    # load weights
    # print (round((time() - t0), 2), 'InceptionV3 function: start load weights')
    if weights == 'imagenet':
        weights_path = get_file('inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5',
                                TF_WEIGHTS_PATH_NO_TOP,
                                cache_subdir='models',
                                md5_hash='2f3609166de1d967d1a481094754f691')
        model.load_weights(weights_path)
    # print (round((time() - t0), 2), 'InceptionV3 function: finish load weights')

    return model

def preprocess_input(x):
    x /= 255.
    x -= 0.5
    x *= 2.
    return x

def build_top_model(input_shape):
    
    # define constants
    nb_train_classes = 88

    # print (round((time() - t0), 2), 'build_top_model function start')
    model = Sequential()
    model.add(Convolution2D(32, 3, 3, input_shape=input_shape, border_mode='same',
                            activation='relu'))
    model.add(Dropout(0.2))
    model.add(Convolution2D(32, 3, 3, activation='relu', border_mode='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(BatchNormalization())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nb_train_classes, activation='softmax'))
    # print (round((time() - t0), 2), 'build_top_model function end')
    return model

def load_recycle_ai_model(input_shape):

    # point to top model weight path 
    top_model_weights_path = '/home/pi/Desktop/inception/rai_top_model.h5'

    # print (round((time() - t0), 2), 'load_recycle_ai_model function start')
    model = build_top_model(input_shape)
    model.load_weights(top_model_weights_path)
    # print (round((time() - t0), 2), 'load_recycle_ai_model function end')
    return model

# run the object prediction function
def predict(image_path):

    # image_path from PiCamera 
    img = image.load_img(image_path, target_size=(256, 256))
    x = image.img_to_array(img)  # do I /255.0 here?
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    # print (round((time() - t0), 2), "test image " + image_path, "shape ", x.shape)

    # classifier_output is a numpy array of scores from keras .predict
    inception_bottleneck_feature = inception_model.predict(x)
    classifier_output = recycle_ai_model.predict(inception_bottleneck_feature)
    # print (round((time() - t0), 2), "classifier output", classifier_output)

    # the column with the max score, ex. classifier_output_columns [45], maxcolumn 45
    classifier_output_columns = np.argmax(classifier_output, axis=1)
    maxcolumn = classifier_output_columns[0]
    # print ('\nthe classifier_output with the highest score is ', maxcolumn)

    # this array is column:object_id
    name2column = json.load(open("class_labels.json"))
    column2name = dict([(v, k) for k, v in name2column.items()])
    # print ("\ncolumn2name is classifier_output : u object_id ", column2name)

    # the object_id for highest weight, example result is 43
    theobject = column2name[classifier_output_columns[0]]
    # print ('\ntheobject', theobject)

    # thescore = classifier_output[classifier_output_columns[0]]
    thescore = classifier_output[0, maxcolumn]
    thescore = int(100*thescore)
    if (thescore < 10):
        thescore = '00'+str(thescore)
    if (thescore < 100 and thescore >= 10):
        thescore = '0'+str(thescore)
    if (thescore == 100):
        thescore = str(thescore)
    # print ('\n theweight classifier_output[0,classifier_output_result]', thescore)

    return (theobject, thescore)


# build inceptionv3 model
# print (round((ctime() - t0), 2), "InceptionV3 START")
inception_model = InceptionV3(include_top=False, weights='imagenet')
# print (round((ctime() - t0), 2), "InceptionV3 END")

# load_recycle_ai_model, build_top_model, load top model weights
# print (round((ctime() - t0), 2), "load_recycle_ai_model START")
recycle_ai_model = load_recycle_ai_model(input_shape=inception_model.output_shape[1:])
# print (round((ctime() - t0), 2), "load_recycle_ai_model END")

