# imports 
from __future__ import print_function
from time import time, sleep
import os
import sys
import h5py  
import numpy as np
import json
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential 
from keras.models import Model
from keras.layers import Flatten, Dense, Input, BatchNormalization, merge, Dropout
from keras.layers import Convolution2D, MaxPooling2D, AveragePooling2D
from keras.utils.layer_utils import convert_all_kernels_in_model
from keras.utils.data_utils import get_file
from keras import backend as K
from imagenet_utils import decode_predictions, preprocess_input

from picamera import PiCamera
import RPi.GPIO as GPIO
from Adafruit_MotorHAT import Adafruit_MotorHAT
import atexit
import pymysql
from datetime import datetime
from random import *
from PIL import Image
import boto3

# start clock 
t0 = time()
print (round((time() - t0), 2), 'starting engine, dependencies loaded')

# weights paths 
top_model_weights_path = '/home/pi/Desktop/inception/rai_top_model.h5'
TF_WEIGHTS_PATH_NO_TOP = 'file:/home/pi/Desktop/inception/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'

# define constants 
nb_train_classes = 88
img_width, img_height = 256, 256

# InceptionV3 model 
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

def InceptionV3(include_top=True, weights='imagenet', input_tensor=None):
    '''
        Inception v3 architecture,
        returns a Keras model instance
    '''
    print (round((time() - t0), 2), 'InceptionV3 function: start ')
    
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

    # Create model - runs fast 
    model = Model(img_input, x)

    # load weights
    print (round((time() - t0), 2), 'InceptionV3 function: start load weights')
    if weights == 'imagenet':
        weights_path = get_file('inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5',
                                TF_WEIGHTS_PATH_NO_TOP,
                                cache_subdir='models',
                                md5_hash='2f3609166de1d967d1a481094754f691')
        model.load_weights(weights_path)
    print (round((time() - t0), 2), 'InceptionV3 function: finish load weights')

    return model

def preprocess_input(x):
    x /= 255.
    x -= 0.5
    x *= 2.
    return x

def build_top_model(input_shape):
    print (round((time() - t0), 2), 'build_top_model function start')
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
    print (round((time() - t0), 2), 'build_top_model function end')
    return model

def load_recycle_ai_model(input_shape):
    print (round((time() - t0), 2), 'load_recycle_ai_model function start')
    model = build_top_model(input_shape)
    model.load_weights(top_model_weights_path)
    print (round((time() - t0), 2), 'load_recycle_ai_model function end')
    return model

# build inceptionv3 model 
print (round((time() - t0), 2), "InceptionV3 START")
inception_model = InceptionV3(include_top=False, weights='imagenet')
print (round((time() - t0), 2), "InceptionV3 END")

# load_recycle_ai_model, build_top_model, load top model weights 
print (round((time() - t0), 2), "load_recycle_ai_model START")
recycle_ai_model = load_recycle_ai_model(input_shape=inception_model.output_shape[1:])
print (round((time() - t0), 2), "load_recycle_ai_model END")

# for image pmatch 
def avhash(im):
    if not isinstance(im, Image.Image):
        im = Image.open(im)
    im = im.resize((8, 8), Image.ANTIALIAS).convert('L')
    avg = reduce(lambda x, y: x + y, im.getdata()) / 64.
    return reduce(lambda x, (y, z): x | (z << y),
                  enumerate(map(lambda i: 0 if i < avg else 1, im.getdata())),
                  0)

def hamming(h1, h2):
    h, d = 0, h1 ^ h2
    while d:
        h += 1
        d &= d - 1
    return h

# run the object prediction function 
def predict(image_path):

    # Pi Camera get images here 

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
    classifier_output_columns =  np.argmax(classifier_output,axis=1)
    maxcolumn = classifier_output_columns[0]
    # print ('\nthe classifier_output with the highest score is ', maxcolumn)

    # this array is column:object_id 
    name2column = json.load(open("class_labels.json"))
    column2name = dict([(v,k) for k,v in name2column.items()])
    # print ("\ncolumn2name is classifier_output : u object_id ", column2name)

    # the object_id for highest weight, example result is 43 
    theobject = column2name[classifier_output_columns[0]]
    # print ('\ntheobject', theobject)

    # thescore = classifier_output[classifier_output_columns[0]]
    thescore = classifier_output[0,maxcolumn]
    thescore = int(100*thescore)
    if (thescore < 10):
        thescore = '00'+str(thescore)
    if (thescore <100 and thescore >=10):
        thescore = '0'+str(thescore)
    if (thescore == 100):
        thescore = str(thescore)
    # print ('\n theweight classifier_output[0,classifier_output_result]', thescore)

    return (theobject, thescore)

# LED blinking function
def blink(pin):
    GPIO.output(pin,GPIO.HIGH)
    sleep(.1)
    GPIO.output(pin,GPIO.LOW)
    sleep(.1)
    GPIO.output(pin,GPIO.HIGH)
    sleep(.1)
    GPIO.output(pin,GPIO.LOW)
    sleep(.5)

# LED flash green function TEMP change to GREEN LED
def flashgreen():
    GPIO.setmode(GPIO.BOARD)
    GPIO.setup(40, GPIO.OUT)
    # print ('LED green')
    for i in range(0,3):
        blink(40)
    GPIO.cleanup()

# LED flash red function 
def flashred():
    GPIO.setmode(GPIO.BOARD)
    GPIO.setup(40, GPIO.OUT)
    # print ("LED red")
    for i in range(0,3):
        blink(40)
    GPIO.cleanup()

# turn off all motors 
def turnOffMotors():
    mh.getMotor(1).run(Adafruit_MotorHAT.RELEASE)
    mh.getMotor(2).run(Adafruit_MotorHAT.RELEASE)
    mh.getMotor(3).run(Adafruit_MotorHAT.RELEASE)
    mh.getMotor(4).run(Adafruit_MotorHAT.RELEASE)
    print("Motors released")


if __name__ == "__main__":

    # number of photos to take for this cycle 
    numofphotos = 10

    #reset the clock
    t0 = time()
    print ('\n')

    # set up the camera, and a warm-up time (2)
    camera = PiCamera()
    camera.resolution = (512, 512)
    camera.start_preview()
    camera.preview_fullscreen = False
    camera.preview_window = (1200,50,512,512)
    sleep(2)

    # connect to the database to get full name for display 
    DBServer = '192.95.31.34' 
    DBUser   = 'rai'
    DBPass   = 'Apricot1olo!'
    DBName   = 'raiobjects'
    connection = pymysql.connect(host=DBServer, user=DBUser, passwd=DBPass, db=DBName)

    # create an S3 service client 
    access_key = 'ADD_KEY_HERE'
    secret_key = 'ADD_SECRET_HERE'
    client = boto3.client('s3', aws_access_key_id=access_key, aws_secret_access_key=secret_key)

    # motor parameters
    mh = Adafruit_MotorHAT(addr = 0x60)
    atexit.register(turnOffMotors)
    myStepper = mh.getStepper(200, 1)
    myStepper.setSpeed(200)

    #  get images 
    for i in range(1,numofphotos):

        # capture a 512px and 256px photo named now256.jpg 
        os.rename('now256.jpg','past256.jpg')
        camera.capture('now512.jpg')
        photoname = 'now256.jpg'
        camera.capture(photoname, resize=(256, 256))

        # get similarity to previous photo
        img1 = 'now256.jpg'
        img2 = 'past256.jpg'
        hash1 = avhash(img1)
        hash2 = avhash(img2)
        dist = hamming(hash1, hash2)
        similarity_to_previous = (64-dist)*100/64

        # get the current time 
        thetime = round((time() - t0), 1)

        # detect when there is a new object  
        if similarity_to_previous < 95:

            # get predicted object and score 
            (theobject, thescore) = predict(image_path=photoname)

            # upload photo and prediction to S3 for training data 
            current = datetime.now().strftime('%Y%m%d%H%M%S')
            rand = randint(100, 999)

            if (int(thescore) > 98):
                
                # go to keep tag 
                path256 = theobject + '/SPT/TRN/' + current + '-' + str(thescore) + str(rand) + '-256.jpg'
                path512 = theobject + '/SPT/ORG/' + current + '-' + str(thescore) + str(rand) + '-512.jpg'
                client.upload_file('now256.jpg', 'rai-objects', path256, ExtraArgs={'StorageClass':'REDUCED_REDUNDANCY'})
                client.upload_file('now512.jpg', 'rai-objects', path512, ExtraArgs={'StorageClass':'REDUCED_REDUNDANCY'})

                # display the object recognized to the terminal 
                with connection.cursor() as cursor:
                    sql = ('SELECT * FROM object WHERE object_id = {}'.format(theobject))
                    cursor.execute(sql)
                    result = cursor.fetchall()
                    for row in result:
                        object_id = row[0]
                        thebin = row[1]
                        material = row[2]
                        object_type = row[3]
                        level1 = row[4]
                        level2 = row[5]
                        level3 = row[6]
                        level4 = row[7]
                        sku = row[8]
                objectdesc = str(object_id) + ' ' + thebin + ' ' + material + ' ' + object_type + ' ' + level1 + ' ' + level2 + ' ' + level3 + ' ' + level4

                if(thebin == 'recycle'):
                    print ('\n[',i,'-',thetime,'] RECYCLE!',objectdesc,thescore,'\n') 
                    flashgreen()
                    myStepper.step(200, Adafruit_MotorHAT.FORWARD, Adafruit_MotorHAT.DOUBLE)
                    sleep(1)
                    myStepper.step(200, Adafruit_MotorHAT.BACKWARD, Adafruit_MotorHAT.DOUBLE)

                if(thebin == 'compost'):
                    print ('\n[',i,'-',thetime,'] COMPOST!',objectdesc,thescore,'\n')
                    flashgreen()
                    myStepper.step(200, Adafruit_MotorHAT.BACKWARD, Adafruit_MotorHAT.DOUBLE)
                    sleep(1)
                    myStepper.step(200, Adafruit_MotorHAT.FORWARD, Adafruit_MotorHAT.DOUBLE)

                if(thebin == 'landfill'):
                    print ('\n[',i,'-',thetime,'] LANDFILL :(',objectdesc,thescore,'\n')
                    flashgreen()
                    myStepper.step(10, Adafruit_MotorHAT.FORWARD, Adafruit_MotorHAT.DOUBLE)
                    sleep(1)
                    myStepper.step(10, Adafruit_MotorHAT.BACKWARD, Adafruit_MotorHAT.DOUBLE)

                if(thebin == 'hazardous'):
                    flashred()
                    print ('\n[',i,'-',thetime,'] HAZARDOUS?',objectdesc,thescore,'\n')

                if(thebin == 'other'):
                    flashred()
                    print ('[',i,'-',thetime,'] HOLD, ',objectdesc,thescore)

            else: 
                # go directly to thumb tag 
                path256 = '0/SPT/TRN/' + current + '-' + str(thescore) + str(rand) + '-256.jpg'
                path512 = '0/SPT/ORG/' + current + '-' + str(thescore) + str(rand) + '-512.jpg'
                client.upload_file('now256.jpg', 'rai-objects', path256, ExtraArgs={'StorageClass':'REDUCED_REDUNDANCY'})
                client.upload_file('now512.jpg', 'rai-objects', path512, ExtraArgs={'StorageClass':'REDUCED_REDUNDANCY'})
                print ('[',i,'-',thetime,'] HOLD, low confidence',thescore,'uploaded to for analysis')
                # flashred()
        else:
            print ('[',i,'-',thetime,'] HOLD, no object change, similarity',similarity_to_previous,'too high')


    connection.close()
    camera.stop_preview()

    


