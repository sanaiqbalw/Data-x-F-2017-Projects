# imports
from __future__ import print_function
from time import sleep
from time import time as ctime 
import os
import sys
import h5py

# import threading 
from multiprocessing import Process

from imagenet_utils import decode_predictions, preprocess_input
from multiprocessing import Process
from picamera import PiCamera

import pymysql
from datetime import datetime
from random import *
import boto3

import predict_inception
from predict_inception import InceptionV3
from predict_inception import load_recycle_ai_model
from predict_inception import predict

from predict_motor import led_red
from predict_motor import led_blue
from predict_motor import led_yellow
from predict_motor import led_green
from predict_motor import move_recycle
from predict_motor import move_compost
from predict_motor import move_landfill

from predict_similarity import similarity

# start clock
t0 = ctime()

if __name__ == "__main__":

    # number of photos to take for this cycle
    numofphotos = 10

    # reset the clock
    t0 = ctime()
    print ('\n')

    # set up the camera, and a warm-up time (2)
    camera = PiCamera()
    camera.resolution = (512, 512)
    camera.start_preview()
    camera.preview_fullscreen = False
    camera.preview_window = (1200, 50, 512, 512)
    sleep(2)

    # connect to the database to get full name for display
    DBServer = '192.95.31.34'
    DBUser = 'rai'
    DBPass = 'Apricot1olo!'
    DBName = 'raiobjects'
    connection = pymysql.connect(host=DBServer, user=DBUser,
                                 passwd=DBPass, db=DBName)

    # create an S3 service client
    access_key = 'ADD_KEY_HERE'
    secret_key = 'ADD_SECRET_HERE'
    client = boto3.client('s3', aws_access_key_id=access_key,
                          aws_secret_access_key=secret_key)

    #  get images
    for i in range(1, numofphotos):

        # capture a 512px and 256px photo named now256.jpg
        os.rename('now256.jpg', 'past256.jpg')
        camera.capture('now512.jpg')
        photoname = 'now256.jpg'
        camera.capture(photoname, resize=(256, 256))

        # get similarity to previous photo
        img1 = 'now256.jpg'
        img2 = 'past256.jpg'
        similarity_to_previous = similarity(img1, img2)

        # get the current time
        thetime = round((ctime() - t0), 1)

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
                client.upload_file('now256.jpg', 'rai-objects', path256, ExtraArgs={'StorageClass': 'REDUCED_REDUNDANCY'})
                client.upload_file('now512.jpg', 'rai-objects', path512, ExtraArgs={'StorageClass': 'REDUCED_REDUNDANCY'})

                # get the bin for motor movement, display object recognized
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
                    print ('\n[', i, '-', thetime, '] RECYCLE!', objectdesc, thescore, '\n')
                    p1 = Process(target=led_blue)
                    p2 = Process(target=move_recycle)
                    p1.start()
                    p2.start()
                    p1.join()
                    p2.join()

                if(thebin == 'compost'):
                    print ('\n[', i, '-', thetime, '] COMPOST!', objectdesc, thescore, '\n')
                    p1 = Process(target=led_green)
                    p2 = Process(target=move_compost)
                    p1.start()
                    p2.start()
                    p1.join()
                    p2.join()

                if(thebin == 'landfill'):
                    print ('\n[', i, '-', thetime, '] LANDFILL :(', objectdesc, thescore, '\n')
                    p1 = Process(target=led_yellow)
                    p2 = Process(target=move_landfill)
                    p1.start()
                    p2.start()
                    p1.join()
                    p2.join()

                if(thebin == 'hazardous'):
                    led_red()
                    print ('\n[', i, '-', thetime, '] HAZARDOUS?', objectdesc, thescore, '\n')

                if(thebin == 'other'):
                    led_red()
                    print ('[', i, '-', thetime, '] HOLD, ', objectdesc, thescore)

            else:
                # go directly to thumb tag
                path256 = '0/SPT/TRN/' + current + '-' + str(thescore) + str(rand) + '-256.jpg'
                path512 = '0/SPT/ORG/' + current + '-' + str(thescore) + str(rand) + '-512.jpg'
                client.upload_file('now256.jpg', 'rai-objects', path256, ExtraArgs={'StorageClass': 'REDUCED_REDUNDANCY'})
                client.upload_file('now512.jpg', 'rai-objects', path512, ExtraArgs={'StorageClass': 'REDUCED_REDUNDANCY'})
                print ('[', i, '-', thetime, '] HOLD, low confidence', thescore, 'uploaded to for analysis')
                # flashred()
        else:
            print ('[', i, '-', thetime, '] HOLD, no object change, similarity', similarity_to_previous, 'too high')

    connection.close()
    camera.stop_preview()
