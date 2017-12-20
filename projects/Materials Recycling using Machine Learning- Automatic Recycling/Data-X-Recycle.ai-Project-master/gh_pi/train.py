'''
        Rapid photo capture from PiCamera for training data 
'''

from __future__ import print_function
from time import time, sleep
import os
import sys
from random import *
from datetime import datetime
from PIL import Image
import boto3
from picamera import PiCamera

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

if __name__ == "__main__":

    # number of photos to take for this cycle 
    numofphotos = 50

    #reset the clock
    t0 = time()

    # set up the camera, and a warm-up time (2)
    camera = PiCamera()
    camera.resolution = (512, 512)
    camera.start_preview()
    camera.preview_fullscreen = False
    camera.preview_window = (1200,50,512,512)
    sleep(2)

    # create an S3 service client 
    access_key = 'ADD_KEY_HERE'
    secret_key = 'ADD_SECRET_HERE'
    client = boto3.client('s3', aws_access_key_id=access_key, aws_secret_access_key=secret_key)

    #  get images 
    for i in range(1,numofphotos):

        # capture a 512px and 256px photo named now256.jpg 
        os.rename('now256.jpg','past256.jpg')
        camera.capture('now512.jpg')
        # sleep(1)
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

            # upload photo and prediction to S3 for training data 
            current = datetime.now().strftime('%Y%m%d%H%M%S')
            rand = randint(100, 999)
            thescore = 100
                
            # go to keep tag 
            path256 = '0/SPT/TRN/' + current + '-' + str(thescore) + str(rand) + '-256.jpg'
            path512 = '0/SPT/ORG/' + current + '-' + str(thescore) + str(rand) + '-512.jpg'
            client.upload_file('now256.jpg', 'rai-objects', path256, ExtraArgs={'StorageClass':'REDUCED_REDUNDANCY'})
            client.upload_file('now512.jpg', 'rai-objects', path512, ExtraArgs={'StorageClass':'REDUCED_REDUNDANCY'})
            print ('[',i,'-',thetime,'] uploaded')
           
        else:
            print ('[',i,'-',thetime,'] HOLD, no object change, similarity',similarity_to_previous,'too high')

    camera.stop_preview()

    


