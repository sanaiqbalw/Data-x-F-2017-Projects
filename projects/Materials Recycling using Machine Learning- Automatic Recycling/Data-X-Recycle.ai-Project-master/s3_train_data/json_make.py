'''
    make the json file 
'''
from __future__ import print_function
import os 
import json

warp_dir = '/Users/geraldp/gdrive/recycle_ai/s3_train_data/train/'

i = 0 
json_string = '{'
for dir,subdir,files in os.walk(warp_dir):
    object_id = dir.rsplit('/', 1)[-1]
    if object_id != 0 and object_id != '':
        json_string += '"' + str(object_id) + '": ' + str(i) + ', '
        i += 1
json_string = json_string.rstrip(',')
json_string += '}'

text_file = open("class_labels.json", "w")
text_file.write(json_string)
text_file.close()



# print (json_string)
'''
for file in files:
    if file == '.DS_Store':
        badfile = dir + '/' + file
        print 'deleted .DS_Store file ' + badfile
        os.remove(badfile)
    elif ".jpg" not in file: 
        badfile = dir + '/' + file
        print 'deleted non-jpg file ' + badfile
        os.remove(badfile)
    else: 
        # print (file)
        img_dir = dir + '/' + file
        prefix = file.rsplit('.jpg', 1)[0]
        img = load_img(img_dir)  # this is a PIL image
        x = img_to_array(img)  # this is a Numpy array with shape (256, 256, 3)
        x = x.reshape((1,) + x.shape)  # this is a Numpy array with shape (1, 256, 256, 3)
        i = 0
        # the .flow() command generates batches of randomly transformed images
        for batch in datagen.flow(x, batch_size=1, save_to_dir=dir, save_prefix=prefix, save_format='jpg'):
            i += 1
            if i > 3:
           break  # otherwise the generator would loop indefinitely
    '''
