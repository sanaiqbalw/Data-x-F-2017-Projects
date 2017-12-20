'''
    warp images in a folder 
    if i < 3 ... used to create warped images 
'''

import os 
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

datagen = ImageDataGenerator(
        rotation_range=45, # Int. Degree range for random rotations
        width_shift_range=0.2, # Float (fraction of total width). Range for random horizontal shifts
        height_shift_range=0.2, #  Float (fraction of total height). Range for random vertical shifts
        shear_range=0.2, # Float. Shear Intensity (Shear angle in counter-clockwise direction as radians)
        zoom_range=0.2, # Float or [lower, upper]. Range for random zoom
        horizontal_flip=True, # Boolean. Randomly flip inputs horizontally
        vertical_flip=True, # Boolean. Randomly flip inputs vertically
        fill_mode='reflect') # "constant", "nearest", "reflect" or "wrap" points outside the boundaries are filled

'''
img = load_img('warptest.jpg')  # this is a PIL image
x = img_to_array(img)  # this is a Numpy array with shape (3, 150, 150)
x = x.reshape((1,) + x.shape)  # this is a Numpy array with shape (1, 3, 150, 150)
'''

towarp_dir = '/Users/geraldp/gdrive/recycle_ai/dev_train_data/train/112'

for dir,subdir,files in os.walk(towarp_dir):
    print dir
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
