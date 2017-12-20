from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras import backend as K

from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dropout, Activation, Flatten, Dense
from keras.callbacks import ModelCheckpoint, TensorBoard
import h5py
from keras import backend as K
import keras
import numpy as np

width, height = 150, 150

training_path = "/datasets/train"
# val_path = "validation"
# training_path = "trashnet_data/train"
# val_path = "trashnet_data/validation"
n_train = 2400
n_val = 400
epochs = 50
batch_size = 32



if K.image_data_format() =='channels_first':
    input_shape = (3, width, height)
else:
    input_shape = (width, height, 3)

# test_data = ImageDataGenerator(rescale=1./255)
# validation = test_data.flow_from_directory(
#         val_path,
#         class_mode='categorical',
#         batch_size=batch_size,
#         target_size=(150,150))


# create the base pre-trained model
base_model = InceptionV3(weights='imagenet', include_top=False)

# add a global spatial average pooling layer
x = base_model.output
x = GlobalAveragePooling2D()(x)
# let's add a fully-connected layer
x = Dense(1024, activation='relu')(x)
x = Dense(1024, activation='relu')(x)
# and a logistic layer -- let's say we have 200 classes
predictions = Dense(6, activation='softmax')(x)

# this is the model we will train
model = Model(inputs=base_model.input, outputs=predictions)

model.load_weights("trash-weights.hdf5")
print("loaded")

# # prediction = model.predict_generator(validation, steps = 1000)
# print(prediction)


from scipy import misc
import os
import os.path
from skimage.transform import resize
for f in os.listdir("test"):
    print(f)
    image = resize(misc.imread("test/" +f),(150,150,3)).reshape(1,150,150,3)

    prediction = model.predict(image)
    print("prediction", np.argmax(prediction,axis = 1))
