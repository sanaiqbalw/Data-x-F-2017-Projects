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
from skimage.transform import resize
from keras.optimizers import SGD

width, height = 150, 150

training_path = "/datasets/train"
val_path = "/datasets/validation"
# training_path = "trashnet_data/train"
# val_path = "trashnet_data/validation"
n_train = 2400
n_val = 400
batch_size = 32

if K.image_data_format() =='channels_first':
    input_shape = (3, width, height)
else:
    input_shape = (width, height, 3)


# create the base pre-trained model
base_model = InceptionV3(weights='imagenet', include_top=False)

# add a global spatial average pooling layer
# base_model.add(GlobalAveragePooling2D())
# base_model.add(Flatten())
# base_model.add(Dense(1024, activation='relu'))
# base_model.add(Dropout(0.5))
# base_model.add(Dense(1024, activation='relu'))
# base_model.add(Dense(6, activation='softmax'))
x = base_model.output
x = GlobalAveragePooling2D()(x)
# let's add a fully-connected layer
# x = Flatten()(x)
x = Dense(1000, activation='relu')(x)
x = Dropout(0.2)(x)
x = Dense(1000, activation='relu')(x)
x = Dropout(0.2)(x)
x = Dense(1000, activation='relu')(x)
# and a logistic layer -- let's say we have 200 classes
predictions = Dense(6, activation='softmax')(x)

# this is the model we will train
model = Model(inputs=base_model.input, outputs=predictions)
# model = base_model

# first: train only the top layers (which were randomly initialized)
# i.e. freeze all convolutional InceptionV3 layers
for layer in base_model.layers:
    layer.trainable = False

# compile the model (should be done *after* setting layers to non-trainable)
model.compile(optimizer=SGD(lr=0.01 , momentum=0.9), loss='categorical_crossentropy', metrics=['categorical_accuracy'])




train_data = ImageDataGenerator(
        rescale= 1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_data = ImageDataGenerator(rescale=1./255)

train = train_data.flow_from_directory(
        training_path,
        class_mode='categorical',
        batch_size=batch_size,
        target_size=(width,height))

validation = test_data.flow_from_directory(
        val_path,
        class_mode='categorical',
        batch_size=batch_size,
        target_size=(width,height))


history = model.fit_generator(
        train,
        steps_per_epoch = n_train // batch_size,
        epochs = 30,
        validation_data = validation,
        validation_steps = n_val // batch_size
        )

with open('/output/myfile0.txt', 'a') as f:
    f.write(str(history.history))


# at this point, the top layers are well trained and we can start fine-tuning
# convolutional layers from inception V3. We will freeze the bottom N layers
# and train the remaining top layers.

# let's visualize layer names and layer indices to see how many layers
# we should freeze:
for i, layer in enumerate(base_model.layers):
   print(i, layer.name)

# we chose to train the top 2 inception blocks, i.e. we will freeze
# the first 249 layers and unfreeze the rest:
for layer in model.layers[:101]:
   layer.trainable = False
for layer in model.layers[101:]:
   layer.trainable = True

# we need to recompile the model for these modifications to take effect
# we use SGD with a low learning rate
model.compile(optimizer=SGD(lr=0.01 , momentum=0.9), loss='categorical_crossentropy', metrics=['categorical_accuracy'])

# we train our model again (this time fine-tuning the top 2 inception blocks
# alongside the top Dense layers
history = model.fit_generator(
        train,
        steps_per_epoch = n_train // batch_size,
        epochs = 30,
        validation_data = validation,
        validation_steps = n_val // batch_size
        )

with open('/output/myfile.txt', 'a') as f:
    f.write(str(history.history))

model.save_weights('/output/trash-weights.hdf5')
model.save('/output/my_model.h5')

# from keras.utils import plot_model
# plot_model(model, to_file='/output/model.png')



from scipy import misc
import os
import os.path
for f in os.listdir("/datasets/test"):
    print(f)
    image = resize(misc.imread("/datasets/test" +f),(150,150,3)).reshape(1,150,150,3)

    prediction = model.predict(image)
    print(prediction)
