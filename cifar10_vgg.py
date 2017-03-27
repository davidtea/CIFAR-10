'''Using VGGNet-16 on the CIFAR10 small images dataset.

Code for VGGNet architecture from: https://gist.github.com/baraldilorenzo/07d7802847aaad0a35d3

'''

from __future__ import print_function
import keras
import time
import matplotlib as mpl
mpl.use('Agg')    # Save graph to file
import matplotlib.pyplot as plt
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Flatten, Dense, Dropout, Activation
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD
import cv2, numpy as np

now = time.time()

batch_size = 32
num_classes = 10
epochs = 25
data_augmentation = True

# input image dimensions
img_rows, img_cols = 32, 32
# The CIFAR10 images are RGB.
img_channels = 3

# The data, shuffled and split between train and test sets:
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# Convert class vectors to binary class matrices.
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()

# Block 1
model.add(ZeroPadding2D((1,1),input_shape=x_train.shape[1:]))
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))

# Block 2
model.add(ZeroPadding2D((1,1)))
model.add(Conv2D(128, (3, 3)))
model.add(Activation('relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Conv2D(128, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))

# Block 3
model.add(ZeroPadding2D((1,1)))
model.add(Conv2D(256, (3, 3)))
model.add(Activation('relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Conv2D(256, (3, 3)))
model.add(Activation('relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Conv2D(256, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))

# Block 4
model.add(ZeroPadding2D((1,1)))
model.add(Conv2D(512, (3, 3)))
model.add(Activation('relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Conv2D(512, (3, 3)))
model.add(Activation('relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Conv2D(512, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))

# Block 5
model.add(ZeroPadding2D((1,1)))
model.add(Conv2D(512, (3, 3)))
model.add(Activation('relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Conv2D(512, (3, 3)))
model.add(Activation('relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Conv2D(512, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))

model.add(Flatten())
model.add(Dense(4096))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(4096))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes))
model.add(Activation('softmax'))


#sgd = SGD(lr=0.1, decay=1e-3, momentum=1.0, nesterov=True)
adam = keras.optimizers.Adam(lr=0.001)
model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

'''
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
'''

hist = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_test, y_test), shuffle=True)
train_loss = hist.history['loss']
train_accuracy = hist.history['acc']
test_loss = hist.history['val_loss']
test_accuracy = hist.history['val_acc']
print(hist)

fig = plt.figure()     
plt.plot(train_accuracy, '-', test_accuracy, '-', train_loss, '--', test_loss, '--')
plt.title('VGGNet')
plt.ylabel('Train and Test Accuracy/Loss')
plt.xlabel('Epochs')
fig.savefig('graph.png', bbox_inches='tight')

print("Time Elapsed:", time.time() - now)
