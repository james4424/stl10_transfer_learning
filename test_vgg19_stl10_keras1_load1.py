from __future__ import print_function

from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential, Model 
from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D
from keras import backend as k 
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping
from keras.utils import np_utils
from keras.applications.vgg19 import decode_predictions

import sys
import os, sys, tarfile
import numpy as np
import matplotlib.pyplot as plt
#import skimage
#import skimage.io
#import skimage.transform
import scipy

if sys.version_info >= (3, 0, 0):
    import urllib.request as urllib # ugly but works
else:
    import urllib

print(sys.version_info) 

'''
This script is adapted from source codes publicly available, and which can be found below:
https://towardsdatascience.com/transfer-learning-using-keras-d804b2e04ef8
https://github.com/mttk/STL10
'''

# image shape
HEIGHT = 96
WIDTH = 96
DEPTH = 3

# size of a single image in bytes
SIZE = HEIGHT * WIDTH * DEPTH

# path to the directory with the data
DATA_DIR = './data'

# url of the binary data
DATA_URL = 'http://ai.stanford.edu/~acoates/stl10/stl10_binary.tar.gz'

# path to the binary train file with image data
DATA_PATH = './data/stl10_binary/train_X.bin'
#DATA_PATH = './data/stl10_binary/unlabeled_X.bin'


# path to the binary train file with labels
LABEL_PATH = './data/stl10_binary/train_y.bin'


DATA_PATH1 = './data/stl10_binary/test_X.bin'
#DATA_PATH = './data/stl10_binary/unlabeled_X.bin'


# path to the binary train file with labels
LABEL_PATH1 = './data/stl10_binary/test_y.bin'

DATA_PATH2 = './data/stl10_binary/unlabeled_X.bin'



def read_single_image(image_file):
    """
    CAREFUL! - this method uses a file as input instead of the path - so the
    position of the reader will be remembered outside of context of this method.
    :param image_file: the open file containing the images
    :return: a single image
    """
    # read a single image, count determines the number of uint8's to read
    image = np.fromfile(image_file, dtype=np.uint8, count=SIZE)
    # force into image matrix
    image = np.reshape(image, (3, 96, 96))
    # transpose to standard format
    # You might want to comment this line or reverse the shuffle
    # if you will use a learning algorithm like CNN, since they like
    # their channels separated.
    image = np.transpose(image, (2, 1, 0))
    return image

def read_all_images(path_to_data):
    """
    :param path_to_data: the file containing the binary images from the STL-10 dataset
    :return: an array containing all the images
    """

    with open(path_to_data, 'rb') as f:
        # read whole file in uint8 chunks
        everything = np.fromfile(f, dtype=np.uint8)

        # We force the data into 3x96x96 chunks, since the
        # images are stored in "column-major order", meaning
        # that "the first 96*96 values are the red channel,
        # the next 96*96 are green, and the last are blue."
        # The -1 is since the size of the pictures depends
        # on the input file, and this way numpy determines
        # the size on its own.

        images = np.reshape(everything, (-1, 3, 96, 96))
        #images = np.reshape(everything, (-1, 3, 224, 224))

        # Now transpose the images into a standard image format
        # readable by, for example, matplotlib.imshow
        # You might want to comment this line or reverse the shuffle
        # if you will use a learning algorithm like CNN, since they like
        # their channels separated.
        images = np.transpose(images, (0, 3, 2, 1))
        return images

def read_labels(path_to_labels):
    """
    :param path_to_labels: path to the binary file containing labels from the STL-10 dataset
    :return: an array containing the labels
    """
    with open(path_to_labels, 'rb') as f:
        labels = np.fromfile(f, dtype=np.uint8)
        return labels

def plot_image(image):
    """
    :param image: the image to be plotted in a 3-D matrix format
    :return: None
    """
    plt.imshow(image)
    plt.show()

# returns the top1 string
def print_prob(prob, file_path):
    synset = [l.strip() for l in open(file_path).readlines()]

    # print prob
    pred = [np.argmax(x) for x in prob]

    # Get top1 label
    top1 = [synset[x] for x in pred]
    return top1


def download_and_extract():
    """
    Download and extract the STL-10 dataset
    :return: None
    """
    dest_directory = DATA_DIR
    if not os.path.exists(dest_directory):
        os.makedirs(dest_directory)
    filename = DATA_URL.split('/')[-1]
    filepath = os.path.join(dest_directory, filename)
    if not os.path.exists(filepath):
        def _progress(count, block_size, total_size):
            sys.stdout.write('\rDownloading %s %.2f%%' % (filename,
                float(count * block_size) / float(total_size) * 100.0))
            sys.stdout.flush()
        filepath, _ = urllib.urlretrieve(DATA_URL, filepath, reporthook=_progress)
        print('Downloaded', filename)
        tarfile.open(filepath, 'r:gz').extractall(dest_directory)

if __name__ == "__main__":

	# download data if needed
	#download_and_extract()

	img_width, img_height = 256, 256
	#train_data_dir = "data/train"
	#validation_data_dir = "data/val"
	nb_train_samples = 5000
	nb_validation_samples = 8000 
	batch_size = 20
	epochs = 50
	classes = 10

	model = applications.VGG19(weights = "imagenet", include_top=False, input_shape = (img_width, img_height, 3))

	"""
	Layer (type)                 Output Shape              Param #   
	=================================================================
	input_1 (InputLayer)         (None, 256, 256, 3)       0         
	_________________________________________________________________
	block1_conv1 (Conv2D)        (None, 256, 256, 64)      1792      
	_________________________________________________________________
	block1_conv2 (Conv2D)        (None, 256, 256, 64)      36928     
	_________________________________________________________________
	block1_pool (MaxPooling2D)   (None, 128, 128, 64)      0         
	_________________________________________________________________
	block2_conv1 (Conv2D)        (None, 128, 128, 128)     73856     
	_________________________________________________________________
	block2_conv2 (Conv2D)        (None, 128, 128, 128)     147584    
	_________________________________________________________________
	block2_pool (MaxPooling2D)   (None, 64, 64, 128)       0         
	_________________________________________________________________
	block3_conv1 (Conv2D)        (None, 64, 64, 256)       295168    
	_________________________________________________________________
	block3_conv2 (Conv2D)        (None, 64, 64, 256)       590080    
	_________________________________________________________________
	block3_conv3 (Conv2D)        (None, 64, 64, 256)       590080    
	_________________________________________________________________
	block3_conv4 (Conv2D)        (None, 64, 64, 256)       590080    
	_________________________________________________________________
	block3_pool (MaxPooling2D)   (None, 32, 32, 256)       0         
	_________________________________________________________________
	block4_conv1 (Conv2D)        (None, 32, 32, 512)       1180160   
	_________________________________________________________________
	block4_conv2 (Conv2D)        (None, 32, 32, 512)       2359808   
	_________________________________________________________________
	block4_conv3 (Conv2D)        (None, 32, 32, 512)       2359808   
	_________________________________________________________________
	block4_conv4 (Conv2D)        (None, 32, 32, 512)       2359808   
	_________________________________________________________________
	block4_pool (MaxPooling2D)   (None, 16, 16, 512)       0         
	_________________________________________________________________
	block5_conv1 (Conv2D)        (None, 16, 16, 512)       2359808   
	_________________________________________________________________
	block5_conv2 (Conv2D)        (None, 16, 16, 512)       2359808   
	_________________________________________________________________
	block5_conv3 (Conv2D)        (None, 16, 16, 512)       2359808   
	_________________________________________________________________
	block5_conv4 (Conv2D)        (None, 16, 16, 512)       2359808   
	_________________________________________________________________
	block5_pool (MaxPooling2D)   (None, 8, 8, 512)         0         
	=================================================================
	Total params: 20,024,384.0
	Trainable params: 20,024,384.0
	Non-trainable params: 0.0
	"""

	# Freeze the layers which you don't want to train. Here I am freezing the first 5 layers.
	#for layer in model.layers[:5]:
	#    layer.trainable = False

	for layer in model.layers:
	   layer.trainable = False

	#Adding custom Layers 
	x = model.output
	x = Flatten()(x)
	x = Dense(1024, activation="relu")(x)
	x = Dropout(0.5)(x)
	x = Dense(1024, activation="relu")(x)
	predictions = Dense(classes, activation="softmax")(x)

	# creating the final model 
	model_final = Model(inputs = model.input, outputs = predictions)
	model_final.load_weights("vgg16_1.h5")

	# compile the model 
	model_final.compile(loss = "categorical_crossentropy", optimizer = optimizers.SGD(lr=0.0001, momentum=0.9), metrics=["accuracy"])

	# Initiate the train and test generators with data Augumentation 
	train_datagen = ImageDataGenerator(
	rescale = 1./255,
	horizontal_flip = True,
	fill_mode = "nearest",
	zoom_range = 0.3,
	width_shift_range = 0.3,
	height_shift_range=0.3,
	rotation_range=30)

	test_datagen = ImageDataGenerator(
	rescale = 1./255,
	horizontal_flip = True,
	fill_mode = "nearest",
	zoom_range = 0.3,
	width_shift_range = 0.3,
	height_shift_range=0.3,
	rotation_range=30)

	new_shape = (img_width, img_height, 3)

	x_train = read_all_images(DATA_PATH)
	x_train = np.array([scipy.misc.imresize(image, new_shape) for image in x_train]) 
	print(x_train.shape)
	plot_image(x_train[0])

	y_train = read_labels(LABEL_PATH)
	y_train = np_utils.to_categorical(np.expand_dims((y_train-1),axis=1), classes)     
	print(y_train.shape)

	x_valid = read_all_images(DATA_PATH1)
	x_valid = np.array([scipy.misc.imresize(image, new_shape) for image in x_valid]) 
	print(x_valid.shape)
	plot_image(x_valid[0])

	y_valid = read_labels(LABEL_PATH1)
	y_valid = np_utils.to_categorical(np.expand_dims((y_valid-1),axis=1), classes)     
	print(y_valid.shape)

	
	x_test = read_all_images(DATA_PATH2)
	#x_test = np.array([scipy.misc.imresize(image, new_shape) for image in x_test]) 
	#print(x_test.shape)
	#plot_image(x_test[0])
	

	'''
	train_generator = train_datagen.flow_from_directory(
	train_data_dir,
	target_size = (img_height, img_width),
	batch_size = batch_size, 
	class_mode = "categorical")

	validation_generator = test_datagen.flow_from_directory(
	validation_data_dir,
	target_size = (img_height, img_width),
	class_mode = "categorical")
	'''

	#train_generator = train_datagen.flow(x_train, y_train, batch_size=batch_size)

	#validation_generator = test_datagen.flow(x_valid, y_valid)



	# Save the model according to the conditions  
	#checkpoint = ModelCheckpoint("vgg16_1.h5", monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
	#early = EarlyStopping(monitor='val_acc', min_delta=0, patience=10, verbose=1, mode='auto')

	'''
	# Train the model 
	model_final.fit_generator(
	train_generator,
	steps_per_epoch = nb_train_samples/batch_size,
	epochs = epochs,
	validation_data = validation_generator,
	validation_steps = nb_validation_samples/batch_size,
	callbacks = [checkpoint, early])
	'''

	scores = model_final.evaluate(x_valid, y_valid, verbose=0)
	print("Accuracy: {}".format(scores))

	'''
	y_test_predicted1 = model_final.predict(x_valid)
	#pred = decode_predictions(np.asarray(y_test_predicted))
	y_test_predicted_singleDigit = np.argmax(y_test_predicted1,axis=1)
	print("Accuracy1: {}".format(float(np.size(np.where(y_test_predicted_singleDigit==np.argmax(y_valid,axis=1))))/float(y_valid.shape[0]))


	scores = model_final.evaluate(np.expand_dims(x_valid[0],axis=0), np.expand_dims(y_valid[0],axis=0), verbose=0)
	print("Accuracy: {}".format(scores))

	y_test_predicted = model_final.predict(np.expand_dims(x_valid[0],axis=0))
	'''

	predlist = []
	for i in range(x_test.shape[0]):
		tmp = scipy.misc.imresize(x_test[i], new_shape)
		prob = model_final.predict(np.expand_dims(tmp,axis=0))
		predlist.append(prob[0])

	prediction = print_prob(predlist, './synset10.txt')


	f = open('prediction.txt', 'w')
	for i in range(len(prediction)):
		f.write(prediction[i]+"\n")
	f.close()

