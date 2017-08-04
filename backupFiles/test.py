#-*- coding:utf-8 -*-
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
train_data_dir = './data/train_set/train'
validation_data_dir = 'data/validation/vali_set/test1'



def train_model(train_generator,validation_generator):
	model = Sequential()
	model.add(Conv2D(32, (3, 3), input_shape=(3, 150, 150)))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))

	model.add(Conv2D(32, (3, 3)))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))

	model.add(Conv2D(64, (3, 3)))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))

	# the model so far outputs 3D feature maps (height, width, features)

	model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
	model.add(Dense(64))
	model.add(Activation('relu'))
	model.add(Dropout(0.5))
	model.add(Dense(1))
	model.add(Activation('sigmoid'))

	model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
	model.fit_generator(
		train_generator,
		steps_per_epoch = 2000 // batch_size,
		epochs=50,
		validation_data = validation_generator,
		validation_steps = 800 // batch_size)
	model.save_weights('first_try.h5')  # always save your weights after training or during training
	return model

def picture_init():
	datagen = ImageDataGenerator(
		rotation_range=40,
		width_shift_range=0.2,
		height_shift_range=0.2,
		shear_range=0.2,
		zoom_range=0.2,
		horizontal_flip=True,
		fill_mode='nearest')
	batch_size = 16

	# this is the augmentation configuration we will use for training
	train_datagen = ImageDataGenerator(
		rescale=1./255,
		shear_range=0.2,
		zoom_range=0.2,
		horizontal_flip=True)

	# this is the augmentation configuration we will use for testing:
	# only rescaling
	test_datagen = ImageDataGenerator(rescale=1./255)

	# this is a generator that will read pictures found in
	# subfolers of 'data/train', and indefinitely generate
	# batches of augmented image data
	train_generator = train_datagen.flow_from_directory(
		'data/train',  # this is the target directory
		target_size=(150, 150),  # all images will be resized to 150x150
		batch_size=batch_size,
		class_mode='binary')  # since we use binary_crossentropy loss, we need binary labels

	# this is a similar generator, for validation data
	validation_generator = test_datagen.flow_from_directory(
		'data/validation',
		target_size=(150, 150),
		batch_size=batch_size,
		class_mode='binary')
	return train_generator,validation_generator

if __name__ == '__main__':
	train_generator,validation_generator = picture_init();
	train_model(train_generator,validation_generator);
