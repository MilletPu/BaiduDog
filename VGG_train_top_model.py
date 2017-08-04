#-*- coding:utf-8 -*-
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from keras import applications
import pandas as pd

# dimensions of our images.
img_width, img_height = 150, 150
top_model_weights_path = 'bottleneck_fc_model.h5'
train_data_dir = './data/train_set/train'
validation_data_dir = './data/validation/vali_set/test1'
log_path = "./logs/split_logs.csv"
df = pd.read_csv(log_path)
nb_train_samples = df['train_num'].sum()
nb_validation_samples = df['vali_num'].sum()
print(nb_train_samples)
epochs = 50
batch_size = 16


def save_bottlebeck_features():
    datagen = ImageDataGenerator( rescale=1./255 )
    # build the VGG16 network
    model = applications.VGG16( include_top=False, weights='imagenet' )
    #######################################################################################
    generator = datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        shuffle=False)
    bottleneck_features_train = model.predict_generator( generator, nb_train_samples // batch_size)
    np.save( open('vgg_bottleneck_features_train.npy', 'w'), bottleneck_features_train )
    #######################################################################################
    generator = datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        shuffle=False)
    bottleneck_features_validation = model.predict_generator( generator, nb_validation_samples // batch_size)
    np.save( open('vgg_bottleneck_features_validation.npy', 'w'), bottleneck_features_validation)


def train_top_model():
    train_data = np.load(open('bottleneck_features_train.npy'))
    train_labels = np.array( [0] * (nb_train_samples / 2) + [1] * (nb_train_samples / 2) )
    validation_data = np.load(open('bottleneck_features_validation.npy'))
    validation_labels = np.array( [0] * (nb_validation_samples / 2) + [1] * (nb_validation_samples / 2) )

    model = Sequential()
    model.add(Flatten(input_shape=train_data.shape[1:]))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='softmax'))
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(
        train_data, 
        train_labels,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(validation_data, validation_labels))
    model.save_weights(top_model_weights_path)

if __name__ == '__main__':   
    save_bottlebeck_features()
    # train_top_model()