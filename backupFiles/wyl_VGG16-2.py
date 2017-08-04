#-*- coding:utf-8 -*-
import numpy as np
import pandas as pd
import os,math
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, Conv2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
from keras.models import load_model
from keras import applications
from keras.models import Model
from keras.applications.resnet50 import ResNet50

from model.models import VGG_16
# dimensions of our images.
img_width, img_height = 224, 224   # 模型需要的参数大小
img_dir = "./imgs/" 
train_data_dir = './data/train'
validation_data_dir = './data/validation'
log_path = "./logs/split_logs.csv"
df = pd.read_csv(log_path)
nb_train_samples = df['train_num'].sum()
nb_validation_samples = df['vali_num'].sum()
print nb_train_samples,nb_validation_samples
epochs = 10 # best is 47
batch_size = 512
map2filename = lambda name: str( name[name.index("/")+1:name.index(".")] )
cnn_model_path = "./models/vgg_recent.model"

def getClass2DirDict(img_dir = "./imgs"):
    class2DirDict = dict();
    dirs = sorted( os.listdir(img_dir) )
    for i in range(len(dirs)):
        class2DirDict[i] = int( dirs[i] )
    return class2DirDict

def get_max_id(x_narray): # 获取最大值的下标
    res = []
    for ary in x_narray:
        if abs(sum(ary))<1e-5: print sum(ary)
        _max = np.max(ary)
        for i in range(len(ary)):
            if abs(_max-ary[i])<1e-5: 
                res.append(i)
                break;
    return res

def get_model():
    if K.image_data_format() == 'channels_first': # 通道
        input_shape = (3, img_width, img_height)
    else:
        input_shape = (img_width, img_height, 3)
    #######################################################################
    #model = VGG_16(input_shape=input_shape)
    #######################################################################
    vgg_model = applications.VGG16(
        include_top=False,
        weights='imagenet',
        input_shape = input_shape,
    )
    #initialise top model
    top_model = Sequential()
    top_model.add(Flatten(input_shape=vgg_model.output_shape[1:]))
    top_model.add(Dense(4096, activation='relu'))
    top_model.add(Dropout(0.35))
    top_model.add(Dense(512, activation='relu'))
    top_model.add(Dropout(0.35))
    top_model.add(Dense(100, activation='softmax'))

    model = Model(input= vgg_model.input, output= top_model(vgg_model.output))
    #######################################################################
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def train_model():
    print "begin to process imgs and train model......"
    #if os.path.exists(cnn_model_path): os.remove(cnn_model_path); print "Remove file %s."%(cnn_model_path)
    if os.path.exists(cnn_model_path): return load_model(cnn_model_path)
    # 图片生成器
    train_datagen = ImageDataGenerator( 
        rescale = 1./255, #重放缩因子,默认为None. 如果为None或0则不进行放缩,否则会将该数值乘到数据上(在应用其他变换之前)
        shear_range = 0.2, #浮点数，剪切强度（逆时针方向的剪切变换角度）
        zoom_range = 0.2,  #浮点数或形如[lower,upper]的列表，随机缩放的幅度，若为浮点数，则相当于[lower,upper] = [1 - zoom_range, 1+zoom_range]
        horizontal_flip=True, #布尔值，进行随机水平翻转
    )
    vali_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range = 0.2,
        zoom_range = 0.2,
        horizontal_flip=True,
    )
    # 使用.flow_from_directory()来从我们的jpgs图片中直接产生数据和标签
    # 每次随机采样batch_size个样本,无限循环的取
    train_generator = train_datagen.flow_from_directory(
        train_data_dir, # 目标文件夹路径,对于每一个类,该文件夹都要包含一个子文件夹.子文件夹中任何JPG、PNG和BNP的图片都会被生成器使用.详情请查看此脚本
        target_size = (img_width, img_height), # 整数tuple,默认为(256, 256). 图像将被resize成该尺寸
        batch_size = batch_size, # batch数据的大小,默认32
        class_mode = 'categorical', # http://keras-cn.readthedocs.io/en/latest/preprocessing/image/
        seed = 2017, # 可选参数,打乱数据和进行变换时的随机数种子
    )
    vali_generator = vali_datagen.flow_from_directory(
        validation_data_dir,
        target_size = (img_width, img_height),
        batch_size = batch_size,
        class_mode = 'categorical',
        seed = 2017,
    )
    model = get_model()

    model.fit_generator(
        train_generator, #训练数据无限生成器
        steps_per_epoch = nb_train_samples // batch_size, # 让训练集生成器随机采样几轮数据
        epochs = epochs, # 训练的轮数
        workers = 50, # 最大进程数
        validation_data = vali_generator,
        validation_steps = nb_validation_samples // batch_size, # 让验证集生成器随机采样几轮数据作为验证
    )
    model.save(cnn_model_path)
    return model

def predict(model):
    test_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range = 0.2,
        zoom_range = 0.2,
        horizontal_flip=True,
    )
    generator = test_datagen.flow_from_directory(
        './data/test_set',  #'./data/test_set/image'
        target_size = (img_width, img_height),
        batch_size = batch_size,
        class_mode = 'categorical',
        shuffle = False, # 默认为随机打乱数据
    )
    class2DirDict = getClass2DirDict() #类名到文件名的映射
    pre_data = []
    for i in range( generator.samples//batch_size + 1 ):
        x_test, y_test = generator.next()
        res_narray = model.predict( x_test )
        res_narray =  get_max_id(res_narray)
        pre_data.extend( res_narray )
        print "iter %d / %d"%(i+1, generator.samples // batch_size + 1)
    sub_df = pd.DataFrame( pre_data, columns = ["pre"])
    sub_df['pre'] = sub_df['pre'].map( lambda x: class2DirDict[x] )
    sub_df['imgID'] = pd.Series(generator.filenames).map( map2filename )
    sub_df.to_csv('./sub/submission.txt', sep="\t", index=False, index_label=False, header=False)

if __name__ == '__main__':
    model = train_model();
    predict(model)

