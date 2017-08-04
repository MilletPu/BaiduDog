#-*- coding:utf-8 -*-
import numpy as np
import pandas as pd
import os,math
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras import optimizers
from keras.layers import Convolution2D, MaxPooling2D, Conv2D, GlobalAveragePooling2D 
from keras.layers import Activation, Dropout, Flatten, Dense, Dropout
from keras import backend as K
from keras.models import load_model
from keras.preprocessing import image
from keras import applications
from keras.models import Model
from keras.utils.data_utils import get_file
os.environ["CUDA_VISIBLE_DEVICES"] = "2"


img_width, img_height = 299, 299   # 模型需要的参数大小
epochs = 40 # best is 47
batch_size = 16
map2filename = lambda name: str( name[name.index("/")+1:name.index(".")] )


img_dir = "./imgs/" 
#train_data_dir = './data/train'
train_data_dir = "./imgs"
validation_data_dir = './data/validation'
test_data_dir = './data/test_set'
split_data_log_path = "./logs/split_logs.csv"
model_weights_file = "./models/Xception_weight_2.h5"
sub_file = './sub/Xception2_submission.txt'
#cnn_model_file = './models/cnn_basic_50_epochs.model'
#cnn_model_file = './models/cnn_basic_10_epochs.model'
cnn_model_file = "./models/Xception2Recent.model"


df = pd.read_csv(split_data_log_path)
nb_train_samples = df['train_num'].sum() + df['vali_num'].sum()
nb_validation_samples = df['vali_num'].sum()


def getClass2DirDict():
    # 获取{ 模型类别序号 <-> 文件夹名 }的映射
    class2DirDict = dict();
    dirs = sorted( os.listdir(img_dir) )
    for i in range(len(dirs)):
        class2DirDict[i] = int( dirs[i] )
    return class2DirDict

def reOneHot(x_narray):
    # 获取最大值的下标
    res = []
    for ary in x_narray:
        if abs(sum(ary))<1e-5: print sum(ary)
        _max = np.max(ary)
        for i in range(len(ary)):
            if abs(_max-ary[i])<1e-5: 
                res.append(i)
                break;
    return res

def buildGenerators():
    print "begin to buildGenerators......"
    # 图片生成器
    train_datagen = ImageDataGenerator(
        rescale = 1./255, #重放缩因子,默认为None. 如果为None或0则不进行放缩,否则会将该数值乘到数据上(在应用其他变换之前)
        shear_range = 0.2, #浮点数，剪切强度（逆时针方向的剪切变换角度）
        zoom_range = 0.2,  #浮点数或形如[lower,upper]的列表，随机缩放的幅度，若为浮点数，则相当于[lower,upper] = [1 - zoom_range, 1+zoom_range]
        horizontal_flip=True, #布尔值，进行随机水平翻转
    )
    vali_datagen = ImageDataGenerator( rescale=1./255, )
    test_datagen = ImageDataGenerator( rescale=1./255, )
    # 使用.flow_from_directory()来从我们的jpgs图片中直接产生数据和标签
    # 每次随机采样batch_size个样本,无限循环的取
    # http://keras-cn.readthedocs.io/en/latest/preprocessing/image/
    train_generator = train_datagen.flow_from_directory(
        train_data_dir, # 目标文件夹路径,对于每一个类,该文件夹都要包含一个子文件夹.子文件夹中任何JPG、PNG和BNP的图片都会被生成器使用.详情请查看此脚本
        target_size = (img_width, img_height), # 整数tuple,默认为(256, 256). 图像将被resize成该尺寸
        batch_size = batch_size, # batch数据的大小,默认32
        class_mode = 'categorical', 
        seed = 2017, # 可选参数,打乱数据和进行变换时的随机数种子
    )
    vali_generator = vali_datagen.flow_from_directory(
        validation_data_dir,
        target_size = (img_width, img_height),
        batch_size = batch_size,
        class_mode = 'categorical',
        seed = 2017,
    )
    test_generator = test_datagen.flow_from_directory(
        test_data_dir,  #'./data/test_set/image'
        target_size = (img_width, img_height),
        batch_size = batch_size,
        class_mode = 'categorical',
        shuffle = False, # 默认为随机打乱数据
    )
    return train_generator,vali_generator,test_generator

def buildModel( init = False ):
    print "begin to buildModel......"
    if K.image_data_format() == 'channels_first': # 通道
        input_shape = (3, img_width, img_height)
    else:
        input_shape = (img_width, img_height, 3)
    #################################### famous models #####################################
    # VGG16，VGG19以及ResNet接受224×224的输入图像,而InceptionV3和Xception要求为299×299
    model = applications.Xception( include_top=False, weights='imagenet', input_shape = input_shape, )
    #########################################################################################
    top_tensor = GlobalAveragePooling2D(name='avg_pool')(model.output)
    top_tensor = Dense(1024, activation='sigmoid', name='dense1')(top_tensor)
    top_tensor = Dense(512, activation='sigmoid', name='dense2')(top_tensor)
    top_tensor = Dense(100, activation='softmax', name='predictions')(top_tensor)
    if init==True: # 冻结的所有卷积层,预处理全连接层
        for layer in model.layers: 
            layer.trainable = False
        model = Model( model.input, top_tensor, name='Xception')
        #######################################################################
        model.compile (
            loss='categorical_crossentropy',
            #optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
            optimizer='adam', 
            metrics=['accuracy'],
        )
    else: # 微调整个网络
        model = Model( model.input, top_tensor, name='Xception')
        model.load_weights(model_weights_file)
        #######################################################################
        model.compile (
            loss='categorical_crossentropy',
            optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
            #optimizer='adam', 
            metrics=['accuracy'],
        )   
    return model

def trainModel( model, train_gen, vali_gen, train_epochs ):
    print "begin to trainModel......"
    if os.path.exists(cnn_model_file): os.remove(cnn_model_file); print "Remove file %s."%(cnn_model_file)
    if os.path.exists(cnn_model_file): return load_model(cnn_model_file)
    model.fit_generator(
        train_gen, #训练数据无限生成器
        steps_per_epoch = nb_train_samples // batch_size, # 让训练集生成器随机采样几轮数据
        epochs = train_epochs, # 训练的轮数
        workers = 10, # 最大进程数
        validation_data = vali_gen,
        validation_steps = nb_validation_samples // batch_size, # 让验证集生成器随机采样几轮数据作为验证
        verbose=2, #日志显示，0为不在标准输出流输出日志信息，1为输出进度条记录，2为每个epoch输出一行记录
    )
    model.save(cnn_model_file)
    model.save_weights(os.path.join(model_weights_file))
    return model

def predict( model,test_gen ):
    print "begin to predict......"
    class2DirDict = getClass2DirDict() #类名到文件名的映射
    #######################################################################
    pre_data = []
    buckets = test_gen.samples // batch_size + 1 # 分批预测结果
    for i in range(buckets):
        x_test, y_test = test_gen.next()
        res_narray = model.predict(x_test)
        res_narray =  reOneHot(res_narray)
        pre_data.extend( res_narray )
        if i % (buckets/10) == 0:
            print "iter %d / %d"%(i+1, buckets)
    #######################################################################
    print len( pre_data ), test_gen.samples, len(test_gen.filenames)
    sub_df = pd.DataFrame( pre_data, columns = ["pre"])
    sub_df['pre'] = sub_df['pre'].map( lambda x: class2DirDict[x] )
    sub_df['imgID'] = pd.Series(test_gen.filenames).map( map2filename )
    sub_df.to_csv( sub_file, sep="\t", index=False, index_label=False, header=False)


if __name__ == '__main__':
    train_gen,vali_gen,test_gen = buildGenerators();
    model = buildModel(init=True)
    model = trainModel( model, train_gen, vali_gen, train_epochs=3 );
    model = buildModel(init=False)
    model = trainModel( model, train_gen, vali_gen, train_epochs=epochs );
    predict( model,test_gen )

