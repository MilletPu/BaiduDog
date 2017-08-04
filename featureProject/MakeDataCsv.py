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


# dimensions of our images.
img_width, img_height,  = 256,256 # 模型需要的参数大小
data_dir = "./imgs/"
data_dir = "./data/validation"
dump_path = "./cache/Samples.csv"
batch_size = 2048

def getClassDict(): # 获取最大值的下标
    new_key = 0;
    classDict = dict();
    dirs = os.listdir(data_dir)
    for i in range(len(dirs)):
        classDict[i] = dirs[i]
    return classDict

def get_max_id(x_narray): # 获取最大值的下标
    res = []
    for ary in x_narray:
        _max = np.max(ary)
        for i in range(len(ary)):
            if abs(_max-ary[i])<1e-5: res.append(i)
    return res

def buildDataCsv():
    # 图片生成器
    ImgDataGen = ImageDataGenerator( 
        rescale = 1./255, #重放缩因子,默认为None. 如果为None或0则不进行放缩,否则会将该数值乘到数据上(在应用其他变换之前)
        shear_range = 0.2, #浮点数，剪切强度（逆时针方向的剪切变换角度）
        zoom_range = 0.2,  #浮点数或形如[lower,upper]的列表，随机缩放的幅度，若为浮点数，则相当于[lower,upper] = [1 - zoom_range, 1+zoom_range]
        horizontal_flip=True, #布尔值，进行随机水平翻转
    )
    # 使用.flow_from_directory()来从我们的jpgs图片中直接产生数据和标签
    gen = ImgDataGen.flow_from_directory(
        data_dir, # 目标文件夹路径,对于每一个类,该文件夹都要包含一个子文件夹.子文件夹中任何JPG、PNG和BNP的图片都会被生成器使用.详情请查看此脚本
        target_size = (img_width, img_height), # 整数tuple,默认为(256, 256). 图像将被resize成该尺寸
        batch_size = batch_size,    # batch数据的大小,默认32
        class_mode = 'categorical', # http://keras-cn.readthedocs.io/en/latest/preprocessing/image/
        shuffle = False,            # 默认为随机打乱数据
    )
    classDict = getClassDict()
    data = []
    target = []
    for i in range( gen.samples//batch_size + 1 ):
        data_x, data_y = gen.next();
        data_x = np.array(data_x)
        target.extend( get_max_id(data_y) )
        data.extend( data_x.reshape((-1,img_height*img_width*3)) )
        print "iter %d / %d"%(i+1,gen.samples//batch_size + 1)
    print "一共有样本%d个,len(target)=%d"%(len(data),len(target))
    df = pd.DataFrame( data )
    df["target"] = pd.Series(target).map( lambda x: classDict[x] )
    df.to_csv( dump_path, index=False, index_label=False)

if __name__ == '__main__':
    buildDataCsv()

