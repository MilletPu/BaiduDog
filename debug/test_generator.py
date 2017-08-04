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
img_width, img_height = 150, 150   # 模型需要的参数大小
train_data_dir = './data/train'
validation_data_dir = './data/validation'
log_path = "./logs/split_logs.csv"
df = pd.read_csv(log_path)
nb_train_samples = df['train_num'].sum()
nb_validation_samples = df['vali_num'].sum()
epochs = 50 #best is 47
batch_size = 5000

def get_max_id(x_narray): # 获取最大值的下标
    res = []
    for ary in x_narray:
        _max = np.max(ary)
        for i in range(len(ary)):
            if abs(_max-ary[i])<1e-5: res.append(i)
    return res

def getClass2DirDict(img_dir = "./imgs"):
    new_key = 0;
    class2DirDict = dict();
    dirs = os.listdir(img_dir)
    dirs = sorted(dirs)
    for i in range(len(dirs)):
        class2DirDict[i] = int( dirs[i] )
    return class2DirDict

def testDir():
    count = 0;
    filename2classDict = gen.class_indices
    testDict = getClass2DirDict()
    for i in range(200):
        key = "%.3d"%(i)
        value = filename2classDict.get(key)
        if value==None: continue
        if testDict[value] != i: print "error"
        count+=1
    print "count = ",count
    
# 图片生成器
train_datagen = ImageDataGenerator( 
    rescale = 1./255, #重放缩因子,默认为None. 如果为None或0则不进行放缩,否则会将该数值乘到数据上(在应用其他变换之前)
    shear_range = 0.2, #浮点数，剪切强度（逆时针方向的剪切变换角度）
    zoom_range = 0.2,  #浮点数或形如[lower,upper]的列表，随机缩放的幅度，若为浮点数，则相当于[lower,upper] = [1 - zoom_range, 1+zoom_range]
    horizontal_flip=True, #布尔值，进行随机水平翻转
)
#使用.flow_from_directory()来从我们的jpgs图片中直接产生数据和标签
gen = train_datagen.flow_from_directory(
    train_data_dir, #目标文件夹路径,对于每一个类,该文件夹都要包含一个子文件夹.子文件夹中任何JPG、PNG和BNP的图片都会被生成器使用.详情请查看此脚本
    target_size = (img_width, img_height), # 整数tuple,默认为(256, 256). 图像将被resize成该尺寸
    batch_size = batch_size, #batch数据的大小,默认32
    class_mode = 'categorical', # http://keras-cn.readthedocs.io/en/latest/preprocessing/image/
    shuffle = False, #默认为随机打乱数据
)
#test_x, test_y = gen.next();
#print get_max_id( test_y )
testDir();


exit()
for name in gen.filenames:
    print name[name.index("/")+1:name.index(".")]
print len(gen.filenames)


exit();
data = []
target = []
for i in gen:
	test_x, test_y = gen.next();
	test_x = np.array(test_x)
	target.append( get_max_id( test_y ) )
	data.extend( test_x.reshape((-1,150*150*3)) )
	print "iter"
df = pd.DataFrame(data)
df["target"] = target
df.to_csv("./cache/train.csv",index=False, index_label=False)


