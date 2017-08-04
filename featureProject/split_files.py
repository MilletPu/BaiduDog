#-*- coding:utf-8 -*-
import os,math
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
import shutil 
import time,  datetime

train_data_path = "./data/train" #训练数据所在目录(/类别/**.jpg )
vali_data_path = "./data/validation"
split_ratio = 0.8

# 获得图片目录下所有文件名称,并生成名称字典
def get_file_name_dic(file_path):
	# {类别:[文件名称,...];...}
	print "get file name dict"
	ans_dic = dict()
	for root,dirs,file_names in os.walk(file_path+"/"):
		for _dir in dirs:
			for root,dirs,file_names in os.walk(file_path+"/"+_dir+"/"):
				for file_name in file_names:
					class_id = _dir
					if ans_dic.get(class_id)==None:
						ans_dic[class_id] = []
					ans_dic[class_id].append(file_path+"/"+_dir+"/"+file_name)
	print "end build file name dict"
	return ans_dic

#将一个文件copy到指定目录
def moveFileto( sourceDir, targetDir ): 
	shutil.copy( sourceDir, targetDir )

# 删除目录下的所有文件
def removeDir(dirPath):
	if not os.path.isdir(dirPath):
		return
	files = os.listdir(dirPath)
	try:
		for file in files:
			filePath = os.path.join(dirPath, file)
			if os.path.isfile(filePath):
				os.remove(filePath)
			elif os.path.isdir(filePath):
				removeDir(filePath)
		os.rmdir(dirPath)
	except Exception, e:
		print e

def copyAllSamples2Imgs():
	removeDir("./imgs"); os.mkdir("./imgs");
	################ 读取train_set中的文件 #################
	imgs_from_path = "./data/train_set/train/"
	with open("./data/data_train_image.txt","r") as f:
		train_dic = dict(); #{文件名称:所属类别;}
		for line in f.readlines():
			data = line.split(" ")
			train_dic[ data[0] ] = "%.3d"%(int(data[1]))
	for root,dirs,file_names in os.walk(imgs_from_path):
		for file_name in file_names:
			key = file_name.split(".")[0]
			class_id = train_dic.get(key)
			if class_id==None: 
				print file_name + " not exit in map."
			from_file_name_path = imgs_from_path + file_name
			to_file_name_path = "./imgs/"+class_id+"/"+file_name
			if os.path.exists("./imgs/"+class_id+"/")==False: 
				os.mkdir("./imgs/"+class_id+"/");
			moveFileto(from_file_name_path,to_file_name_path)
	################# 读取vali_set中的文件 #################
	imgs_from_path = "./data/vali_set/test1/"
	with open("./data/val.txt","r") as f:
		train_dic = dict(); #{文件名称:所属类别;}
		for line in f.readlines():
			data = line.split(" ")
			train_dic[ data[0] ] = "%.3d"%(int(data[1]))
	for root,dirs,file_names in os.walk(imgs_from_path):
		for file_name in file_names:
			key = file_name.split(".")[0]
			class_id = train_dic.get(key)
			if class_id==None: 
				print file_name + " not exit in map."
				continue;
			from_file_name_path = imgs_from_path + file_name
			to_file_name_path = "./imgs/"+class_id+"/"+file_name
			if os.path.exists("./imgs/"+class_id+"/")==False: 
				os.mkdir("./imgs/"+class_id+"/");
			moveFileto(from_file_name_path,to_file_name_path)

#切分训练集和验证机图片主程序
def split2data( split_ratio = 0.8 ):
	########################### 重建目录结构 ##################################
	removeDir(train_data_path); os.mkdir(train_data_path);
	removeDir(vali_data_path);  os.mkdir(vali_data_path);
	###########################################################################
	file_name_dic = get_file_name_dic("./imgs");
	logs_array = []
	for key,value in file_name_dic.items():
		key_train_path = train_data_path+"/"+str(key)
		key_validation_path = vali_data_path+"/"+str(key)
		if os.path.exists(key_train_path) == False:      os.mkdir(key_train_path);
		if os.path.exists(key_validation_path) == False: os.mkdir(key_validation_path);
		################ copy 一类狗的img文件 #######################
		value = shuffle(value)
		for from_file_name in value[ :int(split_ratio*len(value)) ]:
			moveFileto( from_file_name, key_train_path )
		for from_file_name in value[ int(split_ratio*len(value)): ]:
			moveFileto( from_file_name, key_validation_path)
		logs_array.append( [key,len(value),int(split_ratio*len(value)),len(value)-int(split_ratio*len(value))] )
	logs_df = pd.DataFrame(np.array(logs_array),columns=['class_name','all_samples','train_num','vali_num'])
	logs_df['all_samples'] = logs_df['all_samples'].map(lambda x: int(x) )
	print "imgs has samples is ", np.sum( logs_df[['all_samples']] )
	logs_df.to_csv( "./logs/split_logs.csv", index=False, index_label=False )

if __name__ == '__main__':
	#copyAllSamples2Imgs()
	split2data( split_ratio )

