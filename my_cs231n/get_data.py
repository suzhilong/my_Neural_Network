# -*- coding: utf-8 -*-
import re
from decimal import Decimal

class Get_data(object):
	"""
	get data from features.txt
	
	example:
		path = '/home/su/code/sEMG/3.14sEMG/code/data/features'
		fileName = 'feature1.txt'
		data = Get_data(path, fileName)
		data.get_data()
		chan1_features = data.features['chan1']
		ect.

	"""
	def __init__(self, path, fileName):
		self.path = path
		self.fileName = fileName
		self.features = {}

	def get_data(self):
		file = open(self.path +'/'+self.fileName,'r')
		matchList = re.findall(r'[0-9.-]+',file.read())
		features_list = []	#每19个一组的特征值列表
		temp = []
		for i in range(len(matchList)):
			if i % 19 == 0:
				temp = []
			temp.append(Decimal(matchList[i]))
			if len(temp) == 19:
				features_list.append(temp)
		#分开通道
		chan1_featurelist = features_list[0:len(features_list)/4]
		chan2_featurelist = features_list[len(features_list)/4:len(features_list)/2]
		chan3_featurelist = features_list[len(features_list)/2:len(features_list)*3/4]
		chan4_featurelist = features_list[len(features_list)*3/4:]
		self.features['chan1'] = chan1_featurelist
		self.features['chan2'] = chan2_featurelist
		self.features['chan3'] = chan3_featurelist
		self.features['chan4'] = chan4_featurelist


'''
#未封装
path = '/home/su/code/sEMG/3.14sEMG/code/data/features'
fileName = 'feature1.txt'
file = open(path +'/'+fileName,'r')
matchList = re.findall(r'[0-9.-]+',file.read())
#print len(matchList)
features_list = []	#每19个一组的特征值列表
temp = []
for i in range(len(matchList)):
	if i % 19 == 0:
		temp = []
	temp.append(matchList[i])
	if len(temp) == 19:
		features_list.append(temp)
#print len(features_list)

#分开通道
chan1_featurelist = features_list[0:len(features_list)/4]
chan2_featurelist = features_list[len(features_list)/4:len(features_list)/2]
chan3_featurelist = features_list[len(features_list)/2:len(features_list)*3/4]
chan4_featurelist = features_list[len(features_list)*3/4:]
#print len(chan1_featurelist)
'''
