#import logging  # kafeng
# from Controller import Controller, Controller_sequence, Controller_pure, Controller_attention, Controller_random, Controller_bms
from Controller_tran import Controller_tran, Controller_trajectory, Controller_dt
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings("ignore")
from multiprocessing import Pool, cpu_count, Process
#from multiprocessing import cpu_count, Process
from pathos.multiprocessing import ProcessingPool # kafeng for pass multi parameters
import pathos
import multiprocessing
from collections import ChainMap
from subprocess import Popen, PIPE
from time import time, sleep
import os
import random
# import tensorflow as tf
import datetime
import time
import copy

# from WeightedMinHashToolbox.WeightedMinHash import WeightedMinHash  # weiwu

# from utils_sklearn.py
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression, LogisticRegression, Lasso, Ridge
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn import svm
import sys
import numpy as np
import xlwt
from xlutils.copy import copy
import xlrd
from sklearn import metrics   # kafeng modify
from sklearn.metrics import f1_score  # kafeng modify
from scipy import stats
from sklearn.model_selection import RandomizedSearchCV, KFold
from sklearn.feature_selection import SelectFromModel  # kafeng add

# from gp.py 
import argparse
from xlutils.copy import copy
import shutil 
from sklearn.decomposition import KernelPCA
import math

#import autosklearn.classification
import pickle
#import cPickle
#from sklearn.externals import joblib 
import torch
import scipy.stats as stats

def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'

parser = argparse.ArgumentParser(description="Run.")
parser.add_argument('--num_op_unary', type=int,
	default=4, help='unary operation num')
parser.add_argument('--num_op_binary', type=int,
	default=5, help='binary operation num')
parser.add_argument('--max_order', type=int,
	default=5, help='max order of feature operation')
parser.add_argument('--num_batch', type=int,
	default=32, help='batch num')
parser.add_argument('--optimizer', nargs='?',
	default='adam', help='choose an optimizer')
parser.add_argument('--lr', type=float,
	default=0.01, help='set learning rate')
parser.add_argument('--epochs', type=int,
	default=100, help='training epochs')
parser.add_argument('--evaluate', nargs='?',
	default='f_score', help='choose evaluation method')  # 1-rae
parser.add_argument('--task', nargs='?',
	default='classification', help='choose between classification and regression')
parser.add_argument('--dataset', nargs='?',
	default='fertility', help='choose dataset to run  PimaIndian,fertility,hepatitis_cleaned')  # credit_default_float
parser.add_argument('--model', nargs='?',
	default='RF', help='choose a model')
parser.add_argument('--alpha', type=float,
	default=0.99, help='set discount factor')
parser.add_argument('--lr_value', type=float,
	default=1e-3, help='value network learning rate')
parser.add_argument('--RL_model', nargs='?',
	default='PG', help='choose RL model, PG or AC')
parser.add_argument('--reg', type=float,
	default=1e-5, help='regularization')
parser.add_argument('--controller', nargs='?',
	default='tran', help='choose a controller, dt, tran, bms, random, transfer, rnn, pure, attention')  # tran change from bms
parser.add_argument('--num_random_sample', type=int,
	default=5, help='sample num of random baseline')
parser.add_argument('--lambd', type=float,
	default=0.4, help='TD lambd')
#parser.add_argument('--multiprocessing', type=bool,
parser.add_argument('--multiprocessing', type=boolean_string,  # kafeng modify
	default=True, help='whether get reward using multiprocess True or False')
parser.add_argument('--package', nargs='?',
	default='sklearn', help='choose sklearn or weka to evaluate')
parser.add_argument('--num_process', type=int,
	default=48, help='process num')	 # kafeng add
parser.add_argument('--cache_method', nargs='?',
	default='selection', help='choose cache method, no_cache, selection ,or trees ')

parser.add_argument('--train_data', nargs='?',
	default='openml/', help='datasests   openml/ ,  paperData/')
parser.add_argument('--threshold', type=float,
	default=0.01, help='meta label threshold. f1 0.01, 0.015, 0.02,   recall 0.0')
parser.add_argument('--dimension_pcws', type=int,
	default=52, help='pcws output length for feature vector. 32: 0.63, 48: 0.709, 52: 0.709 , 56: 0.63,  64:0.665  128: 0.7339  256: ')
parser.add_argument('--PtoN', type=int,
	default=1, help='openml train posive to negtive')
parser.add_argument('--feature_extract_alg', nargs='?',
	default='CCWS', help='meta feature extract algorithm, such as minhash algorithm PCWS, statistic. CCWS 0.01 48 0.729, CCWS 0.01 52 0.90 ')
parser.add_argument('--minhash', type=boolean_string, 
	default=True, help='whether get reward using multiprocess True or False')

parser.add_argument('--log_dir', nargs='?',
	default='log', help='tensorboard log dir')

np.set_printoptions(suppress=True)

# from utils_sklearn.py
def mod_column(c1, c2):
	r = []
	for i in range(c2.shape[0]):
		if c2[i] == 0:
			r.append(0)
		else:
			r.append(np.mod(c1[i],c2[i]))
	return r


def paper_eval(train_x, test_x, train_y, test_y):
	pre_test = opengl_rf.predict(test_x)
	pre_prob = opengl_rf.predict_proba(test_x)
	return pre_test, pre_prob

synth_features = pd.DataFrame()

evaluate_count = 0 # kafeng add
evaluate_time = 0.0

constructed_features_list = [] # save indices
constructed_trees_list = []  # save trees for less time

selected_features_list = []  # for feature selection
selected_fscore_list = [] 

def evaluate(X, y, args):
	#global evaluate_count, evaluate_time, model
	global evaluate_count, evaluate_time, all_entropy_mean#, all_f1_score #, model  # mulprocess can distrube ???
	global constructed_features_list, constructed_trees_list, selected_features_list  # mulprocess can distrube ???
	
	s = 0

	evaluate_count += 1  # kafeng add
	#print('evaluate_count = ', evaluate_count)
	if args.task == 'regression':
		if args.model == 'LR':
			model = Lasso()
		elif args.model == 'RF':
			model = RandomForestRegressor(n_estimators=10, random_state=0)
			# add for cache
			select_rate = 0.5  # kafeng pay attention to featrures
			#print('int(np.ceil(X.shape[1]*select_rate)) = ', int(np.ceil(X.shape[1]*select_rate)))
			sfm = SelectFromModel(model, max_features=int(np.ceil(X.shape[1]*select_rate)))

		if args.evaluate == 'mae':
			s = cross_val_score(model, X, y, cv=5, 
				scoring='neg_mean_absolute_error').mean()
		elif args.evaluate == 'mse':
			s = cross_val_score(model, X, y, cv=5, 
				scoring='neg_mean_squared_error').mean()
		elif args.evaluate == 'r2':
			if args.cache_method == 'no_cache':
				s = cross_val_score(model, X, y, cv=5).mean()  # adaptive_valid
			elif args.cache_method == 'selection':
				# use feature selection
				sfm.fit(X, y)   # 
				#print('sfm.estimator_ = ', sfm.estimator_)  # RandomForestRegressor
				mask = sfm._get_support_mask()
				if mask.tolist() in selected_features_list:
					same_index = selected_features_list.index(mask.tolist())
					#print('same_index = ', same_index)
					return selected_fscore_list[same_index]

				selected_features_list.append(mask.tolist())
				#print(selected_features_list)
				
				X_transform = sfm.transform(X)  # 
				#print('X_transform.shape = ', X_transform.shape)
				s = cross_val_score(model, X=X_transform, y=y, cv=5).mean()  # auto run use f1_score  adaptive_valid

				selected_fscore_list.append(s)
				#print(selected_fscore_list)

	elif args.task == 'classification':
		le = LabelEncoder()
		y = le.fit_transform(y)    # kafeng pandas.core.series.Series to numpy.ndarray

		if args.model == 'RF':
			model = RandomForestClassifier(n_estimators=20, random_state=0)
			select_rate = 0.5  # kafeng pay attention to featrures
			sfm = SelectFromModel(model, max_features=int(np.ceil(X.shape[1]*select_rate)))

		elif args.model == 'LR':
			model = LogisticRegression(multi_class='ovr')
		elif args.model == 'SVM':
			model = svm.SVC()
		
		cv = 5
		if args.evaluate == 'f_score':
			if args.cache_method == 'no_cache':
				s = cross_val_score(model, X, y, scoring='f1_micro', cv=5).mean()
			elif args.cache_method == 'selection':
				# use feature selection
				sfm.fit(X, y)   # 0.03706 seconds   0.06638 seconds
				mask = sfm._get_support_mask()
				if mask.tolist() in selected_features_list:
					same_index = selected_features_list.index(mask.tolist())
					#print('same_index = ', same_index)
					return selected_fscore_list[same_index]

				selected_features_list.append(mask.tolist())
				X_transform = sfm.transform(X)  # 0.00307 seconds   0.00289 seconds
				s = cross_val_score(model, X=X_transform, y=y, scoring='f1_micro', cv=5).mean()  # 0.24863 seconds

				selected_fscore_list.append(s)
				#print(selected_fscore_list)
		elif args.evaluate == 'f1_macro':
			s = cross_val_score(model, X, y, scoring='f1_macro', cv=5).mean()
		elif args.evaluate == 'roc_auc':
			s = cross_val_score(model, X, y, scoring='roc_auc', cv=5).mean()
		elif args.evaluate == 'recall':
			s = cross_val_score(model, X, y, scoring='recall', cv=cv).mean()
		elif args.evaluate == 'precision':
			s = cross_val_score(model, X, y, scoring='precision', cv=cv).mean()	
	#print('s = ', s)
	all_f1_score.append(s)
	#print('all_f1_score = ', all_f1_score)
	return s


def transformation_search_space_attention(actions):
	#print('len(actions) = ', len(actions))
	action_per_feature = int(len(actions) / num_feature)   # 40/8 = 5
	transformed_feturs = {}
	max_order_fetrs =  []

	for feature_count in range(num_feature):      # 8
		feature_name = orig_data.columns[feature_count]
		feature_orders_actions = actions[feature_count*action_per_feature: (feature_count+1)*action_per_feature]  # 
		transformed_feturs[feature_count] = []     # kafeng 2 demension  8 * 5

		#print('feature_orders_actions = ', feature_orders_actions)
		if feature_orders_actions[0] == 0:
			continue
		else:
			#print('feature_name = ', feature_name)
			fetr = np.array(orig_data[feature_name].values)  # 0-7 columns 

		for action in feature_orders_actions:  # 5
			#print('action = ', action)
			if action == 0:  # EOF 1 # kafeng no transform max_orders, stop
				break
			elif action > 0 and action <= args.num_op_unary:   # 0-4 = 5 types
				# unary
				action_unary = action - 1
				if action_unary == 0:
					fetr = np.squeeze(np.sqrt(abs(fetr)))
				elif action_unary == 1:
					scaler = MinMaxScaler()
					fetr = np.squeeze(scaler.fit_transform(np.reshape(fetr,[-1,1])))
				elif action_unary == 2:
					while (np.any(fetr == 0)):
						fetr = fetr + 1e-5
					fetr = np.squeeze(np.log(abs(np.array(fetr)))) 
				elif action_unary == 3:
					while (np.any(fetr == 0)):
						fetr = fetr + 1e-5
					fetr = np.squeeze(1 / (np.array(fetr))) 
			else:   # 5-39 = 35 types    =5*7 
				# binary
				action_binary = (action-args.num_op_unary-1) // (num_feature-1)
				#print('action_binary = ', action_binary)
				rank = np.mod(action-args.num_op_unary-1, num_feature-1)
				#print('rank = ', rank)

				if rank >= feature_count:  #  kafeng ????
					rank += 1
				target_feature_name = orig_data.columns[rank]  # kafeng get next feature ???
				#print('target_feature_name = ', target_feature_name)
				target = np.array(orig_data[target_feature_name].values)

				if action_binary == 0:
					fetr = np.squeeze(fetr + target)  # 0-8  ,  actions decode column
				elif action_binary == 1:
					fetr = np.squeeze(fetr - target)
				elif action_binary == 2:
					fetr = np.squeeze(fetr * target)
				elif action_binary == 3:
					while (np.any(target == 0)):
						target = target + 1e-5
					fetr = np.squeeze(fetr / target)
				elif action_binary == 4:
					fetr = np.squeeze(mod_column(fetr, orig_data[target_feature_name].values))  # kafeng this will generate all 0 column ?

			if fetr.max() != fetr.min():
				transformed_feturs[feature_count].append(fetr)  # append 5 ,  1-5 transform orders
			else:
				continue

		if fetr.max() != fetr.min():
			max_order_fetrs.append(fetr)  # append 8 , use the last action , just for test ????  no thing about train ???
		else:
			continue
	#print(transformed_feturs)

	return transformed_feturs, max_order_fetrs



def transformation_search_space(actions):
	action_per_feature = int(len(actions) / num_feature)   # 40/8 = 5
	transformed_feturs, max_order_fetrs = {}, []

	for feature_count in range(num_feature):      # 8
		feature_name = orig_data.columns[feature_count]
		feature_orders_actions = actions[feature_count*action_per_feature: (feature_count+1)*action_per_feature]  # 
		transformed_feturs[feature_count] = []     # kafeng 2 demension  8 * 5

		#print('feature_orders_actions = ', feature_orders_actions)
		if feature_orders_actions[0] == 0:
			continue
		else:
			#print('feature_name = ', feature_name)
			fetr = np.array(orig_data[feature_name].values)  # 0-7 columns 

		for action in feature_orders_actions:  # 5
			if action == 0:  # EOF 1 # kafeng no transform max_orders, stop
				break
			elif action > 0 and action <= args.num_op_unary:   # 0-4 = 5 types
				# unary
				action_unary = action - 1
				if action_unary == 0:
					fetr = np.squeeze(np.sqrt(abs(fetr)))
				elif action_unary == 1:
					scaler = MinMaxScaler()
					fetr = np.squeeze(scaler.fit_transform(np.reshape(fetr,[-1,1])))
				elif action_unary == 2:
					while (np.any(fetr == 0)):
						fetr = fetr + 1e-5
					fetr = np.squeeze(np.log(abs(np.array(fetr)))) 
				elif action_unary == 3:
					while (np.any(fetr == 0)):
						fetr = fetr + 1e-5
					fetr = np.squeeze(1 / (np.array(fetr))) 
			else:   # 5-39 = 35 types    =5*7 
				# binary
				action_binary = (action-args.num_op_unary-1) // (num_feature-1)
				#print('action_binary = ', action_binary)
				rank = np.mod(action-args.num_op_unary-1, num_feature-1)
				#print('rank = ', rank)

				if rank >= feature_count:  #  kafeng ????
					rank += 1
				target_feature_name = orig_data.columns[rank]
				#print('target_feature_name = ', target_feature_name)
				target = np.array(orig_data[target_feature_name].values)

				if action_binary == 0:
					fetr = np.squeeze(fetr + target)  # 0-8  ,  actions decode column
				elif action_binary == 1:
					fetr = np.squeeze(fetr - target)
				elif action_binary == 2:
					fetr = np.squeeze(fetr * target)
				elif action_binary == 3:
					while (np.any(target == 0)):
						target = target + 1e-5
					fetr = np.squeeze(fetr / target)  
				elif action_binary == 4:
					fetr = np.squeeze(mod_column(fetr, orig_data[target_feature_name].values))  # kafeng this will generate all 0 column ?

			transformed_feturs[feature_count].append(fetr)  # append 5 ,  1-5 transform orders 
		max_order_fetrs.append(fetr)  # append 8 , use the last action , just for test ????  no thing about train ???
	#print(transformed_feturs)

	return transformed_feturs, max_order_fetrs

def get_reword_train(actions):
	action_per_feature = int(len(actions) / num_feature)   # 40/8 = 5
	rewards = []

	#start_time = time.time()

	#print('get_reword_train actions = ', actions)	
	if args.controller == 'attention':
		transformed_feturs, max_order_fetrs = transformation_search_space_attention(actions)
	else:
		#transformed_feturs, max_order_fetrs = transformation_search_space(actions)  # minhash empty seqence.
		transformed_feturs, max_order_fetrs = transformation_search_space_attention(actions)

	# add for minhash
	if args.minhash:
		norm_data = pd.DataFrame()
		count = 0
		for i in range(len(transformed_feturs)):  #
			for j in range(len(transformed_feturs[i])):
				norm_new = (transformed_feturs[i][j]-transformed_feturs[i][j].min())/(transformed_feturs[i][j].max()-transformed_feturs[i][j].min())
				norm_data.insert(count, '%d'%(count), norm_new)  # title is str
				count = count + 1

		#print(norm_data)
		#print('norm_data.shape = ', norm_data.shape)
		weighted_set = norm_data.values
		wmh = WeightedMinHash(weighted_set, args.dimension_pcws, seed=0)
		if args.feature_extract_alg == 'PCWS':
			k, y, e = wmh.pcws()
		elif args.feature_extract_alg == 'ICWS':
			k, y, e = wmh.icws()
		elif args.feature_extract_alg == 'LICWS':
			k, e = wmh.licws()
		elif args.feature_extract_alg == 'CCWS':
			device = 'cpu'
			if device == 'cpu':
				k, y, e = wmh.ccws_pytorch()
				k = k.numpy()
			elif device == 'cuda':
				k, y, e = wmh.ccws_gpu()
				k = k.cpu().numpy()
			#else:
			#	k, y, e = wmh.ccws()
		indexs = np.transpose(k.astype(np.int32))
		#print('indexs = ', indexs)
		pcws = pd.DataFrame()
		for col in range(norm_data.shape[1]):
			indexs_values = norm_data['%d'%(col)][indexs[:,col]].reset_index(drop=True)
			pcws = pd.concat([pcws, indexs_values], axis=1, ignore_index=True )

		test_x = pd.DataFrame(np.transpose(pcws.values))  # = meta_features
		pre_prob = opengl_rf.predict_proba(test_x)
		#print('pre_prob = ', pre_prob)
		# paser prob
		probs = []
		for i in range(len(transformed_feturs)):
			probs.append(pre_prob[i : i+ len(transformed_feturs[i])])

		former_result = origin_result
		former_max_order_fetrs = []  # original  # delete None  ValueError: Length of values does not match length of index ??

		#print('transformed_feturs.keys() = ', transformed_feturs.keys())  # continue
		for key in sorted(transformed_feturs.keys()):   # 0-7   len(transformed_feturs) = 8   # transformed_feturs is 2 demension  8 * 5  # 8 
			#reward, former_result, return_fetr = get_reward_per_feature(transformed_feturs[key], action_per_feature, former_result, former_max_order_fetrs)  # 5
			reward, former_result, return_fetr = get_reward_per_feature(transformed_feturs[key], action_per_feature, former_result, former_max_order_fetrs, probs[key])
			former_max_order_fetrs.append(return_fetr)   # kafeng  append the last columns of transformed_feturs[key] .  max_order 
			rewards += reward
		#print("rewards: ", rewards)
	else:
		former_result = origin_result
		former_max_order_fetrs = []  # original  # delete None  ValueError: Length of values does not match length of index ??

		#print('transformed_feturs.keys() = ', transformed_feturs.keys())  # continue
		for key in sorted(transformed_feturs.keys()):   # 0-7   len(transformed_feturs) = 8   # transformed_feturs is 2 demension  8 * 5  # 8 
			reward, former_result, return_fetr = get_reward_per_feature(transformed_feturs[key], action_per_feature, former_result, former_max_order_fetrs, 0)  # 5
			former_max_order_fetrs.append(return_fetr)   # kafeng  append the last columns of transformed_feturs[key] .  max_order 
			rewards += reward
		#print("rewards: ", rewards)

	#duration = time.time() - start_time
	#print('get_reword_train %s  duration = %.5f seconds' %(datetime.datetime.now(), duration))

	constructed_features_list.clear()  # kafeng add
	constructed_trees_list.clear()
	selected_features_list.clear()
	selected_fscore_list.clear()
	return rewards

'''
def get_reword_train(actions):
	action_per_feature = int(len(actions) / num_feature)   # 40/8 = 5
	rewards = []

	#print('get_reword_train actions = ', actions)
	transformed_feturs, max_order_fetrs = transformation_search_space(actions)

	former_result = origin_result
	former_max_order_fetrs = []  # original  # delete None  ValueError: Length of values does not match length of index ??

	for key in sorted(transformed_feturs.keys()):   # 0-7   len(transformed_feturs) = 8   # transformed_feturs is 2 demension  8 * 5  # 8 
		reward, former_result, return_fetr = get_reward_per_feature(transformed_feturs[key], action_per_feature, former_result, former_max_order_fetrs)  # 5
		former_max_order_fetrs.append(return_fetr)   # kafeng  append the last columns of transformed_feturs[key] .  max_order 
		rewards += reward
	#print("rewards: ", rewards)

	constructed_features_list.clear()  # kafeng add
	constructed_trees_list.clear()
	selected_features_list.clear()
	selected_fscore_list.clear()
	return rewards
'''


def get_reword_test(actions):
	global constructed_features_list
	global synth_features

	X = orig_features.copy()
	
	transformed_feturs, max_order_fetrs = transformation_search_space_attention(actions)
	#print('len(max_order_fetrs) = ', len(max_order_fetrs))
	
	new_features = len(max_order_fetrs)
	norm_data = pd.DataFrame()
	for i in range(new_features):  #
		
		norm_new = (max_order_fetrs[i]-max_order_fetrs[i].min())/(max_order_fetrs[i].max()-max_order_fetrs[i].min())
		#synth_features = pd.concat([synth_features, pd.Series(norm_new)], axis=1, ignore_index=True)  # self title is num
		norm_data.insert(i, '%d'%(i), norm_new)  # title is str
		
		X.insert(len(X.columns), '%d'%(len(X.columns)+1), max_order_fetrs[i])  # this must change ??
	
	if args.minhash:
		#print(norm_data)
		#print('norm_data.shape = ', norm_data.shape)
		weighted_set = norm_data.values
		wmh = WeightedMinHash(weighted_set, args.dimension_pcws, seed=0)
		if args.feature_extract_alg == 'PCWS':
			k, y, e = wmh.pcws()
		elif args.feature_extract_alg == 'ICWS':
			k, y, e = wmh.icws()
		elif args.feature_extract_alg == 'LICWS':
			k, e = wmh.licws()
		elif args.feature_extract_alg == 'CCWS':
			#k, y, e = wmh.ccws()
			k, y, e = wmh.ccws_pytorch()
			#k, y, e = wmh.ccws_gpu()
			k = k.numpy()
		#print('e = ', e)
		indexs = np.transpose(k.astype(np.int32))
		#print('indexs = ', indexs)
		pcws = pd.DataFrame()
		for col in range(norm_data.shape[1]):
			indexs_values = norm_data['%d'%(col)][indexs[:,col]].reset_index(drop=True)
			pcws = pd.concat([pcws, indexs_values], axis=1, ignore_index=True )

		#print('pcws = ', pcws)
		meta_features = pd.DataFrame(np.transpose(pcws.values))
		test_x = meta_features
		#pre_test = paper_eval(train_x, test_x, train_y, test_y=0)
		pre_test = opengl_rf.predict(test_x)
		pre_prob = opengl_rf.predict_proba(test_x)
		
		# give a random dropout
		feature_live = np.random.uniform(0, 1, new_features)
		for i in range(new_features):
			if feature_live[i] < pre_prob[i][1]:  # negtive feature
				X = X.drop('%d'%(len(orig_features.columns)+1+i), axis=1)  # from orig col baseline 

	
	args.evaluate = 'f_score'
	result = evaluate(X, target_label, args)
	
	return result

def get_reward_per_feature(transformed_feturs, action_per_feature, former_result, former_max_order_fetrs, probs):
	X = orig_features.copy()

	reward = []
	previous_result = former_result

	for i, former_fetr in enumerate(former_max_order_fetrs):     # old transform features
		if former_fetr != []:  # prevent former []
			X.insert(len(X.columns), '%d'%(len(X.columns)+1), former_fetr)  # 

	if args.minhash:
		# give a random dropout
		feature_live = np.random.uniform(0, 1, len(transformed_feturs))

		i = 0
		for fetr in transformed_feturs:  #  5   new transform features diffent orders
			#if random.random() > 0.5:
			if feature_live[i] > probs[i][1]:  # negtive feature
				X.insert(len(X.columns), '%d'%(len(X.columns)+1), fetr)  # 

				current_result = evaluate(X, target_label, args)

				reward.append(current_result - previous_result) 
				previous_result = current_result
				del X['%d'%len(X.columns)]  #  X.shape[1]  #  delete use other order
			i = i+1
	else:
		for fetr in transformed_feturs:  #  5   new transform features diffent orders
			X.insert(len(X.columns), '%d'%(len(X.columns)+1), fetr)  # 

			current_result = evaluate(X, target_label, args)

			reward.append(current_result - previous_result) 
			previous_result = current_result
			del X['%d'%len(X.columns)]  #  X.shape[1]  #  delete use other order

	for _ in range(action_per_feature - len(reward)):   # kafeng padded to 5
		reward.append(0)

	if len(transformed_feturs) == 0:
		return_fetr = []    # no feature be transformed
	else:
		return_fetr = transformed_feturs[-1]  # use the highest order transform, reserved
	return reward, previous_result, return_fetr

'''
def get_reward_per_feature(transformed_feturs, action_per_feature, former_result, former_max_order_fetrs):	# kafeng just incremantal features  # delete None
	X = orig_features.copy()

	#print('len(transformed_feturs) = ', len(transformed_feturs))  # 5  or < 5
	#print('transformed_feturs = ', transformed_feturs)
	#print('len(former_max_order_fetrs) = ', len(former_max_order_fetrs)) # 0-7

	reward = []
	previous_result = former_result

	for i, former_fetr in enumerate(former_max_order_fetrs):     # old transform features
		if former_fetr != []:  # prevent former []
			X.insert(len(X.columns), '%d'%(len(X.columns)+1), former_fetr)  # 

	for fetr in transformed_feturs:  #  5   new transform features diffent orders
		#print('X = ', X)
		#print('X.columns.values.tolist() = ', X.columns.values.tolist())
		X.insert(len(X.columns), '%d'%(len(X.columns)+1), fetr)  # 
		args.evaluate = 'f_score'
		current_result = evaluate(X, target_label, args)

		reward.append(current_result - previous_result) 
		previous_result = current_result
		del X['%d'%len(X.columns)]  #  X.shape[1]  #  delete use other order

	for _ in range(action_per_feature - len(reward)):   # kafeng padded to 5
		reward.append(0)

	if len(transformed_feturs) == 0:
		return_fetr = []    # no feature be transformed
	else:
		return_fetr = transformed_feturs[-1]  # use the highest order transform, reserved
	return reward, previous_result, return_fetr
'''

def train_bms(model, l=None, p=None):
	global origin_result  
	
	args.evaluate = 'f_score'
	origin_result = evaluate(orig_features, target_label, args)

	best_result = origin_result
	print('origin_result  = ', origin_result)

	all_best_result = best_result
	
	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True
	with tf.Session(config=config) as sess:
		nfs_best_result = tf.Variable(best_result, name="best_result")
		our_best_result = tf.Variable(all_best_result, name="best_result")
		tf.summary.scalar(ds + "_bms_nfs_best_result", nfs_best_result)
		tf.summary.scalar(ds + "_bms_all_best_result", our_best_result)
		summary_merge_op = tf.summary.merge_all()

		init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
		log_fold = './' + args.log_dir + '/' + args.controller + '_' + str(args.minhash) + '_' + args.cache_method
		if not os.path.exists(log_fold):
			os.makedirs(log_fold)
		writer = tf.summary.FileWriter(log_fold, sess.graph)
		sess.run(init_op)
		
		model_result = -10000.0
		train_set, values = [], []  # for AC

		for epoch_count in range(args.epochs):
			concat_action = []
			probs_action = sess.run(tf.nn.softmax(model.concat_output))  # kafeng probs_action connect RL controller and features transfrom
			
			# kafeng add for random sample
			
			for i in range(args.num_batch):  # 5
				batch_action = []
				for _ in range(model.num_action):  # 8 * 5
					batch_action.append(np.random.randint(model.num_op))  # 4 + (8-1)*5  + 1 = 40
				concat_action.append(batch_action)
			#print('random concat_action = ', concat_action)

			# get rewards
			if args.multiprocessing:
				pool = Pool(num_process)
				rewards = np.array(pool.map(get_reword_train, concat_action))  # kafeng parallel actions
				pool.close()
				pool.join()
			else:
				rewards = []
				for action in concat_action:   # kafeng series actions
					rewards.append(get_reword_train(action))    # 
				rewards = np.array(rewards)

			if args.multiprocessing:
				pool = Pool(num_process)
				results = pool.map(get_reword_test, concat_action)
				pool.close()
				pool.join()
			else:
				results = []
				for action in concat_action:
					results.append(get_reword_test(action))         # 
			model_result = max(model_result, max(results))


			# test
			# probs_action = sess.run(tf.nn.softmax(model.concat_output))
			probs_action = torch.nn.Softmax(model.concat_output)
			best_action = probs_action.argmax(axis=1)       # kafeng  ?????   rnn every time must  max order ??? low order max F1 ??
			#print('len(best_action) = ', len(best_action))
			model_result = max(model_result, get_reword_test(best_action))

			# update best_result
			best_result = max(best_result, model_result)  # this is not the best ... no middle result 
			#print('best_result = ', best_result)

			#print('all_f1_score = ', all_f1_score)
			all_best_result = max(all_best_result, max(all_f1_score))
			all_f1_score[:] = []  # ListProxy
			#print('all_best_result = ', all_best_result)

			#if (epoch_count+1) % 20 == 0:
			if (epoch_count+1) % 1 == 0:
				print('Epoch %d:  origin_result = %.4f,  \n model_result = %.4f, best_action = %s, \n best_result = %.4f' 
					% (epoch_count+1,  origin_result,   model_result, str(best_action), best_result))
				print('all_best_result = ', all_best_result)

			
			# Write logs at every iteration
			# sess.run(tf.assign(nfs_best_result, best_result))
			# sess.run(tf.assign(our_best_result, all_best_result))
			# summary_info = sess.run(summary_merge_op)
			# writer.add_summary(summary_info, epoch_count+1)
			# writer.flush()
			

def train_tran(model, l=None, p=None):
	global origin_result  
	
	args.evaluate = 'f_score'
	origin_result = evaluate(orig_features, target_label, args)

	best_result = origin_result
	print('origin_result  = ', origin_result)

	all_best_result = best_result
	

	# init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
	# log_fold = './' + args.log_dir + '/' + args.controller + '_' + str(args.minhash) + '_' + args.cache_method
	# if not os.path.exists(log_fold):
	# 	os.makedirs(log_fold)
	# writer = tf.summary.FileWriter(log_fold, sess.graph)
	# sess.run(init_op)
	
	model_result = -10000.0
	train_set, values = [], []  # for AC

	LR = 0.001               
	optimizer = torch.optim.Adam(model.parameters(), lr=LR)    # kafeng   parameters()    must    nn.Module
	# 选择计算误差的方法
	# loss_func = nn.CrossEntropyLoss()

	for epoch_count in range(args.epochs):
		concat_action = []
		# probs_action = sess.run(tf.nn.softmax(model.concat_output))  # kafeng probs_action connect RL controller and features transfrom
		# print('train model.concat_output = ', model.concat_output)   
		print('train model.concat_output.shape = ', model.concat_output.shape)  #  [40 40] 
		probs_action =  torch.nn.functional.softmax(model.concat_output, dim=1)
		# print('probs_action = ', probs_action)
		
		# kafeng add for random sample
		
		for i in range(args.num_batch):  # 5
			batch_action = []
			for _ in range(model.num_action):  # 8 * 5
				batch_action.append(np.random.randint(model.num_op))  # 4 + (8-1)*5  + 1 = 40
			concat_action.append(batch_action)
		#print('random concat_action = ', concat_action)

		# get rewards
		if args.multiprocessing:
			pool = Pool(num_process)
			rewards = np.array(pool.map(get_reword_train, concat_action))  # kafeng parallel actions
			pool.close()
			pool.join()
		else:
			rewards = []
			for action in concat_action:   # kafeng series actions
				rewards.append(get_reword_train(action))    # 
			rewards = np.array(rewards)

		if args.multiprocessing:
			pool = Pool(num_process)
			results = pool.map(get_reword_test, concat_action)
			pool.close()
			pool.join()
		else:
			results = []
			for action in concat_action:
				results.append(get_reword_test(action))         # 
		model_result = max(model_result, max(results))

		# kafeng this concat_action is not sample by the controller ?  just ramdom sample ???
		# update policy params
		# feed_dict = {model.concat_action: np.reshape(concat_action, [args.num_batch,-1]), model.rewards: np.reshape(rewards,[args.num_batch,-1])}
		# loss_epoch = model.update_policy(feed_dict, sess)   # kafeng train controller ..
		#  concat_action is list
		loss = model(torch.tensor(concat_action), torch.tensor(rewards))

		# loss = loss_func(output, b_y)
		optimizer.zero_grad() 
		loss = loss.requires_grad_()
		loss.backward() 
		optimizer.step()

		# test
		# probs_action = sess.run(tf.nn.softmax(model.concat_output))
		print('test  model.concat_output = ', model.concat_output)   # kafeng why  < 0 ????
		print('test  model.concat_output.shape = ', model.concat_output.shape)
		probs_action =  torch.nn.functional.softmax(model.concat_output, dim=1)
		# print('probs_action = ', probs_action)
		# best_action = probs_action.argmax(axis=1)       # kafeng  ?????   rnn every time must  max order ??? low order max F1 ??
		best_action = torch.argmax(probs_action, dim=1)
		#print('len(best_action) = ', len(best_action))
		model_result = max(model_result, get_reword_test(best_action))

		# update best_result
		best_result = max(best_result, model_result)  # this is not the best ... no middle result 
		#print('best_result = ', best_result)

		#print('all_f1_score = ', all_f1_score)
		all_best_result = max(all_best_result, max(all_f1_score))
		all_f1_score[:] = []  # ListProxy
		#print('all_best_result = ', all_best_result)

		#if (epoch_count+1) % 20 == 0:
		if (epoch_count+1) % 1 == 0:
			print('Epoch %d:  origin_result = %.4f,  \n model_result = %.4f, best_action = %s, \n best_result = %.4f' 
				% (epoch_count+1,  origin_result,   model_result, str(best_action), best_result))
			print('all_best_result = ', all_best_result)


def train_trajectory(model, l=None, p=None):
	global origin_result  
	
	args.evaluate = 'f_score'
	origin_result = evaluate(orig_features, target_label, args)

	best_result = origin_result
	print('origin_result  = ', origin_result)

	all_best_result = best_result
	

	model_result = -10000.0

	LR = 0.001               
	optimizer = torch.optim.Adam(model.parameters(), lr=LR)    # kafeng   parameters()    must    nn.Module
	# 选择计算误差的方法
	# loss_func = nn.CrossEntropyLoss()

	for epoch_count in range(args.epochs):
		concat_action = []
		# probs_action = sess.run(tf.nn.softmax(model.concat_output))  # kafeng probs_action connect RL controller and features transfrom
		# print('train model.concat_output = ', model.concat_output)   
		print('train model.concat_output.shape = ', model.concat_output.shape)  # [40, 40]
		probs_action =  torch.nn.functional.softmax(model.concat_output, dim=1)
		# print('probs_action = ', probs_action)
		
		# kafeng add for random sample
		
		for i in range(args.num_batch):  # 32
			batch_action = []
			for _ in range(model.num_action):  # 8 * 5 = 40
				batch_action.append(np.random.randint(model.num_op))  # 4 + (8-1)*5  + 1 = 40
			concat_action.append(batch_action)
		#print('random concat_action = ', concat_action)
		print('random len(concat_action) =  ', len(concat_action))   # 32
		print('random len(concat_action[0]) =  ', len(concat_action[0]))  # 40

		# get rewards
		if args.multiprocessing:
			pool = Pool(num_process)
			rewards = np.array(pool.map(get_reword_train, concat_action))  # kafeng parallel actions
			pool.close()
			pool.join()
		else:
			rewards = []
			for action in concat_action:   # kafeng series actions
				rewards.append(get_reword_train(action))    # 
			rewards = np.array(rewards)

		if args.multiprocessing:
			pool = Pool(num_process)
			results = pool.map(get_reword_test, concat_action)
			pool.close()
			pool.join()
		else:
			results = []
			for action in concat_action:
				results.append(get_reword_test(action))         # 
		model_result = max(model_result, max(results))

		# kafeng this concat_action is not sample by the controller ?  just ramdom sample ???
		# update policy params
		# feed_dict = {model.concat_action: np.reshape(concat_action, [args.num_batch,-1]), model.rewards: np.reshape(rewards,[args.num_batch,-1])}
		# loss_epoch = model.update_policy(feed_dict, sess)   # kafeng train controller ..
		#  concat_action is list
		loss = model(torch.tensor(concat_action), torch.tensor(rewards))

		# loss = loss_func(output, b_y)
		optimizer.zero_grad() 
		loss = loss.requires_grad_()
		loss.backward() 
		optimizer.step()

		# test
		# probs_action = sess.run(tf.nn.softmax(model.concat_output))
		print('test  model.concat_output = ', model.concat_output)   # kafeng why  < 0 ????       #    states
		print('test  model.concat_output.shape = ', model.concat_output.shape)
		probs_action =  torch.nn.functional.softmax(model.concat_output, dim=1)         #  actions
		# print('probs_action = ', probs_action)
		# best_action = probs_action.argmax(axis=1)       # kafeng  ?????   rnn every time must  max order ??? low order max F1 ??
		best_action = torch.argmax(probs_action, dim=1)
		#print('len(best_action) = ', len(best_action))
		model_result = max(model_result, get_reword_test(best_action))   # rewards  ,  return_to_go..             timesteps ???

		# update best_result
		best_result = max(best_result, model_result)  # this is not the best ... no middle result 
		#print('best_result = ', best_result)

		#print('all_f1_score = ', all_f1_score)
		all_best_result = max(all_best_result, max(all_f1_score))
		all_f1_score[:] = []  # ListProxy
		#print('all_best_result = ', all_best_result)

		#if (epoch_count+1) % 20 == 0:
		if (epoch_count+1) % 1 == 0:
			print('Epoch %d:  origin_result = %.4f,  \n model_result = %.4f, best_action = %s, \n best_result = %.4f' 
				% (epoch_count+1,  origin_result,   model_result, str(best_action), best_result))
			print('all_best_result = ', all_best_result)



def evaluate_episode_rtg(
        # env,
        state_dim,
        act_dim,
        model,
        # max_ep_len=1000,
		max_ep_len=40,  #  kafeng  operators
        scale=1000.,
        state_mean=0.,
        state_std=1.,
        device='cuda',
        target_return=None,
        mode='normal',
    ):

    model.eval()
    model.to(device=device)

    # state_mean = torch.from_numpy(state_mean).to(device=device)
    # state_std = torch.from_numpy(state_std).to(device=device)

    # state = env.reset()
    if mode == 'noise':
        state = state + np.random.normal(0, 0.1, size=state.shape)

    # we keep all the histories on the device
    # note that the latest action and reward will be "padding"
    states = torch.from_numpy(state).reshape(1, state_dim).to(device=device, dtype=torch.float32)
    actions = torch.zeros((0, act_dim), device=device, dtype=torch.float32)
    rewards = torch.zeros(0, device=device, dtype=torch.float32)

    ep_return = target_return
    target_return = torch.tensor(ep_return, device=device, dtype=torch.float32).reshape(1, 1)
    timesteps = torch.tensor(0, device=device, dtype=torch.long).reshape(1, 1)

    sim_states = []

    episode_return, episode_length = 0, 0
    for t in range(max_ep_len):

        # add padding
        actions = torch.cat([actions, torch.zeros((1, act_dim), device=device)], dim=0)
        rewards = torch.cat([rewards, torch.zeros(1, device=device)])

        action = model.get_action(
            (states.to(dtype=torch.float32) - state_mean) / state_std,
            actions.to(dtype=torch.float32),
            rewards.to(dtype=torch.float32),
            target_return.to(dtype=torch.float32),
            timesteps.to(dtype=torch.long),
        )
        actions[-1] = action
        action = action.detach().cpu().numpy()

        state, reward, done, _ = env.step(action)

        cur_state = torch.from_numpy(state).to(device=device).reshape(1, state_dim)
        states = torch.cat([states, cur_state], dim=0)
        rewards[-1] = reward

        if mode != 'delayed':
            pred_return = target_return[0,-1] - (reward/scale)
        else:
            pred_return = target_return[0,-1]
        target_return = torch.cat(
            [target_return, pred_return.reshape(1, 1)], dim=1)
        timesteps = torch.cat(
            [timesteps,
             torch.ones((1, 1), device=device, dtype=torch.long) * (t+1)], dim=1)

        episode_return += reward
        episode_length += 1

        if done:
            break

    return episode_return, episode_length


def eval_episodes(target_rew):
		def fn(model):
			returns, lengths = [], []
			for _ in range(100):   # num_eval_episodes
				with torch.no_grad():
					ret, length = evaluate_episode_rtg(
						# env, 
						40, 40, model,
						max_ep_len=40, scale=100, target_return=target_rew/100,
						# mode=mode, state_mean=state_mean, state_std=state_std, device=device,
					)
				returns.append(ret)
				lengths.append(length)
			return {
                f'target_{target_rew}_return_mean': np.mean(returns),
                f'target_{target_rew}_return_std': np.std(returns),
                f'target_{target_rew}_length_mean': np.mean(lengths),
                f'target_{target_rew}_length_std': np.std(lengths),
            }
		return fn



def train_dt(model, l=None, p=None):   # kafeng modify from  DT  trainer.py/Trainer

	global origin_result
	
	args.evaluate = 'f_score'
	origin_result = evaluate(orig_features, target_label, args)

	best_result = origin_result
	print('origin_result  = ', origin_result)

	all_best_result = best_result

	model_result = -10000.0

	optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4,)
	scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda steps: min((steps+1)/10000, 1))
	batch_size = 32
	# loss_fn = lambda s_hat, a_hat, r_hat, s, a, r: torch.mean((a_hat - a)**2)   # kafeng from experiment.py
	loss_fn = lambda  a_hat,   a : torch.mean((a_hat - a)**2)  


	# if env_name == 'hopper':
		# env = gym.make('Hopper-v3')
		# max_ep_len = 1000
		# env_targets = [3600, 1800]  # evaluation conditioning targets
		# scale = 1000.  # normalization for rewards/returns
	# max_ep_len = 5*8
	# env_targets = [5, 8]  # evaluation conditioning tar

	# eval_fns = [eval_episodes(tar) for tar in env_targets]          #   num_eval_episodes   100   

	train_losses = []

	for epoch_count in range(args.epochs):
		concat_action = []
		# probs_action = sess.run(tf.nn.softmax(model.concat_output))  # kafeng probs_action connect RL controller and features transfrom
		# print('train model.concat_output = ', model.concat_output)   
		print('train model.concat_output.shape = ', model.concat_output.shape)  #  [40, 40]
		probs_action =  torch.nn.functional.softmax(model.concat_output, dim=1)
		# print('probs_action = ', probs_action)
		print('probs_action.shape = ', probs_action.shape)
		

		for i in range(args.num_batch):  # 32
			batch_action = []
			for _ in range(model.num_action):  # 8 * 5 = 40
				batch_action.append(np.random.randint(model.num_op))  # 4 + (8-1)*5  + 1 = 40
			concat_action.append(batch_action)
		#print('random concat_action = ', concat_action)
		print('random len(concat_action) =  ', len(concat_action))   # 32
		print('random len(concat_action[0]) =  ', len(concat_action[0]))  # 40

		# get rewards
		if args.multiprocessing:
			pool = Pool(num_process)
			rewards = np.array(pool.map(get_reword_train, concat_action))  # kafeng parallel actions
			pool.close()
			pool.join()
		else:
			rewards = []
			for action in concat_action:   # kafeng series actions
				rewards.append(get_reword_train(action))    # 
			rewards = np.array(rewards)

		if args.multiprocessing:
			pool = Pool(num_process)
			results = pool.map(get_reword_test, concat_action)
			pool.close()
			pool.join()
		else:
			results = []
			for action in concat_action:
				results.append(get_reword_test(action))         # 
		model_result = max(model_result, max(results))



		# for DT
		model.train()
		
		# train_loss = train_step()  # kafeng  from decision_transformer/training/trainer.py

		# states, actions, rewards, dones, attention_mask, returns, mask = get_batch(batch_size=32, max_len=5)
		# states, actions, rewards, dones, timesteps, returns, mask = get_batch(batch_size=40, max_len=5)
		# states = torch.rand(40, 5, 8)  
		states = probs_action
		print('states = ', states)
		print('states.shape = ', states.shape)
		# actions = torch.rand(40, 5, 8)
		# actions = torch.cat(concat_action)
		actions =  torch.tensor(concat_action, dtype=torch.float32)
		print('actions = ', actions)
		print('actions.shape = ', actions.shape)  # [32, 40]
		# rewards = torch.rand(40, 5, 1)
		rewards = torch.tensor(results)
		print('rewards = ', rewards)
		print('rewards.shape = ', rewards.shape)
		# dones = torch.rand(40, 5, )
		returns = torch.rand(40, 1, 1)  # rtg
		# timesteps =  torch.randint(1, 40, (40, 5))
		timesteps =  torch.randint(1, 40, (40, 1))
		# mask = torch.rand(40, 5)
		
		state_target, action_target, reward_target = torch.clone(states), torch.clone(actions), torch.clone(rewards)

		state_preds, action_preds, reward_preds = model.forward(
			# states, actions, rewards, masks=None, attention_mask=attention_mask, target_return=returns,
			states, actions, rewards, returns, timesteps,
		)
		# print('state_preds.shape = ', state_preds.shape)
		# print('action_preds.shape = ', action_preds.shape)   # [40, 40, 32]
		# print('reward_preds.shape = ', reward_preds.shape)


		print('-- state_target.shape = ', state_target.shape)
		# print('-- state_target[:,1:].shape = ', state_target[:,1:].shape)
		print('-- action_target.shape = ', action_target.shape)   #  [32, 40]
		# print('action_target[:,1:].shape = ', action_target[:,1:].shape)
		# action_target_ex = action_target.repeat(1, 8, 5)   #  40/5,  40/8
		action_target_ex = action_target.permute(1, 0).repeat(40, 1, 1)
		print('-- action_target_ex.shape = ', action_target_ex.shape)
		print('-- reward_target.shape = ', reward_target.shape)
		# print('-- reward_target[:,1:].shape = ', reward_target[:,1:].shape)

		# note: currently indexing & masking is not fully correct
		# loss = loss_fn(
		# 	state_preds, action_preds, reward_preds,
		# 	# state_target[:,1:], action_target, reward_target[:,1:],   # kafeng modify 
		# 	state_target[:,1:], action_target_ex, reward_target[:,1:],
		# )
		loss = loss_fn(action_preds, action_target_ex)
		optimizer.zero_grad()
		loss.backward()   # kafeng why run epoch > 1 error ??  retain_graph
		optimizer.step()

		train_loss = loss.detach().cpu().item()

		train_losses.append(train_loss)
		if scheduler is not None:
			scheduler.step()


		model.eval()
		# eval_episodes()
		# for eval_fn in eval_fns:
			# outputs = eval_fn(model)
			# for k, v in outputs.items():
			# 	logs[f'evaluation/{k}'] = v

		
		# test
		# probs_action = sess.run(tf.nn.softmax(model.concat_output))
		# print('test  model.concat_output = ', model.concat_output)   # kafeng why  < 0 ????       #    states     train model no change ????
		# print('test  model.concat_output.shape = ', model.concat_output.shape)
		# probs_action =  torch.nn.functional.softmax(model.concat_output, dim=1)         #  actions
		probs_action =  torch.nn.functional.softmax(action_preds[:, :, 0], dim=1)    # action_preds  -> model.concat_output
		# print('probs_action = ', probs_action)
		# best_action = probs_action.argmax(axis=1)       # kafeng  ?????   rnn every time must  max order ??? low order max F1 ??
		best_action = torch.argmax(probs_action, dim=1)
		#print('len(best_action) = ', len(best_action))
		model_result = max(model_result, get_reword_test(best_action))   # rewards  ,  return_to_go..             timesteps ???


		# update best_result
		best_result = max(best_result, model_result)  # this is not the best ... no middle result 
		#print('best_result = ', best_result)

		#print('all_f1_score = ', all_f1_score)
		all_best_result = max(all_best_result, max(all_f1_score))
		all_f1_score[:] = []  # ListProxy
		#print('all_best_result = ', all_best_result)

		#if (epoch_count+1) % 20 == 0:
		if (epoch_count+1) % 1 == 0:
			print('Epoch %d:  origin_result = %.4f,  \n model_result = %.4f, best_action = %s, \n best_result = %.4f' 
				% (epoch_count+1,  origin_result,   model_result, str(best_action), best_result))
			print('all_best_result = ', all_best_result)
		


def train(model, l=None, p=None):
	global origin_result  
	
	args.evaluate = 'f_score'
	origin_result = evaluate(orig_features, target_label, args)

	best_result = origin_result
	print('origin_result  = ', origin_result)

	all_best_result = best_result
	#return  # for less log from cpython

	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True
	with tf.Session(config=config) as sess:
		nfs_best_result = tf.Variable(best_result, name="best_result")
		our_best_result = tf.Variable(all_best_result, name="best_result")
		tf.summary.scalar(ds + "_rnn_nfs_best_result", nfs_best_result)
		tf.summary.scalar(ds + "_rnn_all_best_result", our_best_result)
		summary_merge_op = tf.summary.merge_all()

		init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
		log_fold = './' + args.log_dir + '/' + args.controller + '_' + str(args.minhash) + '_' + args.cache_method
		if not os.path.exists(log_fold):
			os.makedirs(log_fold)
		writer = tf.summary.FileWriter(log_fold, sess.graph)
		sess.run(init_op)

		
		model_result = -10000.0
		train_set, values = [], []  # for AC

		for epoch_count in range(args.epochs):
			#utils_sklearn.all_entropy_mean, utils_sklearn.all_f1_score = [], []
			concat_action = []
			probs_action = sess.run(tf.nn.softmax(model.concat_output))  # kafeng probs_action connect RL controller and features transfrom
			#print('probs_action = ', probs_action)

			# sample actions           # kafeng how the controller have some thing to the sample actions ??
			for batch_count in range(args.num_batch):  # 32
				batch_action = []
				for i in range(probs_action.shape[0]):  # 40
					sample_action = np.random.choice(len(probs_action[i]), p=probs_action[i])  # 0-39  probs_action update epochs
					batch_action.append(sample_action)
				#print('batch_action = ', batch_action)
				concat_action.append(batch_action)
			#print('concat_action = ', concat_action)

			# get rewards
			if args.multiprocessing:
				pool = Pool(num_process)
				rewards = np.array(pool.map(get_reword_train, concat_action))  # kafeng parallel actions
				pool.close()
				pool.join()
			else:
				rewards = []
				for action in concat_action:   # kafeng series actions
					rewards.append(get_reword_train(action))    # 
				rewards = np.array(rewards)

			if args.multiprocessing:
				pool = Pool(num_process)
				results = pool.map(get_reword_test, concat_action)
				pool.close()
				pool.join()
			else:
				results = []
				for action in concat_action:
					results.append(get_reword_test(action))         # 
			model_result = max(model_result, max(results))

			if args.RL_model == 'AC':
				# using actor critic
				target_set = []
				for batch_count in range(args.num_batch):
					action = concat_action[batch_count]
					for i in range(model.num_action):
						train_tmp = list(np.zeros(model.num_action, dtype=int))
						target_tmp = list(np.zeros(model.num_action, dtype=int))
						
						train_tmp[0:i] = list(action[0:i])
						target_tmp[0:i+1] = list(action[0:i+1])

						train_set.append(train_tmp)
						target_set.append(target_tmp)

				state = np.reshape(train_set, [-1,model.num_action])
				next_state = np.reshape(target_set, [-1,model.num_action])

				value = model.predict_value(next_state) * args.alpha + rewards.flatten()
				values += list(value)
				model.update_value(state, values)

				# compute estimate reward
				rewards_predict = model.predict_value(next_state) * args.alpha - \
					model.predict_value(state[-np.shape(next_state)[0]:]) + rewards.flatten()
				rewards = np.reshape(rewards_predict, [args.num_batch,-1])

			elif args.RL_model == 'PG':
				for i in range(model.num_action):   # 40   kafeng lamda return ????
					base = rewards[:,i:]
					rewards_order = np.zeros_like(rewards[:,i])
					for j in range(base.shape[1]):
						order = j + 1
						base_order = base[:,0:order]
						alphas = []
						for o in range(order):
							alphas.append(pow(args.alpha, o))
						base_order = np.sum(base_order*alphas, axis=1)
						base_order = base_order * np.power(args.lambd, j)
						rewards_order = rewards_order.astype(float)
						rewards_order += base_order.astype(float)  # G t k
					rewards[:,i] = (1-args.lambd) * rewards_order  # kafeng  G t lambd

			# kafeng this concat_action is not sample by the controller ?  just ramdom sample ???
			# update policy params
			feed_dict = {model.concat_action: np.reshape(concat_action, [args.num_batch,-1]), model.rewards: np.reshape(rewards,[args.num_batch,-1])}
			loss_epoch = model.update_policy(feed_dict, sess)   # kafeng train controller ..

			# test
			probs_action = sess.run(tf.nn.softmax(model.concat_output))
			best_action = probs_action.argmax(axis=1)
			#print('len(best_action) = ', len(best_action))
			model_result = max(model_result, get_reword_test(best_action))

			# update best_result
			best_result = max(best_result, model_result)
			#print('best_result = ', best_result)

			# kafeng add
			all_best_result = max(all_best_result, max(all_f1_score))
			all_f1_score[:] = []  # ListProxy
			#print('all_best_result = ', all_best_result)

			
			#if (epoch_count+1) % 20 == 0:
			if (epoch_count+1) % 1 == 0:
				print('Epoch %d: loss = %.4f, origin_result = %.4f, lr = %.3f, \n model_result = %.4f, best_action = %s, \n best_result = %.4f' 
					% (epoch_count+1, loss_epoch, origin_result, args.lr, model_result, str(best_action), best_result))
				print('all_best_result = ', all_best_result)


			# Write logs at every iteration
			sess.run(tf.assign(nfs_best_result, best_result))
			sess.run(tf.assign(our_best_result, all_best_result))
			summary_info = sess.run(summary_merge_op)
			writer.add_summary(summary_info, epoch_count+1)
			writer.flush()
			
		

def random_run(num_random_sample, model, l=None, p=None):
	global args, num_process
	samples = []

	for i in range(num_random_sample):  # 5
		sample = []
		for _ in range(model.num_action):  # 8 * 5
			sample.append(np.random.randint(model.num_op))  # 4 + (8-1)*5  + 1 = 40
		samples.append(sample)

	if args.multiprocessing:	
		pool = Pool(num_process)
		res = list(pool.map(get_reword_test, samples))
		pool.close()
		pool.join()
	else:
		res = []
		for sample in samples:
			res.append(get_reword_test(sample))
	duration = time.time() - start_time

	random_result = max(res)  # 5 or 32   * 8
	random_sample = samples[res.index(random_result)]

	return random_result, random_sample



def train_random(model, l=None, p=None):
	global origin_result, random_max_result, random_max_samples
	
	args.evaluate = 'f_score'
	origin_result = evaluate(orig_features, target_label, args)

	best_result = origin_result
	print('origin_result  = ', origin_result)

	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True
	with tf.Session(config=config) as sess:
		init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
		sess.run(init_op)

		for epoch_count in range(args.epochs):
			concat_action = []
			probs_action = sess.run(tf.nn.softmax(model.concat_output))  # kafeng probs_action connect RL controller and features transfrom
			#print('probs_action = ', probs_action)   #  uniform 
			#print('probs_action.shape = ', probs_action.shape)
			# sample actions           # kafeng how the controller have some thing to the sample actions ??
			for batch_count in range(args.num_batch):  # 32
				batch_action = []
				for i in range(probs_action.shape[0]):  # 40
					sample_action = np.random.choice(len(probs_action[i]), p=probs_action[i])  # 0-39  probs_action update epochs
					#print('sample_action = ', sample_action)
					batch_action.append(sample_action)
				#print('batch_action = ', batch_action)
				concat_action.append(batch_action)
			#print('concat_action = ', concat_action)
			#print('len(concat_action) = ', len(concat_action))
			
			# random result
			#for batch_count in range(args.num_batch):  # 32
			# train 32 + test 33
			#args.num_random_sample = len(concat_action) + len(concat_action) + 1
			args.num_random_sample = len(concat_action) * model.num_action
			#args.num_random_sample = len(concat_action) * model.num_action * 100 # 100 epochs
			#args.num_random_sample = 100
			random_result, random_sample = random_run(args.num_random_sample, model, l, p)   
	
			# save new random_result, random_sample 
			if random_result > random_max_result:
				random_max_result = random_result
				random_max_samples = random_sample

			#if (epoch_count+1) % 20 == 0:
			print('Epoch %d: origin_result = %.4f, random_max_result = %.4f, random_max_samples = %s' 
				% (epoch_count+1, origin_result, random_max_result, str(random_max_samples)))
			

def get_meta_label(f1_results, origin_result):
	pn_results = []
	f1_changes = []

	for i in range(len(f1_results)):
		reduce_change = origin_result - f1_results[i]
		f1_changes.append(reduce_change) 
		if reduce_change >= args.threshold:
			pn_results.append(1)  # add this feature is positive 
		else:
			pn_results.append(0)  # add this feature is negtive or no use 
		
	return pn_results, f1_changes


def load_train_results():
	meta_label_all = [] 
	f1_changes_all = []

	for ds in datasets:
		path = dataPath + '.preprocess_data/' + ds + '_results.csv'
		results = pd.read_csv(path, header=0, index_col=0)
		results_list = results['results'].tolist()
		origin_result = results_list[0]
		f1_results = results_list[1:]
		pn_results, f1_changes = get_meta_label(f1_results, origin_result)
		meta_label_all = meta_label_all + pn_results
		f1_changes_all = f1_changes_all + f1_changes

	f1_change_label = pd.DataFrame({'meta_label_all': meta_label_all, 'f1_changes_all': f1_changes_all})
	negtive_samples = f1_change_label.loc[f1_change_label['meta_label_all'] == 0]
	positive_samples = f1_change_label.loc[f1_change_label['meta_label_all'] == 1]
	step = int(np.ceil(negtive_samples.shape[0] / (positive_samples.shape[0]*args.PtoN)))  # args.PtoN = 1, PtoN < step
	sorted_samples = negtive_samples.sort_values(by='f1_changes_all')
	compress_samples = sorted_samples.iloc[range(0, negtive_samples.shape[0], step)]

	f1_change_label = pd.concat([compress_samples, positive_samples], axis=0) # ignore_index=True  must save
	
	return meta_label_all, f1_changes_all, f1_change_label

def meta_feature_eval(meta_features, meta_label_all):
	args.evaluate = 'recall'
	recall_result = evaluate(meta_features, meta_label_all, args)  # recall
	print('recall_result = ', recall_result)

def train_data_process():

	#preprocess_data()  # 1481 s  in train 
	
	meta_label_all, f1_changes_all, f1_change_compress = load_train_results()
	
	norm_mean = []
	norm_var = []
	all_mean = []
	all_var = []

	all_pcws = pd.DataFrame()
	ds_count = 0
	feature_count = 0
	
	for ds in datasets:
		print('ds = ', ds)
		path = dataPath + ds + '.csv'
		orig_data = pd.read_csv(path)  #  pandas.core.frame.DataFrame

		# preprocess feature name. cloumn name  
		new_col = []
		for col in range(orig_data.shape[1]):
			new_col.append('%d'%(col))
		orig_data.columns = new_col

		# preprocess special feature, all the same
		for col in range(orig_data.shape[1]-1):
			feature = orig_data['%d'%(col)]
			if feature.max() == feature.min():
				del orig_data['%d'%(col)]

		# reset name
		new_col = []
		for col in range(orig_data.shape[1]):
			new_col.append('%d'%(col))
		orig_data.columns = new_col

		target_label = orig_data[orig_data.columns[-1]]  # pandas.core.frame.Series

		orig_features = orig_data.copy()  # 
		del orig_features[orig_features.columns[-1]]
	
		num_feature = orig_data.shape[1] - 1   # 9-1
		
		norm_data = pd.DataFrame()
		for col in range(num_feature):
			feature = orig_features['%d'%(col)]
			norm = (feature-feature.min())/(feature.max()-feature.min())
			
			norm_data.insert(col, '%d'%(col), norm)
			norm_mean.append(norm.mean())
			norm_var.append(norm.var())

		# weiwu pcws
		weighted_set = norm_data.values
		wmh = WeightedMinHash(weighted_set, args.dimension_pcws, seed=0)
		if args.feature_extract_alg == 'PCWS':
			k, y, e = wmh.pcws()
		elif args.feature_extract_alg == 'ICWS':
			k, y, e = wmh.icws()
		elif args.feature_extract_alg == 'LICWS':
			k, e = wmh.licws()
		elif args.feature_extract_alg == 'CCWS':
			k, y, e = wmh.ccws()
		indexs = np.transpose(k.astype(np.int32))
		pcws = pd.DataFrame()
		for col in range(norm_data.shape[1]):
			indexs_values = norm_data['%d'%(col)][indexs[:,col]].reset_index(drop=True)
			pcws = pd.concat([pcws, indexs_values], axis=1, ignore_index=True )

		norm_sum_mean = norm_data.mean(axis=1)

		all_mean = all_mean + [norm_sum_mean.mean()] * num_feature
		all_var = all_var + [norm_sum_mean.var()] * num_feature

		all_pcws = pd.concat([all_pcws, pcws], axis=1, ignore_index=True)

		ds_count = ds_count + 1
		
		meta_features = pd.DataFrame(np.transpose(pcws.values))
		
		feature_count = feature_count + num_feature
	
	print('ds_count = ', ds_count, ' feature_count = ', feature_count)
	meta_features = pd.DataFrame(np.transpose(all_pcws.values))  # 0.71
	
	meta_feature_eval(meta_features.iloc[f1_change_compress.index], f1_change_compress['meta_label_all'])

	return meta_features.iloc[f1_change_compress.index], f1_change_compress['meta_label_all']

if __name__ == '__main__':
	#start_time = time.time()
	device = 'cuda' if torch.cuda.is_available() else 'cpu'
	print('device =', device)

	args = parser.parse_args()
	print('args = ', args)

	openml_model = 'openml_model.txt'
	have_opengl_model = os.path.exists(openml_model)

	if have_opengl_model:
		opengl_rf = pickle.load(open(openml_model, 'rb'))
	
	if args.minhash and have_opengl_model == False:
		# for openml dataset
		dataPath = args.train_data
		fileList = os.listdir(dataPath)
		datasets = []
		for data in fileList:
			fileName = data.split('.')[0]
			if len(fileName) > 1:
				datasets.append(fileName)
		train_x, train_y = train_data_process()

		opengl_rf = RandomForestClassifier()
		opengl_rf.fit(train_x,train_y)
		#opengl_rf = autosklearn.classification.AutoSklearnClassifier(time_left_for_this_task=120)
		#opengl_rf.fit(train_x,train_y)

		pickle.dump(opengl_rf, open(openml_model, 'wb'))
	
	all_f1_score=multiprocessing.Manager().list()

	# for paper dataset
	print('args.dataset = ', args.dataset)
	for ds in args.dataset.split(","):
		start_time = time.time()
		method = None # kafeng train/test
		num_process = args.num_process
		#path = 'data/' + args.dataset + '.csv'
		# orig_data = pd.read_csv(path) 
		print('ds = ', ds)
		# path = 'paperData/.preprocess_data/' + ds + '.csv'
		path = '../paperData/.preprocess_data/' + ds + '.csv'
		orig_data = pd.read_csv(path, header=0, index_col=0) #  pandas.core.frame.DataFrame
		target_label = orig_data[orig_data.columns[-1]]  # pandas.core.frame.Series
		orig_features = orig_data.copy()  #
		del orig_features[orig_features.columns[-1]]

		num_feature = orig_data.shape[1] - 1   # 9-1
		mum_columns = orig_data.shape[1]  # 9

		# norm X
		norm_orig = pd.DataFrame()
		for col in range(num_feature):
			feature = orig_features['%d'%(col)]
			norm = (feature-feature.min())/(feature.max()-feature.min())
			norm_orig.insert(col, '%d'%(col), norm)
		

		# tf.reset_default_graph()
		all_f1_score[:] = []
		if args.controller == 'bms':
			controller = Controller_bms(args, num_feature)  # just use para
			controller.build_graph()
			bms_max_result = 0.0 # save result
			bms_max_samples = []  # save samples 
			train_bms(controller)  # run this  , change from rnn 
		elif  args.controller == 'tran':  # transformer
			controller = Controller_tran(args, num_feature)  # just use para

			'''
			for epoch_count in range(args.epochs):
				model = controller
				LR = 0.001               
				optimizer = torch.optim.Adam(model.parameters(), lr=LR)
				print('train model.concat_output = ', model.concat_output)    # kafeng why not update ????
				print('model.concat_output.shape = ', model.concat_output.shape)     #  [40, 40]
				probs_action =  torch.nn.functional.softmax(model.concat_output, dim=1)

				concat_action = torch.rand([args.num_batch, model.num_action])
				rewards = torch.rand([args.num_batch, model.num_action])
				loss = model(torch.tensor(concat_action), torch.tensor(rewards))
				optimizer.zero_grad() 
				loss = loss.requires_grad_()
				loss.backward() 
				optimizer.step()

				# test
				# probs_action = sess.run(tf.nn.softmax(model.concat_output))
				print('test  model.concat_output = ', model.concat_output)
				print('model.concat_output.shape = ', model.concat_output.shape)   #  [40, 40]
				probs_action =  torch.nn.functional.softmax(model.concat_output, dim=1)
			'''
			
			# controller.build_graph()
			bms_max_result = 0.0 # save result
			bms_max_samples = []  # save samples 
			train_tran(controller)  # run this  , change from rnn 
		elif  args.controller == 'trajectory':  # transformer
			controller = Controller_trajectory(args, num_feature)  # just use para

			''''
			for epoch_count in range(args.epochs):
				model = controller
				LR = 0.001               
				optimizer = torch.optim.Adam(model.parameters(), lr=LR)
				print('train model.concat_output = ', model.concat_output)    # kafeng   
				print('model.concat_output.shape = ', model.concat_output.shape)    # [40, 40]   ???
				probs_action =  torch.nn.functional.softmax(model.concat_output, dim=1)

				concat_action = torch.rand([args.num_batch, model.num_action])
				rewards = torch.rand([args.num_batch, model.num_action])
				loss = model(torch.tensor(concat_action), torch.tensor(rewards))
				optimizer.zero_grad() 
				loss = loss.requires_grad_()
				loss.backward() 
				optimizer.step()

				# test
				# probs_action = sess.run(tf.nn.softmax(model.concat_output))
				print('test  model.concat_output = ', model.concat_output)
				print('model.concat_output.shape = ', model.concat_output.shape)    #  [64, 40]   ????
				probs_action =  torch.nn.functional.softmax(model.concat_output, dim=1)
			'''
			
			# controller.build_graph()
			bms_max_result = 0.0 # save result
			bms_max_samples = []  # save samples 
			train_trajectory(controller)  # run this  , change from rnn 
		elif  args.controller == 'dt':  # transformer
			controller = Controller_dt(args, num_feature, 
			      state_dim=17,  act_dim=6, max_length=20, hidden_size=128,  n_head=1 )  # just use para

			'''
			for epoch_count in range(args.epochs):
				model = controller
				LR = 0.001               
				optimizer = torch.optim.Adam(model.parameters(), lr=LR)
				print('train model.concat_output = ', model.concat_output)    # kafeng why not update ????
				probs_action =  torch.nn.functional.softmax(model.concat_output, dim=1)

				concat_action = torch.rand([args.num_batch, model.num_action])
				rewards = torch.rand([args.num_batch, model.num_action])
				loss = model(torch.tensor(concat_action), torch.tensor(rewards))
				optimizer.zero_grad() 
				loss = loss.requires_grad_()
				loss.backward() 
				optimizer.step()

				# test
				# probs_action = sess.run(tf.nn.softmax(model.concat_output))
				print('test  model.concat_output = ', model.concat_output)
				probs_action =  torch.nn.functional.softmax(model.concat_output, dim=1)
			'''
			
			# controller.build_graph()
			bms_max_result = 0.0 # save result
			bms_max_samples = []  # save samples 
			train_dt(controller)  # run this  , change from rnn 
		elif args.controller == 'random':
			#controller = Controller_random(args, num_feature) # random
			controller = Controller(args, num_feature)  # just use para
			controller.build_graph()
			random_max_result = 0.0 # save result
			random_max_samples = []  # save samples 
			train_random(controller)  # run this 
		else:
			if args.controller == 'rnn':
				controller = Controller(args, num_feature)
			elif args.controller == 'pure':
				controller = Controller_pure(args, num_feature)  # pure
			elif args.controller == 'attention':
				controller = Controller_attention(args, num_feature)  # attention
			controller.build_graph()
			train(controller)  # sklearn
	
		duration = time.time() - start_time
		print('%s  duration = %.5f seconds' %(datetime.datetime.now(), duration))
