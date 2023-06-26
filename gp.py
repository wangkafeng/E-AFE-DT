import argparse
import pandas as pd
import numpy as np

#import xlwt
from xlutils.copy import copy
#import xlrd

import shutil 
from sklearn.decomposition import KernelPCA
import math

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
	default=32, help='batch num')                # kafeng  not batch size ......
parser.add_argument('--optimizer', nargs='?',
	default='adam', help='choose an optimizer')
parser.add_argument('--lr', type=float,
	default=0.01, help='set learning rate')
parser.add_argument('--epochs', type=int,
	default=1, help='training epochs')
parser.add_argument('--evaluate', nargs='?',
	default='f_score', help='choose evaluation method')  # 1-rae
parser.add_argument('--task', nargs='?',
	default='classification', help='choose between classification and regression')
parser.add_argument('--dataset', nargs='?',
	default='PimaIndian', help='choose dataset to run  PimaIndian')
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
	default='rnn', help='choose a controller, random, transfer, rnn, pure, attention')
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
	default='no_cache', help='choose cache method, no_cache, selection ,or trees ')

parser.add_argument('--minhash', type=boolean_string, 
	default=True, help='whether get reward using multiprocess True or False')
parser.add_argument('--dimension_pcws', type=int,
	default=52, help='pcws output length for feature vector. 32: 0.63, 48: 0.709, 52: 0.709 , 56: 0.63,  64:0.665  128: 0.7339  256: ')
parser.add_argument('--feature_extract_alg', nargs='?',
	default='CCWS', help='meta feature extract algorithm, such as minhash algorithm PCWS, statistic. CCWS 0.01 48 0.729, CCWS 0.01 52 0.90 ')
parser.add_argument('--hash_thrshd', type=float,
	default=0.010, help='pcws output length for feature vector. 32: 0.63, 48: 0.709, 52: 0.709 , 56: 0.63,  64:0.665  128: 0.7339  256: ')

parser.add_argument('--init_rl', type=boolean_string,  # kafeng add
	default=True, help='whether get reward using multiprocess True or False')
args = parser.parse_args()
print('args = ', args)
#origin_result = None     # kafeng define global 
orig_entropy_mean = None
method = None # kafeng train/test
#num_process = 64   # kafeng
num_process = args.num_process
infos = []

path = 'data/' + args.dataset + '.csv'


orig_data = pd.read_csv(path)  #  pandas.core.frame.DataFrame
target_label = orig_data[orig_data.columns[-1]]  # pandas.core.frame.Series
orig_features = orig_data.copy()  # 
del orig_features[orig_features.columns[-1]]



'''
# seperate train/test  4:1
path_train = 'data_train/' + args.dataset + '_train.csv'
path_test = 'data_test/' + args.dataset + '_test.csv'

orig_data_train = pd.read_csv(path_train)  #  pandas.core.frame.DataFrame
target_label_train = orig_data_train[orig_data_train.columns[-1]]  # pandas.core.frame.Series
orig_features_train = orig_data_train.copy()  # 
del orig_features_train[orig_features_train.columns[-1]]

orig_data_test = pd.read_csv(path_test)  #  pandas.core.frame.DataFrame
target_label_test = orig_data_test[orig_data_test.columns[-1]]  # pandas.core.frame.Series
orig_features_test = orig_data_test.copy()  # 
del orig_features_test[orig_features_test.columns[-1]]

orig_data = pd.concat([orig_data_train, orig_data_test])
target_label = orig_data[orig_data.columns[-1]]
orig_features = orig_data.copy()  #
del orig_features[orig_features.columns[-1]]
'''

num_feature = orig_data.shape[1] - 1   # 9-1
mum_columns = orig_data.shape[1]  # 9
