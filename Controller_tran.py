import pandas as pd
import numpy as np
import torch
import random
import torch.nn as nn
# import tensorflow as tf
# from keras.models import Sequential
# from keras.layers import Dense, Activation
# from keras import optimizers, regularizers
import transformers   # for DT
from trajectory_gpt2 import GPT2Model  # for DT
import warnings
warnings.filterwarnings("ignore")

class Controller_random:  # 
	def __init__(self, args, num_feature):
		self.num_feature = num_feature
		self.num_op_unary = args.num_op_unary
		self.num_op_binary = args.num_op_binary
		self.num_op = args.num_op_unary + (self.num_feature-1)*args.num_op_binary + 1  # 4 + (8-1)*5  + 1 = 40   unary+binary*7+break
		self.max_order = args.max_order
		self.num_batch = args.num_batch
		self.opt = args.optimizer
		self.lr = args.lr
		self.lr_value = args.lr_value
		self.num_action = self.num_feature * self.max_order  # 8 * 5
		self.reg = args.reg
	
	def _create_inference(self):
		self.outputs = {}
		
		for i in range(self.num_feature):
			# self.outputs['output%d'%i] = []
			tmp_h = self.rnns['rnn%d'%i].zero_state(1, tf.float32)
			tmp_input = tf.reshape(tf.nn.embedding_lookup(self.input0, i), [1,-1])
			for order in range(self.max_order):
				tmp_input, tmp_h = self.rnns['rnn%d'%i].__call__(tmp_input, tmp_h)
				if order == 0:
					self.outputs['output%d'%i] = tmp_input
				else:
					self.outputs['output%d'%i] = tf.concat([self.outputs['output%d'%i], tmp_input], axis=0)
		self.concat_output = tf.concat(list(self.outputs.values()), axis=0, name='concat_output')	
	

class Controller:  # NFS paper write this  rnn
	def __init__(self, args, num_feature):
		self.num_feature = num_feature
		self.num_op_unary = args.num_op_unary
		self.num_op_binary = args.num_op_binary
		# 
		self.num_op = args.num_op_unary + (self.num_feature-1)*args.num_op_binary + 1  # 4 + (8-1)*5  + 1 = 40   unary+binary*7+break
		# num_op include features
		self.num_oprator = (args.num_op_unary+args.num_op_binary) + self.num_feature-1 + 1  # Pu Zhao  4+5 + 8-1 + 1 = 17
		self.max_order = args.max_order
		self.num_batch = args.num_batch
		self.opt = args.optimizer
		self.lr = args.lr
		self.lr_value = args.lr_value
		self.num_action = self.num_feature * self.max_order  # 8 * 5
		self.reg = args.reg
	
	def _create_rnn(self):
		self.rnns = {}
		for i in range(self.num_feature):  # 8   # 8 cnn networks
			# Mr. Pu Zhao , can  cut num_units 
			self.rnns['rnn%d'%i] = tf.contrib.rnn.BasicLSTMCell(num_units=self.num_op, name='rnn%d'%i)  # kafeng features * opeartors = 8 * 40

	def _create_placeholder(self):
		self.concat_action = tf.placeholder(tf.int32, shape=[self.num_batch, self.num_action], name='concat_action')  # 32,  40
		self.rewards = tf.placeholder(tf.float32, shape=[self.num_batch, self.num_action], name='rewards')  # 32, 40

		# estimate value
		self.state = tf.placeholder(tf.int32, shape=[None,self.num_action], name='state') # 32
		self.value = tf.placeholder(tf.float32, shape=[None,1], name='value')


	def _create_variable(self):
		self.input0 = np.ones(shape=[self.num_feature, self.num_op], dtype=np.float32)  # 8 , 40
		self.input0 = self.input0 / self.num_op

		self.value_estimator = Sequential([
			Dense(64, input_dim=self.num_action, kernel_regularizer=regularizers.l2(self.reg)),
			Activation('tanh'),
			Dense(16, kernel_regularizer=regularizers.l2(self.reg)),
			Activation('tanh'),
			Dense(4, kernel_regularizer=regularizers.l2(self.reg)),
			Activation('tanh'),
			Dense(1)])
		self.value_optimizer = optimizers.Adam(lr=self.lr_value)
		self.value_estimator.compile(optimizer=self.value_optimizer, loss='mean_squared_error')

	def _create_inference(self):
		self.outputs = {}
		
		for i in range(self.num_feature):  # 8   # 8 cnn networks
			# self.outputs['output%d'%i] = []
			tmp_h = self.rnns['rnn%d'%i].zero_state(1, tf.float32)  # kafeng  [1, -1]
			tmp_input = tf.reshape(tf.nn.embedding_lookup(self.input0, i), [1, -1])  # kafeng batch_size 1    40 ??
			for order in range(self.max_order):  # kafeng  self.num_op units , just call self.max_order times ???
				tmp_input, tmp_h = self.rnns['rnn%d'%i].__call__(tmp_input, tmp_h)  # (output, next_state) = rnn.call(input, state)
				if order == 0:
					self.outputs['output%d'%i] = tmp_input
				else:
					self.outputs['output%d'%i] = tf.concat([self.outputs['output%d'%i], tmp_input], axis=0)
		self.concat_output = tf.concat(list(self.outputs.values()), axis=0, name='concat_output')	


	def _create_loss(self):
		self.loss = 0.0
		for batch_count in range(self.num_batch):
			action = tf.nn.embedding_lookup(self.concat_action, batch_count)	
			reward = tf.nn.embedding_lookup(self.rewards, batch_count)
		
			action_index = tf.stack([list(range(self.num_action)), action], axis=1)
			action_probs = tf.squeeze(tf.nn.softmax(self.concat_output))    # kafeng cnn controller loss 
			pick_action_prob = tf.gather_nd(action_probs, action_index)
			loss_batch = tf.reduce_sum(-tf.log(pick_action_prob) * reward)
			loss_entropy = tf.reduce_sum(-action_probs * tf.log(action_probs)) * self.reg
			loss_reg = 0.0
			for i in range(self.num_feature):
				weights = self.rnns['rnn%d'%i].weights
				for w in weights:
					loss_reg += self.reg * tf.reduce_sum(tf.square(w))	
			self.loss += loss_batch + loss_entropy + loss_reg
		
		self.loss /= self.num_batch   # kafeng output this loss curve


	def _create_optimizer(self):
		if self.opt == 'adam':
			self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)
		elif self.opt == 'adagrad':
			self.optimizer = tf.train.AdagradOptimizer(learning_rate=self.lr).minimize(self.loss)
			

	def build_graph(self):
		self._create_rnn()
		self._create_placeholder()
		self._create_variable()
		self._create_inference()
		self._create_loss()
		self._create_optimizer()

	def update_policy(self, feed_dict, sess=None):
		_, loss = sess.run([self.optimizer,self.loss], feed_dict=feed_dict)
		return loss

	def update_value(self, state, value, sess=None):
		self.value_estimator.fit(state, value, epochs=20, batch_size=32, verbose=0)


	def predict_value(self, state, sess=None):
		value = self.value_estimator.predict(state)
		return np.squeeze(value)


class Controller_bms:  # bms change from rnn
	def __init__(self, args, num_feature):
		self.num_feature = num_feature
		self.num_op_unary = args.num_op_unary
		self.num_op_binary = args.num_op_binary
		# 
		self.num_op = args.num_op_unary + (self.num_feature-1)*args.num_op_binary + 1  # 4 + (8-1)*5  + 1 = 40   unary+binary*7+break
		# num_op include features
		self.num_oprator = (args.num_op_unary+args.num_op_binary) + self.num_feature-1 + 1  # Pu Zhao  4+5 + 8-1 + 1 = 17
		self.max_order = args.max_order
		self.num_batch = args.num_batch
		self.num_action = self.num_feature * self.max_order  # 8 * 5
		
	def _create_rnn(self):
		self.rnns = {}
		for i in range(self.num_feature):  # 8   # 8 cnn networks
			# Mr. Pu Zhao , can  cut num_units 
			self.rnns['rnn%d'%i] = tf.contrib.rnn.BasicLSTMCell(num_units=self.num_op, name='rnn%d'%i)  # kafeng features * opeartors = 8 * 40

	def _create_placeholder(self):
		self.concat_action = tf.placeholder(tf.int32, shape=[self.num_batch, self.num_action], name='concat_action')  # 32,  40
		self.rewards = tf.placeholder(tf.float32, shape=[self.num_batch, self.num_action], name='rewards')  # 32, 40

		# estimate value
		self.state = tf.placeholder(tf.int32, shape=[None,self.num_action], name='state') # 32
		self.value = tf.placeholder(tf.float32, shape=[None,1], name='value')


	def _create_variable(self):
		self.input0 = np.ones(shape=[self.num_feature, self.num_op], dtype=np.float32)  # 8 , 40
		self.input0 = self.input0 / self.num_op

	def _create_inference(self):
		self.outputs = {}
		
		for i in range(self.num_feature):  # 8   # 8 cnn networks
			# self.outputs['output%d'%i] = []
			tmp_h = self.rnns['rnn%d'%i].zero_state(1, tf.float32)  # kafeng  [1, -1]
			tmp_input = tf.reshape(tf.nn.embedding_lookup(self.input0, i), [1, -1])  # kafeng batch_size    1 , 40
			for order in range(self.max_order):  # kafeng  self.num_op units , just call self.max_order times ???
				tmp_input, tmp_h = self.rnns['rnn%d'%i].__call__(tmp_input, tmp_h)  # (output, next_state) = rnn.call(input, state)
				if order == 0:
					self.outputs['output%d'%i] = tmp_input
				else:
					self.outputs['output%d'%i] = tf.concat([self.outputs['output%d'%i], tmp_input], axis=0)
		self.concat_output = tf.concat(list(self.outputs.values()), axis=0, name='concat_output')	 # num_feature *  max_order ,   num_op
		#self.concat_output = self.outputs
		#self.concat_output = 
		#self.tmp_input = tmp_input

	def build_graph(self):
		self._create_rnn()
		self._create_placeholder()
		self._create_variable()
		self._create_inference()


# 搭建神经网络
class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()

        # 这里小编没有用RNN（）而是LSTM（），即长短期记忆这是RNN的另一种形式，因为通常的RNN可能会出现梯度弥散
        # 或者梯度爆炸的情况，而LSTM（）就解决了该情况。
    
        self.rnn = nn.LSTM(
            input_size=28,
            hidden_size=64,
            num_layers=1,
            # 输入的一共有三个维度分别是 (batch, time_step, input_size)即(批次、时间步长、输入大小),
            # batc_first=True是为了将batch放在第一个维度, 不然有可能因为数顺序问题报错
            batch_first=True, 
        )
        # 全连接层
        self.out = nn.Linear(64, 10) # 64是因为隐藏层是64，10是一共有10个基础数字分类

    # 调用神经网络
    def forward(self, x):
        # 因为循环神经网络是每次读取input后会生成自己的理解即h_state，可以当成一段记忆，一张图片一共是读取28次，每读取
        # 一次会生成一段记忆并与上次记忆相加，读完28次后，会将28次记忆整合形成对这张图片的完整记忆。h_state由(h_n, h_c)
        # 组成，也就是这段记忆，h_n可以理解为主线记忆，h_c可以理解分线记忆，每次读取的生成的记忆可以当作分线记忆，主线
        # 记忆就是相加后的记忆。
        r_out, (h_n, h_c) = self.rnn(x, None)# 因为一开始对图片没有任何理解，所以是None
        #  因为只需要读完一张图片后的记忆，所以只需要最后时间点产生的outputs,而输出的维度是(batch, time_step,
        #  hidden_size)所以只要在中间维度取最后一个就行了
        out = self.out(r_out[:, -1, :])
        return out


# class Controller_tran:  # tran change from rnn
class Controller_tran(nn.Module):   #  for  optimizer parameters      RNNS
	def __init__(self, args, num_feature):
		super().__init__()
		self.num_feature = num_feature
		self.num_op_unary = args.num_op_unary
		self.num_op_binary = args.num_op_binary
		# 
		self.num_op = args.num_op_unary + (self.num_feature-1)*args.num_op_binary + 1  # 4 + (8-1)*5  + 1 = 40   unary+binary*7+break
		# num_op include features
		self.num_oprator = (args.num_op_unary+args.num_op_binary) + self.num_feature-1 + 1  # Pu Zhao  4+5 + 8-1 + 1 = 17
		self.max_order = args.max_order
		self.num_batch = args.num_batch
		self.num_action = self.num_feature * self.max_order  # 8 * 5

		self.concat_action = torch.rand([self.num_batch, self.num_action])
		self.rewards =  torch.rand([self.num_batch, self.num_action])
		self.state = [None,self.num_action]
		self.value = [None,1]

		self.input0 = np.ones(shape=[self.num_feature, self.num_op], dtype=np.float32)  # 8 , 40
		self.input0 = self.input0 / self.num_op          #  1/40 probs
		# print('self.input0 = ', self.input0)
		print('self.input0.shape = ', self.input0.shape)
		print('self.input0.size = ', self.input0.size)

		# self.rnns = {}
		# for i in range(self.num_feature):  # 8   # 8 cnn networks
			# Mr. Pu Zhao , can  cut num_units 
			# self.rnns['rnn%d'%i] = tf.contrib.rnn.BasicLSTMCell(num_units=self.num_op, name='rnn%d'%i)  # kafeng features * opeartors = 8 * 40
			# self.rnns['rnn%d'%i] = torch.nn.RNN(input_size=self.num_op, hidden_size=64, num_layers=1) 

		# self.rnns = nn.LSTM(input_size=self.input0.size ,  hidden_size=self.num_op, num_layers=self.num_feature) 
		# self.rnns = nn.LSTM(input_size=self.num_feature ,  hidden_size=self.num_op, num_layers=self.num_feature) 
		# self.rnns = nn.LSTM(input_size=self.num_feature,  hidden_size=64, num_layers=self.num_feature) 
		#  https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html
		# self.rnns = nn.LSTM(input_size=self.num_op,  hidden_size=self.num_op, num_layers=self.num_feature, batch_first=True)   #  < 0 ?
		self.rnns = nn.LSTM(input_size=self.num_op,  hidden_size=self.num_op, num_layers=1, batch_first=True) 
		print('self.num_feature = ', self.num_feature)
		# self.out = nn.Linear(64, self.num_feature)
		# self.out = nn.Linear(self.num_op, self.num_feature)

		self.outputs = {}
		# self.outputs = []
		# for i in range(self.num_feature): 
		# 	for order in range(self.max_order):
				# self.outputs[i*order + order] = torch.tensor(self.input0[i])
				# self.outputs.append(torch.tensor(self.input0[i]))
		for i in range(self.num_feature * self.max_order): 
			self.outputs[i] = torch.tensor(self.input0[1])   # kafeng use the same initialize
		# print('self.outputs = ', self.outputs)
		# print('torch.tensor(self.outputs).shape = ', torch.tensor(self.outputs).shape)
		# self.concat_output = torch.cat(list(self.outputs.values()), axis=0)
		# self.concat_output = torch.cat(list(self.outputs), axis=0)
		self.concat_output = torch.stack(list(self.outputs.values()), axis=0)
		# self.concat_output = torch.tensor(self.outputs)
		print('self.concat_output = ', self.concat_output)
		print('self.concat_output.shape = ', self.concat_output.shape)   #  [40, 40]

		self.reg = args.reg
		# self.loss = 0.0
		
	# def _create_placeholder(self):
	# 	self.concat_action = tf.placeholder(tf.int32, shape=[self.num_batch, self.num_action], name='concat_action')  # 32,  40
	# 	self.rewards = tf.placeholder(tf.float32, shape=[self.num_batch, self.num_action], name='rewards')  # 32, 40

	# 	# estimate value
	# 	self.state = tf.placeholder(tf.int32, shape=[None,self.num_action], name='state') # 32
	# 	self.value = tf.placeholder(tf.float32, shape=[None,1], name='value')


	# def _create_variable(self):
	# 	self.input0 = np.ones(shape=[self.num_feature, self.num_op], dtype=np.float32)  # 8 , 40
	# 	self.input0 = self.input0 / self.num_op

	# def _create_inference(self):
	'''
	def forward(self):
		self.outputs = {}
		
		for i in range(self.num_feature):  # 8   # 8 cnn networks
			# self.outputs['output%d'%i] = []
			tmp_h = self.rnns[i].zero_state(1, tf.float32)  # kafeng  [1, -1]
			# tmp_input = tf.reshape(tf.nn.embedding_lookup(self.input0, i), [1, -1])  # kafeng batch_size    1 , 40
			tmp_input = self.input0[i].reshape[1, -1]
			for order in range(self.max_order):  # kafeng  self.num_op units , just call self.max_order times ???
				tmp_input, tmp_h = self.rnns[i].__call__(tmp_input, tmp_h)  # (output, next_state) = rnn.call(input, state)
				if order == 0:
					self.outputs[i] = tmp_input
				else:
					self.outputs[i] = torch.concat([self.outputs[i], tmp_input], axis=0)
		self.concat_output = torch.concat(list(self.outputs.values()), axis=0)	 # num_feature *  max_order ,   num_op

		return self.concat_output
	'''	
	def forward(self, actions, rewards):   #  behevoir / target    loss / action
		
		self.outputs = {}
		for i in range(self.num_feature):
			tmp_input = self.input0[i]  # kafeng     40 .    N,L,Hin   ,  L -> order 
			tmp_h = None   #  Defaults to zeros if (h_0, c_0) is not provided.   num_layers,N,Hout​            num_layers,N,Hcell​
			# print('tmp_input.shape = ', tmp_input.shape)
			for order in range(self.max_order):    # kafeng no use rewards, action.  use  self.input0
				# r_out, (h_n, h_c) = self.rnns(actions.astype(float), None)  # kafeng  int -> float
				# input_actions = torch.tensor(actions, dtype=torch.float).view(-1)
				# input_actions = torch.tensor(actions, dtype=torch.float32).view(-1, self.num_op,  self.num_feature)   # kafeng    self.num_feature,  self.num_op  ???    # pass
				# input_actions = torch.tensor(actions, dtype=torch.float32).view(-1, self.max_order,  self.num_feature)      # pass   [32, 5, 8]   ????
				# input_actions = torch.tensor(actions, dtype=torch.float32).view(-1, self.num_batch,  self.num_op)    # 
				tmp_input = torch.tensor(tmp_input, dtype=torch.float32).view(-1, 1,  self.num_op)   #  L = sequence length = 1
				# print('tmp_input.shape = ', tmp_input.shape)         #  [1, 1, 40]
				# r_out, (h_n, h_c) = self.rnns(input_actions, (None, None))  # kafeng  int -> float
				tmp_input, tmp_h = self.rnns(tmp_input, tmp_h)  # output       N,L, Hout
				# print('tmp_input.shape = ', tmp_input.shape)           # [1, 1, 40]
				if order == 0:
					self.outputs[i] = tmp_input
				else:
					self.outputs[i] = torch.cat([self.outputs[i], tmp_input], axis=0)
			
		# self.concat_output = self.out(r_out[:, -1, :])
		# print('self.outputs = ', self.outputs)
		print('torch.stack(list(self.outputs.values()), axis=0).shape  = ', torch.stack(list(self.outputs.values()), axis=0).shape) 
		self.concat_output = torch.stack(list(self.outputs.values()), axis=0).reshape(-1, self.num_op)   #  [8, 5, 1, 40] -> [40, 40]          #  < 0
		# print('self.concat_output.shape = ', self.concat_output.shape)   
	


		# print('rewards = ', rewards)
		# print('rewards.shape = ', rewards.shape)   #  [32, 40]
		# print('actions = ', actions)
		# print('actions.shape = ', actions.shape)   #  [32, 40]
		# self.loss = torch.tensor(0.0)
		# self.loss = torch.tensor(1.0)
		self.loss = torch.rand([1])
		
		
		for batch_count in range(self.num_batch):   # actions, rewards
			# print('batch_count = ', batch_count)
			# action = tf.nn.embedding_lookup(self.concat_action, batch_count)	
			# print('self.concat_action = ', self.concat_action)
			action =  torch.index_select(torch.tensor(self.concat_action, dtype=torch.float32), 0, torch.tensor(batch_count))
			# print('action = ', action)
			# print('action.shape = ', action.shape)   #  [1, 40]
			# reward = tf.nn.embedding_lookup(self.rewards, batch_count)
			# print('self.rewards = ', self.rewards)
			reward = torch.index_select(torch.tensor(self.rewards, dtype=torch.float32), 0, torch.tensor(batch_count))
			# print('reward = ', reward)
			# print('reward.shape = ', reward.shape)   #  [1, 40]
		
			# print('self.num_action = ', self.num_action)
			# action_index = torch.stack([list(range(self.num_action)), action], dim=1)
			# print('range(self.num_action) = ', range(self.num_action))
			# print('list(range(self.num_action)) = ', list(range(self.num_action)))
			# print('torch.tensor(list(range(self.num_action))).shape = ', torch.tensor(list(range(self.num_action))).shape)  #  [40]
			# print('[list(range(self.num_action)), action] = ', [torch.tensor(list(range(self.num_action))), action])
			# print('[list(range(self.num_action)), action] = ', [list(range(self.num_action)), action])
			# print('[torch.tensor(list(range(self.num_action))),  torch.squeeze(action)] = ', [torch.tensor(list(range(self.num_action))),  torch.squeeze(action)])
			# action_index = torch.stack([torch.tensor(list(range(self.num_action))).float(),  torch.squeeze(action)], dim=1)   # 
			# print('action_index = ', action_index)
			# action_index = 1
			# print('self.concat_output = ', self.concat_output)   # < 0
			action_probs = torch.squeeze(torch.nn.functional.softmax(self.concat_output))    # kafeng cnn controller loss 
			# print('action_probs = ', action_probs)
			# print('action_probs.shape = ', action_probs.shape)   #  [40, 40]
			# pick_action_prob = torch.gather_nd(action_probs, action_index)   # tensorflow -> pytorch   operator repalce....     torch.gt
			# pick_action_prob = torch.where(action_probs, action_index)
			# print('pick_action_prob.shape = ', pick_action_prob.shape)   # 
			# pick_action_prob = torch.tensor(0.328)
			# pick_action_prob = torch.tensor(action[1:action_probs.shape[1]])  
			# index = torch.gt(action_probs, action)   #  bool -> long
			# print('index = ', index)
			# pick_action_prob = action_probs.gather(dim=1, index=index.long())   #  bool -> long
			index = torch.LongTensor(random.sample(range(self.num_action*self.num_action), self.num_action))
			# print('index = ', index)
			pick_action_prob =  torch.index_select(action_probs.view(-1), 0, index)
			# print('pick_action_prob.shape = ', pick_action_prob.shape)   #  [0, 40]
			loss_batch = torch.sum(-torch.log(pick_action_prob) * reward)
			loss_entropy = torch.sum(-action_probs * torch.log(action_probs)) * self.reg
			loss_reg = 0.0
			# for i in range(self.num_feature):
			# 	weights = self.rnns['rnn%d'%i].weights
			# 	for w in weights:
			# 		loss_reg += self.reg * tf.reduce_sum(tf.square(w))	
			for i in range(self.num_feature):   #  num_layers
				loss_reg += self.reg * torch.sum(torch.mul(self.rnns.weight_ih_l0[i] , self.rnns.weight_ih_l0[i]))	
			self.loss += loss_batch + loss_entropy + loss_reg
		
		self.loss /= self.num_batch   # kafeng output this loss curve
		
		print('self.loss = ', self.loss)

		return self.loss


class Controller_trajectory(nn.Module):   #  for  optimizer parameters      RNNS
	def __init__(self, args, num_feature):
		super().__init__()
		self.num_feature = num_feature
		self.num_op_unary = args.num_op_unary
		self.num_op_binary = args.num_op_binary
		# 
		self.num_op = args.num_op_unary + (self.num_feature-1)*args.num_op_binary + 1  # 4 + (8-1)*5  + 1 = 40   unary+binary*7+break
		# num_op include features
		self.num_oprator = (args.num_op_unary+args.num_op_binary) + self.num_feature-1 + 1  # Pu Zhao  4+5 + 8-1 + 1 = 17
		self.max_order = args.max_order
		self.num_batch = args.num_batch
		self.num_action = self.num_feature * self.max_order  # 8 * 5

		self.concat_action = torch.rand([self.num_batch, self.num_action])
		self.rewards =  torch.rand([self.num_batch, self.num_action])
		self.state = [None,self.num_action]
		self.value = [None,1]

		self.input0 = np.ones(shape=[self.num_feature, self.num_op], dtype=np.float32)  # 8 , 40
		self.input0 = self.input0 / self.num_op          #  1/40 probs
		# print('self.input0 = ', self.input0)
		print('self.input0.shape = ', self.input0.shape)
		print('self.input0.size = ', self.input0.size)

		
		#  https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html
		# self.rnns = nn.LSTM(input_size=self.num_op,  hidden_size=self.num_op, num_layers=self.num_feature, batch_first=True)   #  < 0 ?
		self.rnns = nn.LSTM(input_size=self.num_op,  hidden_size=self.num_op, num_layers=1, batch_first=True) 
		print('self.num_feature = ', self.num_feature)
		# self.out = nn.Linear(64, self.num_feature)
		# self.out = nn.Linear(self.num_op, self.num_feature)

		self.outputs = {}
		for i in range(self.num_feature * self.max_order): 
			self.outputs[i] = torch.tensor(self.input0[1])   # kafeng use the same initialize
		# print('self.outputs = ', self.outputs)
		# print('torch.tensor(self.outputs).shape = ', torch.tensor(self.outputs).shape)
		# self.concat_output = torch.cat(list(self.outputs.values()), axis=0)
		# self.concat_output = torch.cat(list(self.outputs), axis=0)
		# print('len(self.outputs.values()) = ', len(self.outputs.values()))
		self.concat_output = torch.stack(list(self.outputs.values()), axis=0)
		# self.concat_output = torch.tensor(self.outputs)
		print('self.concat_output = ', self.concat_output)
		print('self.concat_output.shape = ', self.concat_output.shape)   #  [40, 40]

		self.reg = args.reg


	
	def forward(self, actions, rewards):   #  behevoir / target    loss / action
		
		self.outputs = {}
		for i in range(self.num_feature):
			tmp_input = self.input0[i]  # kafeng     40 .    N,L,Hin   ,  L -> order 
			tmp_h = None   #  Defaults to zeros if (h_0, c_0) is not provided.   num_layers,N,Hout​            num_layers,N,Hcell​
			# print('tmp_input.shape = ', tmp_input.shape)
			for order in range(self.max_order):    # kafeng no use rewards, action.  use  self.input0
				# input_actions = torch.tensor(actions, dtype=torch.float32).view(-1, self.num_batch,  self.num_op)    # 
				# tmp_input = torch.tensor(tmp_input, dtype=torch.float32).view(-1, 1,  self.num_op)   #  L = sequence length = 1
				print('order = ', order)
				# print('tmp_input.shape = ', tmp_input.shape)
				if order == 0:
					tmp_input = torch.tensor(tmp_input, dtype=torch.float32).view(-1, 1,  self.num_op)  
				else:
					# print('tmp_input.shape[0] = ', tmp_input.shape[0])
					# tmp_input = torch.tensor(tmp_input, dtype=torch.float32).view(-1, tmp_input.shape[0],  self.num_op) 
					tmp_input = torch.tensor(tmp_input, dtype=torch.float32).view(-1, tmp_input.shape[1],  self.num_op)    #  increase L = sequence length 
				# print('tmp_input.shape = ', tmp_input.shape)         #  [1, 1, 40]
				# r_out, (h_n, h_c) = self.rnns(input_actions, (None, None))  # kafeng  int -> float
				# tmp_input, tmp_h = self.rnns(tmp_input, tmp_h)  # output       N,L, Hout
				tmp_output, tmp_h = self.rnns(tmp_input, tmp_h)
				print('tmp_output.shape = ', tmp_output.shape)           # [1, 1, 40]
				if order == 0:
					self.outputs[i] = tmp_output
				else:
					# self.outputs[i] = torch.cat([self.outputs[i], tmp_output], axis=0)
					self.outputs[i] = tmp_output
					# tmp_input = torch.cat([tmp_input,  tmp_output], axis=0)   #  ???
					tmp_input = torch.cat([tmp_input,  tmp_output], axis=1)   #  increase L = sequence length 
					print('tmp_input.shape = ', tmp_input.shape)

				# print('self.outputs[i] = ', self.outputs[i])    #  1, 2, 4, 8
				# print('len(self.outputs[i]) = ', len(self.outputs[i])) 
				# print('len(self.outputs[i].values()) = ', len(self.outputs[i].values())) 

			# print('len(self.outputs.values()) = ', len(self.outputs.values())) 

		# self.concat_output = self.out(r_out[:, -1, :])
		# print('self.outputs = ', self.outputs)
		# print('len(self.outputs) = ', len(self.outputs))   #  8
		# print('len(self.outputs.values()) = ', len(self.outputs.values()))   #  8   
		# print('torch.stack(list(self.outputs.values()), axis=0).shape  = ', torch.stack(list(self.outputs.values()), axis=0).shape)   #   [8, 1, 8, 40] 
		self.concat_output = torch.stack(list(self.outputs.values()), axis=0).reshape(-1, self.num_op)   #  [8, 5, 1, 40] -> [40, 40]          #  < 0
		print('self.concat_output.shape = ', self.concat_output.shape)     #  [64, 40]  
	

		self.loss = torch.rand([1])
		
		
		for batch_count in range(self.num_batch):   # actions, rewards
			action =  torch.index_select(torch.tensor(self.concat_action, dtype=torch.float32), 0, torch.tensor(batch_count))
			reward = torch.index_select(torch.tensor(self.rewards, dtype=torch.float32), 0, torch.tensor(batch_count))
			# print('reward = ', reward)
			# print('reward.shape = ', reward.shape)   #  [1, 40]
	
			# print('self.concat_output = ', self.concat_output)   # < 0
			action_probs = torch.squeeze(torch.nn.functional.softmax(self.concat_output))    # kafeng cnn controller loss 
			# pick_action_prob = action_probs.gather(dim=1, index=index.long())   #  bool -> long
			index = torch.LongTensor(random.sample(range(self.num_action*self.num_action), self.num_action))
			# print('index = ', index)
			pick_action_prob =  torch.index_select(action_probs.view(-1), 0, index)
			# print('pick_action_prob.shape = ', pick_action_prob.shape)   #  [0, 40]
			loss_batch = torch.sum(-torch.log(pick_action_prob) * reward)
			loss_entropy = torch.sum(-action_probs * torch.log(action_probs)) * self.reg
			loss_reg = 0.0
			for i in range(self.num_feature):   #  num_layers
				loss_reg += self.reg * torch.sum(torch.mul(self.rnns.weight_ih_l0[i] , self.rnns.weight_ih_l0[i]))	
			self.loss += loss_batch + loss_entropy + loss_reg
		
		self.loss /= self.num_batch   # kafeng output this loss curve
		
		print('self.loss = ', self.loss)

		return self.loss


class Controller_dt(nn.Module):   #  for  optimizer parameters      decision transformer     #   TrajectoryModel    (self, state_dim, act_dim, max_length=None):
	def __init__(
            self, args, num_feature, 
            state_dim, act_dim,  hidden_size,           # kafeng   self.num_op
            max_length=None, max_ep_len=4096, action_tanh=True,
            **kwargs
    ):
		super().__init__()
		
		self.num_feature = num_feature
		self.num_op_unary = args.num_op_unary
		self.num_op_binary = args.num_op_binary
		# 
		self.num_op = args.num_op_unary + (self.num_feature-1)*args.num_op_binary + 1  # 4 + (8-1)*5  + 1 = 40   unary+binary*7+break
		# num_op include features
		self.num_oprator = (args.num_op_unary+args.num_op_binary) + self.num_feature-1 + 1  # Pu Zhao  4+5 + 8-1 + 1 = 17
		self.max_order = args.max_order
		self.num_batch = args.num_batch
		self.num_action = self.num_feature * self.max_order  # 8 * 5

		self.concat_action = torch.rand([self.num_batch, self.num_action])
		self.rewards =  torch.rand([self.num_batch, self.num_action])

		self.input0 = np.ones(shape=[self.num_feature, self.num_op], dtype=np.float32)  # 8 , 40
		self.input0 = self.input0 / self.num_op
		# print('self.input0 = ', self.input0)
		print('self.input0.shape = ', self.input0.shape)
		print('self.input0.size = ', self.input0.size)

		# self.rnns = nn.LSTM(input_size=self.num_feature,  hidden_size=self.num_op, num_layers=self.num_feature) 
		# print('self.num_feature = ', self.num_feature)
		# self.out = nn.Linear(64, self.num_feature)
		# self.concat_output = torch.tensor(self.input0)
		# self.concat_output = self.input0
		# print('self.concat_output = ', self.concat_output)

		self.outputs = {}
		for i in range(self.num_feature * self.max_order): 
			self.outputs[i] = torch.tensor(self.input0[1])   # kafeng use the same initialize
		# print('self.outputs = ', self.outputs)
		# print('torch.tensor(self.outputs).shape = ', torch.tensor(self.outputs).shape)
		# self.concat_output = torch.cat(list(self.outputs.values()), axis=0)
		# self.concat_output = torch.cat(list(self.outputs), axis=0)
		# print('len(self.outputs.values()) = ', len(self.outputs.values()))
		self.concat_output = torch.stack(list(self.outputs.values()), axis=0)
		# self.concat_output = torch.tensor(self.outputs)
		# print('self.concat_output = ', self.concat_output)
		# print('self.concat_output.shape = ', self.concat_output.shape)   #  [40, 40]

		
		# kafeng from RNN
		state_dim = self.num_op
		# act_dim = self.max_order
		# act_dim = self.num_op
		act_dim =  32  # batch_size 
		hidden_size = self.num_op
		max_length = self.num_feature * self.max_order
		max_ep_len = self.num_op

		# from  DT
		self.state_dim = state_dim
		self.act_dim = act_dim
		self.max_length = max_length


		self.hidden_size = hidden_size
		config = transformers.GPT2Config(
            vocab_size=1,  # doesn't matter -- we don't use the vocab
            n_embd=hidden_size,
            **kwargs
        )

        # note: the only difference between this GPT2Model and the default Huggingface version
        # is that the positional embeddings are removed (since we'll add those ourselves)
		self.transformer = GPT2Model(config)    # kafeng  use  ours transformer ？？？

		# kafeng change all input to the same hidden_size .
		self.embed_timestep = nn.Embedding(max_ep_len, hidden_size)  #    [1, 40]
		self.embed_return = torch.nn.Linear(1, hidden_size)  #    [1, 40]
		self.embed_state = torch.nn.Linear(self.state_dim, hidden_size)   #     [40, 40]
		self.embed_action = torch.nn.Linear(self.act_dim, hidden_size)  #     [5, 40]  ,   [32, 40]

		self.embed_ln = nn.LayerNorm(hidden_size)

        # note: we don't predict states or returns for the paper
		self.predict_state = torch.nn.Linear(hidden_size, self.state_dim)
		self.predict_action = nn.Sequential(
            *([nn.Linear(hidden_size, self.act_dim)] + ([nn.Tanh()] if action_tanh else []))
        )
		self.predict_return = torch.nn.Linear(hidden_size, 1)


	# def forward(self, states, actions, rewards, masks=None, attention_mask=None):    #  TrajectoryModel   (self, states, actions, rewards, masks=None, attention_mask=None):
	def forward(self, states, actions, rewards, returns_to_go, timesteps, attention_mask=None):    # from  DT
		print('states.shape = ', states.shape)
		print('actions.shape = ', actions.shape)
		print('returns_to_go.shape = ', returns_to_go.shape)
		# print('timesteps = ', timesteps)
		print('timesteps.shape = ', timesteps.shape)
		batch_size, seq_length = states.shape[0], states.shape[1]          # [40, 5, 8]  ,  [40, 40]
		# seq_length = states.shape[1] *states.shape[2]   # kafeng add   5*8 = 40

		if attention_mask is None:
			# attention mask for GPT: 1 if can be attended to, 0 if not
			attention_mask = torch.ones((batch_size, seq_length), dtype=torch.long)

        # embed each modality with a different head
		print('states.view(-1,  self.num_op).shape = ', states.view(-1,  self.num_op).shape)  # [40, 5, 8] -> [40, 40]
		# state_embeddings = self.embed_state(states)   # kafeng  modify
		state_embeddings = self.embed_state(states.view(-1,  self.num_op))    # kafeng reshape to  hidden_size 
		print('state_embeddings.shape = ', state_embeddings.shape)
		# action_embeddings = self.embed_action(actions.view(-1, self.num_op))
		action_embeddings = self.embed_action(actions.view(-1, self.act_dim))   #  batch_size of action
		print('action_embeddings.shape = ', action_embeddings.shape)
		returns_embeddings = self.embed_return(returns_to_go.view(-1, 1))   # nn.Linear
		print('returns_embeddings.shape = ', returns_embeddings.shape)
		# time_embeddings = self.embed_timestep(timesteps.view(-1, self.max_order))   #  Embedding
		# time_embeddings = self.embed_timestep(timesteps.view(-1, 1)) 
		time_embeddings = self.embed_timestep(timesteps)
		print('time_embeddings.shape = ', time_embeddings.shape)

        # time embeddings are treated similar to positional embeddings
		state_embeddings = state_embeddings + time_embeddings
		print('+ state_embeddings.shape = ', state_embeddings.shape)
		action_embeddings = action_embeddings + time_embeddings
		print('+ action_embeddings.shape = ', action_embeddings.shape)
		returns_embeddings = returns_embeddings + time_embeddings
		print('+ returns_embeddings.shape = ', returns_embeddings.shape)

        # this makes the sequence look like (R_1, s_1, a_1, R_2, s_2, a_2, ...)
        # which works nice in an autoregressive sense since states predict actions
		print('torch.stack((returns_embeddings, state_embeddings, action_embeddings), dim=1).shape = ', 
			torch.stack((returns_embeddings, state_embeddings, action_embeddings), dim=1).shape)   #   [40, 3, 40, 40] = 192000
		stacked_inputs = torch.stack(
            (returns_embeddings, state_embeddings, action_embeddings), dim=1
	    ).permute(0, 2, 1, 3).reshape(batch_size, 3*seq_length, self.hidden_size)       #   .permute(0, 2, 1, 3)  ??
        # ).permute(0, 2, 1, 3).reshape(batch_size, 3*seq_length, self.hidden_size)             #   40,  3*5,  40
		stacked_inputs = self.embed_ln(stacked_inputs)

        # to make the attention mask fit the stacked inputs, have to stack it as well
		stacked_attention_mask = torch.stack(
            (attention_mask, attention_mask, attention_mask), dim=1
        ).permute(0, 2, 1).reshape(batch_size, 3*seq_length)

        # we feed in the input embeddings (not word indices as in NLP) to the model
		transformer_outputs = self.transformer(
            inputs_embeds=stacked_inputs,
            attention_mask=stacked_attention_mask,
        )
		x = transformer_outputs['last_hidden_state']
		print('x.shape = ', x.shape)  #  [40, 120, 40]

        # reshape x so that the second dimension corresponds to the original
        # returns (0), states (1), or actions (2); i.e. x[:,1,t] is the token for s_t
		x = x.reshape(batch_size, seq_length, 3, self.hidden_size).permute(0, 2, 1, 3)

        # get predictions
		return_preds = self.predict_return(x[:,2])  # predict next return given state and action
		# print('return_preds = ', return_preds)
		print('return_preds.shape = ', return_preds.shape)
		state_preds = self.predict_state(x[:,2])    # predict next state given state and action
		# print('state_preds = ', state_preds)
		print('state_preds.shape = ', state_preds.shape)
		action_preds = self.predict_action(x[:,1])  # predict next action given state       # kafeng  < 0 ???
		# print('action_preds = ', action_preds)
		print('action_preds.shape = ', action_preds.shape)   #  [40, 40, 32]

		# kafeng add 
		# self.concat_output = action_preds[:, :, 0]   #  kafeng this lead to error ?   retain_graph 

		# probs_action =  torch.nn.functional.softmax(state_preds, dim=1)   #  ？？？？
		# # print('probs_action = ', probs_action)
		# print('probs_action.shape = ', probs_action.shape)

		return state_preds, action_preds, return_preds



	# def get_action(self, states, actions, rewards, **kwargs):
    #     # these will come as tensors on the correct device
	# 	return torch.zeros_like(actions[-1])

	def get_action(self, states, actions, rewards, returns_to_go, timesteps, **kwargs):
        # we don't care about the past rewards in this model

		states = states.reshape(1, -1, self.state_dim)
		actions = actions.reshape(1, -1, self.act_dim)
		returns_to_go = returns_to_go.reshape(1, -1, 1)
		timesteps = timesteps.reshape(1, -1)

		if self.max_length is not None:
			states = states[:,-self.max_length:]
			actions = actions[:,-self.max_length:]
			returns_to_go = returns_to_go[:,-self.max_length:]
			timesteps = timesteps[:,-self.max_length:]

            # pad all tokens to sequence length
			attention_mask = torch.cat([torch.zeros(self.max_length-states.shape[1]), torch.ones(states.shape[1])])
			attention_mask = attention_mask.to(dtype=torch.long, device=states.device).reshape(1, -1)
			states = torch.cat(
				[torch.zeros((states.shape[0], self.max_length-states.shape[1], self.state_dim), device=states.device), states],
                dim=1).to(dtype=torch.float32)
			actions = torch.cat(
                [torch.zeros((actions.shape[0], self.max_length - actions.shape[1], self.act_dim),
                             device=actions.device), actions],
                dim=1).to(dtype=torch.float32)
			returns_to_go = torch.cat(
                [torch.zeros((returns_to_go.shape[0], self.max_length-returns_to_go.shape[1], 1), device=returns_to_go.device), returns_to_go],
                dim=1).to(dtype=torch.float32)
			timesteps = torch.cat(
                [torch.zeros((timesteps.shape[0], self.max_length-timesteps.shape[1]), device=timesteps.device), timesteps],
                dim=1
            ).to(dtype=torch.long)
		else:
			attention_mask = None

		_, action_preds, return_preds = self.forward(
            states, actions, None, returns_to_go, timesteps, attention_mask=attention_mask, **kwargs)
	
		return action_preds[0,-1]




class Controller_attention:  # NFS rnn changed
	def __init__(self, args, num_feature):
		self.num_feature = num_feature
		self.num_op_unary = args.num_op_unary
		self.num_op_binary = args.num_op_binary
		self.num_op = args.num_op_unary + (self.num_feature-1)*args.num_op_binary + 1  # 4 + (8-1)*5  + 1 = 40   unary+binary*7+break
		self.max_order = args.max_order
		self.num_batch = args.num_batch
		self.opt = args.optimizer
		self.lr = args.lr
		self.lr_value = args.lr_value
		self.num_action = self.num_feature * self.max_order  # 8 * 5
		self.reg = args.reg
	
	def _create_rnn(self):
		self.rnns = {}
		for i in range(self.num_feature):  # 8
			self.rnns['rnn%d'%i] = tf.contrib.rnn.BasicLSTMCell(num_units=self.num_op, name='rnn%d'%i)

	def _create_placeholder(self):
		self.concat_action = tf.placeholder(tf.int32, shape=[self.num_batch, self.num_action], name='concat_action')  # 32,  40
		self.rewards = tf.placeholder(tf.float32, shape=[self.num_batch, self.num_action], name='rewards')  # 32, 40

		# estimate value
		self.state = tf.placeholder(tf.int32, shape=[None,self.num_action], name='state') # 32
		self.value = tf.placeholder(tf.float32, shape=[None,1], name='value')


	def _create_variable(self):
		self.input0 = np.ones(shape=[self.num_feature, self.num_op], dtype=np.float32)  # 8 , 40
		self.input0 = self.input0 / self.num_op

		self.value_estimator = Sequential([
			Dense(64, input_dim=self.num_action, kernel_regularizer=regularizers.l2(self.reg)),
			Activation('tanh'),
			Dense(16, kernel_regularizer=regularizers.l2(self.reg)),
			Activation('tanh'),
			Dense(4, kernel_regularizer=regularizers.l2(self.reg)),
			Activation('tanh'),
			Dense(1)])
		self.value_optimizer = optimizers.Adam(lr=self.lr_value)
		self.value_estimator.compile(optimizer=self.value_optimizer, loss='mean_squared_error')

	def _create_inference(self):   # kafeng ???
		self.outputs = {}
		
		for i in range(self.num_feature):  # 8
			# self.outputs['output%d'%i] = []
			tmp_h = self.rnns['rnn%d'%i].zero_state(1, tf.float32)
			#print('tmp_h = ', tmp_h)
			tmp_input = tf.reshape(tf.nn.embedding_lookup(self.input0, i), [1,-1])
			#print('tmp_input = ', tmp_input)
			for order in range(self.max_order):
				tmp_input, tmp_h = self.rnns['rnn%d'%i].__call__(tmp_input, tmp_h)
				if order == 0:
					self.outputs['output%d'%i] = tmp_input
				else:
					self.outputs['output%d'%i] = tf.concat([self.outputs['output%d'%i], tmp_input], axis=0)
		self.concat_output = tf.concat(list(self.outputs.values()), axis=0, name='concat_output')	


	def _create_loss(self):
		self.loss = 0.0
		for batch_count in range(self.num_batch):
			action = tf.nn.embedding_lookup(self.concat_action, batch_count)	
			reward = tf.nn.embedding_lookup(self.rewards, batch_count)
		
			action_index = tf.stack([list(range(self.num_action)), action], axis=1)
			action_probs = tf.squeeze(tf.nn.softmax(self.concat_output))
			pick_action_prob = tf.gather_nd(action_probs, action_index)
			loss_batch = tf.reduce_sum(-tf.log(pick_action_prob) * reward)
			loss_entropy = tf.reduce_sum(-action_probs * tf.log(action_probs)) * self.reg
			loss_reg = 0.0
			for i in range(self.num_feature):  # 8
				weights = self.rnns['rnn%d'%i].weights
				for w in weights:
					loss_reg += self.reg * tf.reduce_sum(tf.square(w))	
			self.loss += loss_batch + loss_entropy + loss_reg
		
		self.loss /= self.num_batch


	def _create_optimizer(self):
		if self.opt == 'adam':
			self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)
		elif self.opt == 'adagrad':
			self.optimizer = tf.train.AdagradOptimizer(learning_rate=self.lr).minimize(self.loss)
			

	def build_graph(self):
		self._create_rnn()
		self._create_placeholder()
		self._create_variable()
		self._create_inference()  # kafeng ???
		self._create_loss()
		self._create_optimizer()

	def update_policy(self, feed_dict, sess=None):
		_, loss = sess.run([self.optimizer,self.loss], feed_dict=feed_dict)
		return loss

	def update_value(self, state, value, sess=None):
		self.value_estimator.fit(state, value, epochs=20, batch_size=32, verbose=0)


	def predict_value(self, state, sess=None):
		value = self.value_estimator.predict(state)
		return np.squeeze(value)

# kafeng no use
class Controller_sequence:
	def __init__(self, args, num_feature):
		self.num_feature = num_feature
		self.num_op_unary = args.num_op_unary
		self.num_op_binary = args.num_op_binary
		self.num_op = args.num_op_unary + (self.num_feature-1)*args.num_op_binary + 1
		self.max_order = args.max_order
		self.num_batch = args.num_batch
		self.opt = args.optimizer
		self.lr = args.lr
		self.lr_value = args.lr_value
		self.num_action = self.num_feature * self.max_order

	
	def _create_rnn(self):
		self.rnn = tf.contrib.rnn.BasicLSTMCell(num_units=self.num_op, name='rnn')

	def _create_placeholder(self):
		self.concat_action = tf.placeholder(tf.int32,
			shape=[self.num_batch,self.num_action], name='concat_action')

		self.rewards = tf.placeholder(tf.float32,
			shape=[self.num_batch,self.num_action], name='rewards')

		# estimate value
		self.state = tf.placeholder(tf.int32, shape=[None,self.num_action], name='state')
		self.value = tf.placeholder(tf.float32, shape=[None,1], name='value')


	def _create_variable(self):
		self.input0 = tf.ones(shape=[1, self.num_op], dtype=tf.float32)
		self.input0 = self.input0 / self.num_op

		self.value_estimator = Sequential([
			Dense(64, input_dim=self.num_action, kernel_regularizer=regularizers.l2(0.01)),
			Activation('tanh'),
			Dense(16, kernel_regularizer=regularizers.l2(0.01)),
			Activation('tanh'),
			Dense(4, kernel_regularizer=regularizers.l2(0.01)),
			Activation('tanh'),
			Dense(1)])
		self.value_optimizer = optimizers.Adam(lr=self.lr_value)
		self.value_estimator.compile(
			optimizer=self.value_optimizer, loss='mean_squared_error')

	def _create_inference(self):
		self.outputs = {}

		tmp_h = self.rnn.zero_state(1, tf.float32)
		tmp_input = tf.reshape(self.input0, [1,-1])
		for action_count in range(self.num_action):
			tmp_input, tmp_h = self.rnn.__call__(tmp_input, tmp_h)
			if action_count == 0:
				self.concat_output = tmp_input
			else:
				self.concat_output = tf.concat([self.concat_output, tmp_input], axis=0)


	def _create_loss(self):
		self.loss = 0.0
		for batch_count in range(self.num_batch):
			action = tf.nn.embedding_lookup(self.concat_action, batch_count)	
			reward = tf.nn.embedding_lookup(self.rewards, batch_count)
			
			action_index = tf.stack([list(range(self.num_action)), action], axis=1)
			action_probs = tf.squeeze(tf.nn.softmax(self.concat_output))
			pick_action_prob = tf.gather_nd(action_probs, action_index)
			loss_batch = tf.reduce_sum(-tf.log(pick_action_prob) * reward)
			loss_entropy = tf.reduce_sum(-action_probs * tf.log(action_probs))
			loss_reg = 0.0
			for w in self.rnn.weights:
				loss_reg += 0.01 * tf.nn.l2_loss(w)
	
			self.loss += loss_batch + loss_entropy + loss_reg
		
		self.loss /= self.num_batch


	def _create_optimizer(self):
		if self.opt == 'adam':
			self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)
			
		elif self.opt == 'adagrad':
			self.optimizer = tf.train.AdagradOptimizer(learning_rate=self.lr).minimize(self.loss)
			

	def build_graph(self):
		self._create_rnn()
		self._create_variable()
		self._create_placeholder()
		self._create_inference()
		self._create_loss()
		self._create_optimizer()

	def update_policy(self, feed_dict, sess=None):
		_, loss = sess.run([self.optimizer,self.loss], feed_dict=feed_dict)
		return loss

	def update_value(self, state, value, sess=None):
		self.value_estimator.fit(state, value, epochs=25, batch_size=32, verbose=0)


	def predict_value(self, state, sess=None):
		value = self.value_estimator.predict(state)
		return np.squeeze(value)


class Controller_pure:
	def __init__(self, args, num_feature):
		self.num_feature = num_feature
		self.num_op_unary = args.num_op_unary
		self.num_op_binary = args.num_op_binary
		self.num_op = args.num_op_unary + (self.num_feature-1)*args.num_op_binary + 1   # 4 + (8-1)*5  + 1 = 40   unary+binary*7+break
		print('self.num_op = ', self.num_op)  # 40
		self.max_order = args.max_order
		self.num_batch = args.num_batch
		self.opt = args.optimizer
		self.lr = args.lr
		self.lr_value = args.lr_value
		self.num_action = self.num_feature * self.max_order    # kafeng  8*5 = 40
		self.reg = args.reg


	def _create_placeholder(self):
		self.concat_action = tf.placeholder(tf.int32, shape=[self.num_batch,self.num_action], name='concat_action')  # 32,  40
		self.rewards = tf.placeholder(tf.float32, shape=[self.num_batch,self.num_action], name='rewards')
		# estimate value
		self.state = tf.placeholder(tf.int32, shape=[None,self.num_action], name='state')
		self.value = tf.placeholder(tf.float32, shape=[None,1], name='value')


	def _create_variable(self):
		self.input0 = np.ones(shape=[self.num_action, self.num_op], dtype=np.float32)  #   40 , 40
		#print('self.input0 = ', self.input0)
		#print('self.input0.shape = ', self.input0.shape)
		self.input0 = self.input0 / self.num_op
		#print('self.input0 222 = ', self.input0)
		self.concat_output = tf.Variable(initial_value=self.input0, name='concat_output', dtype=tf.float32)  # 40, 40

		# kafeng used for AC ???  value function ??
		self.value_estimator = Sequential([
			Dense(64, input_dim=self.num_action, kernel_regularizer=regularizers.l2(self.reg)),
			Activation('tanh'),
			Dense(16, kernel_regularizer=regularizers.l2(self.reg)),
			Activation('tanh'),
			Dense(4, kernel_regularizer=regularizers.l2(self.reg)),
			Activation('tanh'),
			Dense(1)])
		self.value_optimizer = optimizers.Adam(lr=self.lr_value)
		self.value_estimator.compile(optimizer=self.value_optimizer, loss='mean_squared_error')
		

	def _create_loss(self):
		self.loss = 0.0
		print('self.num_batch = ', self.num_batch)
		for batch_count in range(self.num_batch):
			action = tf.nn.embedding_lookup(self.concat_action, batch_count)	
			reward = tf.nn.embedding_lookup(self.rewards, batch_count)
			
			action_index = tf.stack([list(range(self.num_action)), action], axis=1)
			action_probs = tf.squeeze(tf.nn.softmax(self.concat_output))
			pick_action_prob = tf.gather_nd(action_probs, action_index)
			# print("reward: ", reward)
			self.reward_test = reward
			loss_batch = -tf.reduce_sum(reward * tf.log(tf.clip_by_value(pick_action_prob,1e-10,1.0)))
			loss_entropy = -tf.reduce_sum(action_probs * tf.log(tf.clip_by_value(action_probs,1e-10,1.0))) * self.reg

			self.loss += loss_batch + loss_entropy
		self.loss /= self.num_batch


	def _create_optimizer(self):
		if self.opt == 'adam':
			self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)
		elif self.opt == 'adagrad':
			self.optimizer = tf.train.AdagradOptimizer(learning_rate=self.lr).minimize(self.loss)

	def build_graph(self):
		self._create_placeholder()
		self._create_variable()
		self._create_loss()
		self._create_optimizer()

	def update_policy(self, feed_dict, sess=None):
		_, loss, reward_test = sess.run([self.optimizer,self.loss, self.reward_test], feed_dict=feed_dict)
		return loss

	def update_value(self, state, value, sess=None):
		self.value_estimator.fit(state, value, epochs=20, batch_size=32, verbose=0)

	def predict_value(self, state, sess=None):
		value = self.value_estimator.predict(state)
		return np.squeeze(value)
