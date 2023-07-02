#!/usr/bin/env python
"""Toolbox for Weighted MinHash Algorithms

This module contains 13 algorithms: the standard MinHash algorithm for binary sets and
12 algorithms for weighted sets. Each algorithm transforms a data instance (i.e., vector)
into the hash code of the specified length, and computes the time of encoding.

Usage
---------
    >>> from WeightedMinHash import WeightedMinHash
    >>> wmh = WeightedMinHash.WeightedMinHash(data, dimension_num, seed)
    >>> fingerprints_k, fingerprints_y, elapsed = wmh.algorithm_name(...)
      or
    >>> fingerprints, elapsed = wmh.algorithm_name(...)

Parameters
----------
data: {array-like, sparse matrix}, shape (n_features, n_instances)
    a data matrix where row represents feature and column is data instance

dimension_num: int
    the length of hash code

seed: int, default: 1
    part of the seed of the random number generator

Returns
-----------
fingerprints_k: ndarray, shape (n_instances, dimension_num)
    one component of hash code from some algorithms, and each row is the hash code for a data instance

fingerprints_y: ndarray, shape (n_instances, dimension_num)
    one component of hash code from some algorithms, and each row is the hash code for a data instance

fingerprints: ndarray, shape (n_instances, dimension_num)
    only one component of hash code from some algorithms, and each row is the hash code for a data instance

elapsed: float
    time of hashing data matrix

Authors
-----------
Wei WU

See also
-----------
https://sites.google.com/site/aiweiwu88/

Note
-----------
"""

import numpy as np
import scipy as sp
import scipy.sparse as sparse
import time
from ctypes import *

from scipy.sparse import coo_matrix

from numpy import matlib as mb  # kafeng  matlab  matlib.repmat

import ctypes  # kafeng for c
import pandas as pd
import torch

class WeightedMinHashGpu:
    """Main class contains 13 algorithms

    Attributes:
    -----------
    PRIME_RANGE: int
        the range of prime numbers

    PRIMES: ndarray
        a 1-d array to save all prime numbers within PRIME_RANGE, which is used to produce hash functions
                 $\pi = (a*i+b) mod c$, $a, b, c \in PRIMES$
        The two constants are used for minhash(self), haveliwala(self, scale), haeupler(self, scale)

    weighted_set: {array-like, sparse matrix}, shape (n_features, n_instances)
        a data matrix where row represents feature and column is data instance

    dimension_num: int
        the length of hash code

    seed: int, default: 1
        part of the seed of the random number generator. Note that the random seed consists of seed and repeat.

    instance_num: int
        the number of data instances

    feature_num: int
        the number of features
    """

    C_PRIME = 10000000000037

    def __init__(self, weighted_set, dimension_num, seed=1):

        self.weighted_set = weighted_set
        self.dimension_num = dimension_num
        self.seed = seed
        self.instance_num = self.weighted_set.shape[1]  # orig
        self.feature_num = self.weighted_set.shape[0]
        
        #self.instance_num = self.weighted_set.shape[0]
        #self.feature_num = self.weighted_set.shape[1]

    def minhash(self, repeat=1):
        """The standard MinHash algorithm for binary sets
           A. Z. Broder, M. Charikar, A. M. Frieze, and M. Mitzenmacher, "Min-wise Independent Permutations",
           in STOC, 1998, pp. 518-529

        Parameters
        ----------
        repeat: int, default: 1
            the number of repeating the algorithm as the part of the seed of the random number generator

        Returns
        ---------
        fingerprints: ndarray, shape (n_instances, dimension_num)
            hash codes for data matrix, where row represents a data instance

        elapsed: float
            time of hashing data matrix
        """
        fingerprints = np.zeros((self.instance_num, self.dimension_num))

        np.random.seed(self.seed * np.power(2, repeat - 1))

        start = time.time()
        hash_parameters = np.random.randint(1, self.C_PRIME, (self.dimension_num, 2))

        for j_sample in range(0, self.instance_num):
            #print('j_sample = ', j_sample)
            #print(self.weighted_set[:, j_sample])
            #print(sparse.find(self.weighted_set[:, j_sample] > 0))
            #feature_id = sparse.find(self.weighted_set[:, j_sample] > 0)[0]
            feature_id = sparse.find(self.weighted_set[:, j_sample] > 0)[1]
            #print('feature_id = ', feature_id)
            feature_id_num = feature_id.shape[0]

            k_hash = np.mod(
                np.dot(np.transpose(np.array([feature_id])), np.array([np.transpose(hash_parameters[:, 1])])) +
                np.dot(np.ones((feature_id_num, 1)), np.array([np.transpose(hash_parameters[:, 1])])),
                self.C_PRIME)

            min_position = np.argmin(k_hash, axis=0)
            fingerprints[j_sample, :] = feature_id[min_position]

        elapsed = time.time() - start

        return fingerprints, elapsed

    def pcws(self, repeat=1):
        """The Practical Consistent Weighted Sampling (PCWS) algorithm improves the efficiency of ICWS
           by simplifying the mathematical expressions.
           W. Wu, B. Li, L. Chen, and C. Zhang, "Consistent Weighted Sampling Made More Practical",
           in WWW, 2017, pp. 1035-1043.

        Parameters
        ----------
        repeat: int, default: 1
            the number of repeating the algorithm as the part of the seed of the random number generator

        Returns
        -----------
        fingerprints_k: ndarray, shape (n_instances, dimension_num)
            one component of hash codes $(k, y_k)$ for data matrix, where row represents a data instance

        fingerprints_y: ndarray, shape (n_instances, dimension_num)
            one component of hash codes $(k, y_k)$ for data matrix, where row represents a data instance

        elapsed: float
            time of hashing data matrix
        """

        fingerprints_k = np.zeros((self.instance_num, self.dimension_num))  # after compress all instances
        fingerprints_y = np.zeros((self.instance_num, self.dimension_num))

        np.random.seed(self.seed * np.power(2, repeat - 1))
        start = time.time()

        beta = np.random.uniform(0, 1, (self.feature_num, self.dimension_num))  #  compress each instance
        x = np.random.uniform(0, 1, (self.feature_num, self.dimension_num))
        u1 = np.random.uniform(0, 1, (self.feature_num, self.dimension_num))
        u2 = np.random.uniform(0, 1, (self.feature_num, self.dimension_num))

        #print('self.feature_num = ', self.feature_num , 'self.instance_num = ', self.instance_num, 'self.dimension_num = ', self.dimension_num)
        for j_sample in range(0, self.instance_num):  # kafeng calc very instance
            #print('self.weighted_set[:, j_sample] = ', self.weighted_set[:, j_sample].shape)
            feature_id = sparse.find(self.weighted_set[:, j_sample] > 0)[1]  # features > 0 , No.
            #print('u1[feature_id, :] = ', u1[feature_id, :])
            gamma = - np.log(np.multiply(u1[feature_id, :], u2[feature_id, :]))
            #print('feature_id = ', feature_id)  # you 0 hui shanchu
            #print('gamma = ', gamma)
            #t_matrix = np.floor(np.divide(
            #    np.matlib.repmat(np.log(self.weighted_set[feature_id, j_sample].todense()), 1, self.dimension_num),
            #    gamma) + beta[feature_id, :])   # kafeng
            
            #print(self.dimension_num)
            oneDim = np.log(self.weighted_set[feature_id, j_sample])  # (7,)
            #temp = self.weighted_set[feature_id, j_sample]  # (7,)    winder * 3 = (7, 3)
            #print('oneDim = ', oneDim)
            #print('oneDim.shape = ', oneDim.shape)
            #print(type(oneDim))
            
            #print('temp.shape = ', temp.shape)
            #temp2 = mb.repmat(temp, 1, self.dimension_num)  # 7 * 3 = 21  column * 3 = winder
            #print('temp2.shape = ', temp2.shape)  # (1, 21)
            #temp4 = beta[feature_id, :]
            #print('temp4.shape = ', temp4.shape)  # (7, 3)

            if self.dimension_num >= 2:
                myrepeat = np.vstack((oneDim,oneDim))
                for i in range(2, self.dimension_num):
                    myrepeat = np.vstack((myrepeat,oneDim))
            #print(repeat.shape)

            t_matrix = np.floor(np.divide(
                #mb.repmat(np.log(self.weighted_set[feature_id, j_sample]), 1, self.dimension_num),  # myrepeat 
                np.transpose(myrepeat),    # kafeng (len(feature_id), j_sample * self.dimension_num)
                gamma) + beta[feature_id, :])         # kafeng formula 7 
            y_matrix = np.exp(np.multiply(gamma, t_matrix - beta[feature_id, :]))   # kafeng formula 7
            a_matrix = np.divide(-np.log(x[feature_id, :]), np.divide(y_matrix, u1[feature_id, :]))  # kafeng formula 8 ?? improve ?

            # for synth_data error
            #min_position = np.argmin(a_matrix, axis=0)
            #fingerprints_k[j_sample, :] = feature_id[min_position]   # colect all intances reuslt
            #fingerprints_y[j_sample, :] = y_matrix[min_position, np.arange(a_matrix.shape[1])]

        elapsed = time.time() - start

        return fingerprints_k, fingerprints_y, elapsed

    def pcws_gpu(self, repeat=1):
        #print('pcws_gpu')
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        fingerprints_k = torch.zeros(self.instance_num, self.dimension_num).type(torch.cuda.FloatTensor)
        fingerprints_y = torch.zeros(self.instance_num, self.dimension_num).type(torch.cuda.FloatTensor)

        np.random.seed(self.seed * np.power(2, repeat - 1))
        start = time.time()

        beta = torch.rand(self.feature_num, self.dimension_num).type(torch.cuda.FloatTensor)
        x = torch.rand(self.feature_num, self.dimension_num).type(torch.cuda.FloatTensor)
        u1 = torch.rand(self.feature_num, self.dimension_num).type(torch.cuda.FloatTensor)
        u2 = torch.rand(self.feature_num, self.dimension_num).type(torch.cuda.FloatTensor)

        #weighted_set = self.weighted_set
        weighted_set = self.weighted_set.cuda()
        #weighted_set.to(device)
        #weighted_set_cuda = weighted_set.cuda()
        #print('weighted_set.dtype = ', weighted_set.dtype)
        #print('weighted_set.device = ', weighted_set.device)
        #print('weighted_set_cuda.dtype = ', weighted_set_cuda.dtype)
        #print('weighted_set_cuda.device = ', weighted_set_cuda.device)

        #print('self.feature_num = ', self.feature_num , 'self.instance_num = ', self.instance_num, 'self.dimension_num = ', self.dimension_num)
        for j_sample in range(0, self.instance_num):  # kafeng calc very instance
            
            feature_id = sparse.find(self.weighted_set[:, j_sample] > 0)[1]
            #mask = weighted_set[:, j_sample].ge(0)
            #print('mask = ', mask)
            #feature_id = torch.masked_select(weighted_set[:, j_sample], mask)
            #print('feature_id = ', feature_id)

            gamma = -torch.log(torch.mul(u1[feature_id, :], u2[feature_id, :])).type(torch.cuda.FloatTensor)
            
            oneDim = torch.log(weighted_set[feature_id, j_sample])
        
            myrepeat = oneDim.repeat( self.dimension_num, 1).t()

            t_matrix = torch.floor(torch.div(
                myrepeat,  
                gamma) + beta[feature_id, :])   
            y_matrix = torch.exp(torch.mul(gamma, t_matrix - beta[feature_id, :]))    
            a_matrix = torch.div(-torch.log(x[feature_id, :]), torch.div(y_matrix, u1[feature_id, :]))

            #print('a_matrix = ', a_matrix)

            # for synth_data error
            #min_position = torch.argmin(a_matrix, dim=0) 
            #fingerprints_k[j_sample, :] = torch.from_numpy(feature_id[min_position])    
            #fingerprints_y[j_sample, :] = y_matrix[min_position, torch.arange(a_matrix.shape[1])]

        torch.cuda.empty_cache()
        elapsed = time.time() - start

        return fingerprints_k, fingerprints_y, elapsed

    def pcws_cpu(self, repeat=1):
        #print('pcws_cpu')
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        fingerprints_k = torch.zeros(self.instance_num, self.dimension_num).type(torch.FloatTensor)
        fingerprints_y = torch.zeros(self.instance_num, self.dimension_num).type(torch.FloatTensor)

        np.random.seed(self.seed * np.power(2, repeat - 1))
        start = time.time()

        beta = torch.rand(self.feature_num, self.dimension_num).type(torch.FloatTensor)
        x = torch.rand(self.feature_num, self.dimension_num).type(torch.FloatTensor)
        u1 = torch.rand(self.feature_num, self.dimension_num).type(torch.FloatTensor)
        u2 = torch.rand(self.feature_num, self.dimension_num).type(torch.FloatTensor)

        #weighted_set = self.weighted_set
        weighted_set = self.weighted_set
        #weighted_set.to(device)
        #weighted_set_cuda = weighted_set.cuda()
        #print('weighted_set.dtype = ', weighted_set.dtype)
        #print('weighted_set.device = ', weighted_set.device)
        #print('weighted_set_cuda.dtype = ', weighted_set_cuda.dtype)
        #print('weighted_set_cuda.device = ', weighted_set_cuda.device)

        #print('self.feature_num = ', self.feature_num , 'self.instance_num = ', self.instance_num, 'self.dimension_num = ', self.dimension_num)
        for j_sample in range(0, self.instance_num):  # kafeng calc very instance
            
            feature_id = sparse.find(self.weighted_set[:, j_sample] > 0)[1]
            #mask = weighted_set[:, j_sample].ge(0)
            #print('mask = ', mask)
            #feature_id = torch.masked_select(weighted_set[:, j_sample], mask)
            #print('feature_id = ', feature_id)

            gamma = -torch.log(torch.mul(u1[feature_id, :], u2[feature_id, :])).type(torch.FloatTensor)
            
            oneDim = torch.log(weighted_set[feature_id, j_sample])
        
            myrepeat = oneDim.repeat( self.dimension_num, 1).t()

            t_matrix = torch.floor(torch.div(
                myrepeat,  
                gamma) + beta[feature_id, :])   
            y_matrix = torch.exp(torch.mul(gamma, t_matrix - beta[feature_id, :]))    
            a_matrix = torch.div(-torch.log(x[feature_id, :]), torch.div(y_matrix, u1[feature_id, :]))  

            #print('a_matrix = ', a_matrix)
            # for synth_data error
            #min_position = torch.argmin(a_matrix, dim=0) 
            #fingerprints_k[j_sample, :] = torch.from_numpy(feature_id[min_position])    
            #fingerprints_y[j_sample, :] = y_matrix[min_position, torch.arange(a_matrix.shape[1])]

        elapsed = time.time() - start

        return fingerprints_k, fingerprints_y, elapsed


    def pcws_pytorch(self, repeat=1):
    
        fingerprints_k = torch.zeros(self.instance_num, self.dimension_num).type(torch.FloatTensor)
        fingerprints_y = torch.zeros(self.instance_num, self.dimension_num).type(torch.FloatTensor)

        np.random.seed(self.seed * np.power(2, repeat - 1))
        start = time.time()

        beta = torch.rand(self.feature_num, self.dimension_num).type(torch.FloatTensor)
        x = torch.rand(self.feature_num, self.dimension_num).type(torch.FloatTensor)
        u1 = torch.rand(self.feature_num, self.dimension_num).type(torch.FloatTensor)
        u2 = torch.rand(self.feature_num, self.dimension_num).type(torch.FloatTensor)

        weighted_set = torch.from_numpy(self.weighted_set).type(torch.FloatTensor)
        #weighted_set.to(device="cuda")

        #print('self.feature_num = ', self.feature_num , 'self.instance_num = ', self.instance_num, 'self.dimension_num = ', self.dimension_num)
        for j_sample in range(0, self.instance_num):  # kafeng calc very instance
            
            feature_id = sparse.find(weighted_set[:, j_sample] > 0)[1]
            #mask = weighted_set[:, j_sample].ge(0)
            #print('mask = ', mask)
            #feature_id = torch.masked_select(weighted_set[:, j_sample], mask)
            #print('feature_id = ', feature_id)

            gamma = -torch.log(torch.mul(u1[feature_id, :], u2[feature_id, :])).type(torch.FloatTensor)
            
            oneDim = torch.log(torch.from_numpy(self.weighted_set[feature_id, j_sample])).type(torch.FloatTensor)
        
            myrepeat = oneDim.repeat( self.dimension_num, 1).t()

            t_matrix = torch.floor(torch.div(
                myrepeat,  
                gamma) + beta[feature_id, :])   
            y_matrix = torch.exp(torch.mul(gamma, t_matrix - beta[feature_id, :]))    
            a_matrix = torch.div(-torch.log(x[feature_id, :]), torch.div(y_matrix, u1[feature_id, :]))  

            # for synth_data error
            #min_position = torch.argmin(a_matrix, dim=0)
            #fingerprints_k[j_sample, :] = torch.from_numpy(feature_id[min_position])    
            #fingerprints_y[j_sample, :] = y_matrix[min_position, torch.arange(a_matrix.shape[1])]

        elapsed = time.time() - start

        return fingerprints_k, fingerprints_y, elapsed


    def pcws_features(self, repeat=1):

        start = time.time()
        fingerprintsI = np.zeros(self.instance_num * self.dimension_num)
        fingerprintsT = np.zeros(self.instance_num * self.dimension_num)

        np.random.seed(self.seed * np.power(2, repeat - 1))
        start = time.time()

        beta = np.random.uniform(0, 1, (self.feature_num * self.dimension_num))  #  compress each instance
        x = np.random.uniform(0, 1, (self.feature_num * self.dimension_num))
        u1 = np.random.uniform(0, 1, (self.feature_num * self.dimension_num))
        u2 = np.random.uniform(0, 1, (self.feature_num * self.dimension_num))

        runtime = np.array([])

        fingerprints = CDLL('./cpluspluslib/pcws_features.so')
        fingerprints.GenerateFingerprintOfInstance.argtypes = [np.ctypeslib.ndpointer(dtype=c_double, ndim=1,
                                                                                        flags="C_CONTIGUOUS"),
                                                                c_int, c_int, 
                                                                np.ctypeslib.ndpointer(dtype=c_double, ndim=1,
                                                                                        flags="C_CONTIGUOUS"),
                                                                np.ctypeslib.ndpointer(dtype=c_double, ndim=1,
                                                                                        flags="C_CONTIGUOUS"),
                                                                 np.ctypeslib.ndpointer(dtype=c_double, ndim=1,
                                                                                        flags="C_CONTIGUOUS"),
                                                                np.ctypeslib.ndpointer(dtype=c_double, ndim=1,
                                                                                        flags="C_CONTIGUOUS"),
                                                                np.ctypeslib.ndpointer(dtype=c_double, ndim=1,
                                                                                        flags="C_CONTIGUOUS"),
                                                                np.ctypeslib.ndpointer(dtype=c_double, ndim=1,
                                                                                        flags="C_CONTIGUOUS"),
                                                                
                                                                np.ctypeslib.ndpointer(dtype=c_double, ndim=1,
                                                                                        flags="C_CONTIGUOUS")]
        fingerprints.GenerateFingerprintOfInstance.restype = None

        for j_sample in range(0, self.instance_num):  # kafeng calc very instance
            feature = self.weighted_set[:, j_sample]
            #print('feature = ', feature)
            feature_len = len(feature)
            fingerprints.GenerateFingerprintOfInstance(feature, feature_len, self.dimension_num, beta, x, u1, u2, fingerprintsI, fingerprintsT,
                                                        runtime)   # 

        print('fingerprintsI = ', fingerprintsI)
        print('len(fingerprintsI) = ', len(fingerprintsI))
        #print('fingerprintsT = ', fingerprintsT)
        elapsed = time.time() - start

        #return fingerprints_k, fingerprints_y, elapsed
        return 0 ,0, elapsed


    def pcws_fullMatrix(self, repeat=1):

        start = time.time()
        fingerprintsI = np.zeros(self.instance_num * self.dimension_num)
        fingerprintsT = np.zeros(self.instance_num * self.dimension_num)
        runtime = np.array([])

        #print('self.weighted_set = ', self.weighted_set)
        #print(type(self.weighted_set))
        dataset = self.weighted_set.reshape(-1)
        #print('dataset = ', dataset)

        fingerprints = CDLL('./cpluspluslib/pcws_full.so')

        fingerprints.GenerateFingerprintOfInstance.argtypes = [np.ctypeslib.ndpointer(dtype=c_double, ndim=1,
                                                                                        flags="C_CONTIGUOUS"),
                                                                c_int, 
                                                                np.ctypeslib.ndpointer(dtype=c_double, ndim=1,
                                                                                        flags="C_CONTIGUOUS"),
                                                                np.ctypeslib.ndpointer(dtype=c_double, ndim=1,
                                                                                        flags="C_CONTIGUOUS"),

                                                                c_int, c_int,  
                                                                np.ctypeslib.ndpointer(dtype=c_double, ndim=1,
                                                                                        flags="C_CONTIGUOUS")]
        fingerprints.GenerateFingerprintOfInstance.restype = None
        fingerprints.GenerateFingerprintOfInstance(dataset, self.feature_num, fingerprintsI, fingerprintsT,
                                                    self.instance_num, self.dimension_num, runtime)   # getHash3



        print('fingerprintsI = ', fingerprintsI)
        print('len(fingerprintsI) = ', len(fingerprintsI))
        #print('fingerprintsT = ', fingerprintsT)
        elapsed = time.time() - start

        #return fingerprints_k, fingerprints_y, elapsed
        return 0 ,0, elapsed


    def pcws_change3(self, repeat=1):
        #start = time.time()
        
        findResult = sparse.find(self.weighted_set > 0)
        ir = findResult[0]
        #print('ir = ', ir)
        jc = findResult[1]
        #print('jc = ', len(jc))

        feature_id_num = []
        for j_sample in range(0, self.instance_num):
            feature_id = sparse.find(self.weighted_set[:, j_sample] > 0)[1]  # features > 0 , No.
            feature_id_num.append(feature_id.shape[0])
        feature_id_num = np.array(feature_id_num, dtype = np.int32)
        print('feature_id_num = ', feature_id_num)
        
        wordTfidf = self.weighted_set[ir, jc]
        #print('wordTfidf = ', wordTfidf)

        start = time.time()
        fingerprintsI = np.zeros(self.instance_num * self.dimension_num)
        fingerprintsT = np.zeros(self.instance_num * self.dimension_num)
        runtime = np.array([])

        fingerprints = CDLL('./cpluspluslib/pcws_change.so')

        fingerprints.GenerateFingerprintOfInstance.argtypes = [np.ctypeslib.ndpointer(dtype=c_double, ndim=1,
                                                                                        flags="C_CONTIGUOUS"),
                                                                np.ctypeslib.ndpointer(dtype=c_int, ndim=1,
                                                                                        flags="C_CONTIGUOUS"),
                                                        
                                                                np.ctypeslib.ndpointer(dtype=c_int, ndim=1,
                                                                                        flags="C_CONTIGUOUS"),
                                                                c_int, 
                                                                np.ctypeslib.ndpointer(dtype=c_double, ndim=1,
                                                                                        flags="C_CONTIGUOUS"),
                                                                np.ctypeslib.ndpointer(dtype=c_double, ndim=1,
                                                                                        flags="C_CONTIGUOUS"),

                                                                c_int, c_int,  
                                                                np.ctypeslib.ndpointer(dtype=c_double, ndim=1,
                                                                                        flags="C_CONTIGUOUS"),
                                                                np.ctypeslib.ndpointer(dtype=c_int, ndim=1,
                                                                                        flags="C_CONTIGUOUS")]
        fingerprints.GenerateFingerprintOfInstance.restype = None
        fingerprints.GenerateFingerprintOfInstance(wordTfidf, ir,
                                                    jc, self.feature_num, fingerprintsI, fingerprintsT,
                                                    self.instance_num, self.dimension_num, runtime, feature_id_num)   # getHash3



        print('fingerprintsI = ', fingerprintsI)
        print('len(fingerprintsI) = ', len(fingerprintsI))
        #print('fingerprintsT = ', fingerprintsT)
        elapsed = time.time() - start

        #return fingerprints_k, fingerprints_y, elapsed
        return 0 ,0, elapsed

    

    def ccws(self, repeat=1, scale=1):
        """The Canonical Consistent Weighted Sampling (CCWS) algorithm directly uniformly discretizes the original weight
           instead of uniformly discretizing the logarithm of the weight as ICWS.
           W. Wu, B. Li, L. Chen, and C. Zhang, "Canonical Consistent Weighted Sampling for Real-Value Weighetd Min-Hash",
           in ICDM, 2016, pp. 1287-1292.

        Parameters
        ----------
        repeat: int, default: 1
            the number of repeating the algorithm as the part of the seed of the random number generator

        scale: int
            a constant to adapt the weight

        Returns
        -----------
        fingerprints_k: ndarray, shape (n_instances, dimension_num)
            one component of hash codes $(k, y_k)$ for data matrix, where row represents a data instance

        fingerprints_y: ndarray, shape (n_instances, dimension_num)
            one component of hash codes $(k, y_k)$ for data matrix, where row represents a data instance

        elapsed: float
            time of hashing data matrix
        """

        fingerprints_k = np.zeros((self.instance_num, self.dimension_num))
        fingerprints_y = np.zeros((self.instance_num, self.dimension_num))

        np.random.seed(self.seed * np.power(2, repeat - 1))
        start = time.time()

        beta = np.random.uniform(0, 1, (self.feature_num, self.dimension_num))
        gamma = np.random.beta(2, 1, (self.feature_num, self.dimension_num))
        c = np.random.gamma(2, 1, (self.feature_num, self.dimension_num))

        

        for j_sample in range(0, self.instance_num):
            #feature_id = sparse.find(self.weighted_set[:, j_sample] > 0)[0]  # kafeng error
            feature_id = sparse.find(self.weighted_set[:, j_sample] > 0)[1]

            oneDim = self.weighted_set[feature_id, j_sample]
            if self.dimension_num >= 2:
                myrepeat = np.vstack((oneDim,oneDim))
                for i in range(2, self.dimension_num):
                    myrepeat = np.vstack((myrepeat,oneDim))

            t_matrix = np.floor(scale * np.divide(
                #np.matlib.repmat(self.weighted_set[feature_id, j_sample].todense(), 1, self.dimension_num),  # kafeng
                np.transpose(myrepeat),
                gamma[feature_id, :]) + beta[feature_id, :])
            y_matrix = np.multiply(gamma[feature_id, :], (t_matrix - beta[feature_id, :]))
            a_matrix = np.divide(c[feature_id, :], y_matrix) - 2 * np.multiply(gamma[feature_id, :], c[feature_id, :])

            min_position = np.argmin(a_matrix, axis=0)
            fingerprints_k[j_sample, :] = feature_id[min_position]
            fingerprints_y[j_sample, :] = y_matrix[min_position, np.arange(a_matrix.shape[1])]

        elapsed = time.time() - start

        return fingerprints_k, fingerprints_y, elapsed

    def i2cws(self, repeat=1):
        """The Improved Improved Consistent Weighted Sampling (I$^2$CWS) algorithm, samples the two special
           "active indices", $y_k$ and $z_k$, independently by avoiding the equation of $y_k$ and $z_k$ in ICWS.
           W. Wu, B. Li, L. Chen, C. Zhang and P. S. Yu, "Improved Consistent Weighted Sampling Revisited",
           DOI: 10.1109/TKDE.2018.2876250, 2018.

        Parameters
        ----------
        repeat: int, default: 1
            the number of repeating the algorithm as the part of the seed of the random number generator

        Returns
        -----------
        fingerprints_k: ndarray, shape (n_instances, dimension_num)
            one component of hash codes $(k, y_k)$ for data matrix, where row represents a data instance

        fingerprints_y: ndarray, shape (n_instances, dimension_num)
            one component of hash codes $(k, y_k)$ for data matrix, where row represents a data instance

        elapsed: float
            time of hashing data matrix
        """

        fingerprints_k = np.zeros((self.instance_num, self.dimension_num))
        fingerprints_y = np.zeros((self.instance_num, self.dimension_num))

        np.random.seed(self.seed * np.power(2, repeat - 1))
        start = time.time()

        beta1 = np.random.uniform(0, 1, (self.feature_num, self.dimension_num))
        beta2 = np.random.uniform(0, 1, (self.feature_num, self.dimension_num))
        u1 = np.random.uniform(0, 1, (self.feature_num, self.dimension_num))
        u2 = np.random.uniform(0, 1, (self.feature_num, self.dimension_num))
        u3 = np.random.uniform(0, 1, (self.feature_num, self.dimension_num))
        u4 = np.random.uniform(0, 1, (self.feature_num, self.dimension_num))
        v1 = np.random.uniform(0, 1, (self.feature_num, self.dimension_num))
        v2 = np.random.uniform(0, 1, (self.feature_num, self.dimension_num))

        for j_sample in range(0, self.instance_num):
            #feature_id = sparse.find(self.weighted_set[:, j_sample] > 0)[0]  # kafeng error
            feature_id = sparse.find(self.weighted_set[:, j_sample] > 0)[1]

            oneDim = self.weighted_set[feature_id, j_sample]
            if self.dimension_num >= 2:
                myrepeat = np.vstack((oneDim,oneDim))
                for i in range(2, self.dimension_num):
                    myrepeat = np.vstack((myrepeat,oneDim))

            r2 = - np.log(np.multiply(u3[feature_id, :], u4[feature_id, :]))
            t_matrix = np.floor(np.divide(
                #np.matlib.repmat(np.log(self.weighted_set[feature_id, j_sample].todense()), 1, self.dimension_num),  # kafeng
                np.transpose(myrepeat),
                r2) + beta2[feature_id, :])
            z_matrix = np.exp(np.multiply(r2, (t_matrix - beta2[feature_id, :] + 1)))
            a_matrix = np.divide(- np.log(np.multiply(v1[feature_id, :], v2[feature_id, :])), z_matrix)

            min_position = np.argmin(a_matrix, axis=0)
            fingerprints_k[j_sample, :] = feature_id[min_position]

            r1 = - np.log(np.multiply(u1[feature_id[min_position], :], u2[feature_id[min_position], :]))
            gamma1 = np.array([-np.log(np.diag(r1[0]))])

            oneDim2 = self.weighted_set[feature_id[min_position], j_sample]
            if self.dimension_num >= 2:
                myrepeat2 = np.vstack((oneDim2,oneDim2))
                for i in range(2, self.dimension_num):
                    myrepeat2 = np.vstack((myrepeat2,oneDim2))

            b = np.array([np.diag(beta1[feature_id[min_position], :][0])])
            t_matrix = np.floor(np.divide(
                #np.log(np.transpose(self.weighted_set[feature_id[min_position], j_sample].todense())),   # kafeng
                np.transpose(myrepeat2),
                gamma1) + b)
            fingerprints_y[j_sample, :] = np.exp(np.multiply(gamma1, (t_matrix - b)))

        elapsed = time.time() - start

        return fingerprints_k, fingerprints_y, elapsed

    def chum(self, repeat=1):
        """[Chum et. al., 2008] samples an element proportionally to its weight via an exponential distribution
           parameterized with the weight.
           O. Chum, J. Philbin, A. Zisserman, "Near Duplicate Image Detection: Min-Hash and Tf-Idf Weighting",
           in BMVC, vol. 810, 2008, pp. 812-815

        Parameters
        ----------
        repeat: int, default: 1
            the number of repeating the algorithm as the part of the seed of the random number generator

        Returns
        ----------
        fingerprints: ndarray, shape (n_instances, dimension_num)
            hash codes for data matrix, where row represents a data instance

        elapsed: float
            time of hashing data matrix
        """

        fingerprints = np.zeros((self.instance_num, self.dimension_num))

        np.random.seed(self.seed * np.power(2, repeat - 1))
        start = time.time()

        x = np.random.uniform(0, 1, (self.feature_num, self.dimension_num))

        for j_sample in range(0, self.instance_num):
            #feature_id = sparse.find(self.weighted_set[:, j_sample] > 0)[0]  # kafeng
            feature_id = sparse.find(self.weighted_set[:, j_sample] > 0)[1]

            oneDim = self.weighted_set[feature_id, j_sample]
            if self.dimension_num >= 2:
                myrepeat = np.vstack((oneDim,oneDim))
                for i in range(2, self.dimension_num):
                    myrepeat = np.vstack((myrepeat,oneDim))

            k_hash = np.divide(
                -np.log(x[feature_id, :]),
                #np.matlib.repmat(self.weighted_set[feature_id, j_sample].todense(), 1, self.dimension_num)
                np.transpose(myrepeat) # kafeng
                )

            min_position = np.argmin(k_hash, axis=0)
            fingerprints[j_sample, :] = feature_id[min_position]

        elapsed = time.time() - start

        return fingerprints, elapsed

    def gollapudi2(self, repeat=1):
        """[Gollapudi et. al., 2006](2) preserves each weighted element by thresholding normalized real-valued weights
           with random samples.
           S. Gollapudi and R. Panigraphy, "Exploiting Asymmetry in Hierarchical Topic Extraction",
           in CIKM, 2006, pp. 475-482.

        Parameters
        ----------
        repeat: int, default: 1
            the number of repeating the algorithm as the part of the seed of the random number generator

        Returns
        ----------
        fingerprints: ndarray, shape (n_instances, dimension_num)
            hash codes for data matrix, where row represents a data instance

        elapsed: float
            time of hashing data matrix
        """

        fingerprints = np.zeros((self.instance_num, self.dimension_num))

        np.random.seed(self.seed * np.power(2, repeat - 1))

        start = time.time()
        hash_parameters = np.random.randint(1, self.C_PRIME, (self.dimension_num, 2))
        u = np.random.uniform(0, 1, (self.feature_num, self.dimension_num))

        for j_sample in range(0, self.instance_num):
            max_f = np.max(self.weighted_set[:, j_sample])

            #feature_id = sparse.find(u <= np.divide(self.weighted_set[:, j_sample], max_f))[0]  # kafeng
            feature_id = sparse.find(u <= np.divide(self.weighted_set[:, j_sample], max_f))[1]
            feature_id_num = feature_id.shape[0]
            k_hash = np.mod(
                np.dot(np.transpose(np.array([feature_id])), np.array([np.transpose(hash_parameters[:, 1])])) +
                np.dot(np.ones((feature_id_num, 1)), np.array([np.transpose(hash_parameters[:, 1])])),
                self.C_PRIME)

            min_position = np.argmin(k_hash, axis=0)
            fingerprints[j_sample, :] = feature_id[min_position]

        elapsed = time.time() - start

        return fingerprints, elapsed

    def shrivastava(self, repeat=1, scale=1):
        """[Shrivastava, 2016] uniformly samples the area which is composed of the upper bound of each element
           in the universal set by simulating rejection sampling.
           A. Shrivastava, "Simple and Efficient Weighted Minwise Hashing", in NIPS, 2016, pp. 1498-1506.

        Parameters
        ----------
        scale: int
            a constant to adapt the weight

        repeat: int, default: 1
            the number of repeating the algorithm as the part of the seed of the random number generator

        Returns
        ----------
        fingerprints: ndarray, shape (n_instances, dimension_num)
            hash codes for data matrix, where row represents a data instance

        elapsed: float
            time of hashing data matrix
        """
        fingerprints = np.zeros((self.instance_num, self.dimension_num))

        start = time.time()

        bound = np.ceil(np.max(self.weighted_set * scale, 1)).todense().astype(int)
        m_max = np.sum(bound)
        seed = np.arange(1, self.dimension_num+1)

        comp_to_m = np.zeros((1, self.feature_num), dtype=int)
        int_to_comp = np.zeros((1, m_max), dtype=int)
        i_dimension = 0
        for i in range(0, m_max):
            if i == comp_to_m[0, i_dimension] and i_dimension < self.feature_num-1:
                i_dimension = i_dimension + 1
                comp_to_m[0, i_dimension] = comp_to_m[0, i_dimension - 1] + bound[i_dimension - 1, 0]
            int_to_comp[0, i] = i_dimension - 1

        for j_sample in range(0, self.instance_num):
            instance = (scale * self.weighted_set[:, j_sample]).todense()

            for d_id in range(0, self.dimension_num):
                np.random.seed(seed[d_id] * np.power(2, repeat - 1))
                while True:
                    rand_num = np.random.uniform(1, m_max)
                    rand_floor = np.floor(rand_num).astype(int)
                    comp = int_to_comp[0, rand_floor]
                    if rand_num <= comp_to_m[0, comp] + instance[comp]:
                        break
                    fingerprints[j_sample, d_id] = fingerprints[j_sample, d_id] + 1

        elapsed = time.time() - start

        return fingerprints, elapsed
