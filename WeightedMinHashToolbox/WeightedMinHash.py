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

from numpy.ctypeslib import ndpointer 

class WeightedMinHash:
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

    def haveliwala(self, repeat=1, scale=1000):
        """[haveliwala et. al., 2000] directly rounds off the remaining float part
            after each weight is multiplied by a large constant.
            T. H. Haveliwala, A. Gionis, and P. Indyk, "Scalable Techniques for Clustering the Web",
            in WebDB, 2000, pp. 129-134

        Parameters
        ----------
        scale: int, default: 1000
            a large constant to transform real-valued weights into integer ones

        repeat: int, default: 1
            the number of repeating the algorithm as the part of the seed of the random number generator

        Returns
        ----------
        fingerprints: ndarray, shape (n_instances, dimension_num)
            hash codes for data matrix, where row represents a data instance

        elapsed: float
            time of hashing data matrix

        Notes
        ----------
        The operation of expanding the original weighted set by scaling the weights is implemented by C++
        due to low efficiency of Python. The operations cannot be vectorized in Python so that it would be
        very slow.
        """
        fingerprints = np.zeros((self.instance_num, self.dimension_num))

        np.random.seed(self.seed * np.power(2, repeat - 1))

        expanded_set_predefined_size = np.ceil(np.max(np.sum(self.weighted_set * scale, axis=0)) * 100).astype(int)
        start = time.time()
        hash_parameters = np.random.randint(1, self.C_PRIME, (self.dimension_num, 2))

        for j_sample in range(0, self.instance_num):
            expanded_feature_id = np.zeros((1, expanded_set_predefined_size))
            feature_id = sparse.find(self.weighted_set[:, j_sample] > 0)[0]
            feature_id_num = feature_id.shape[0]

            expanded_set = CDLL('./cpluspluslib/haveliwala_expandset.so')
            expanded_set.GenerateExpandedSet.argtypes = [c_int,
                                                         np.ctypeslib.ndpointer(dtype=c_double, ndim=1,
                                                                                flags="C_CONTIGUOUS"),
                                                         np.ctypeslib.ndpointer(dtype=c_int, ndim=1,
                                                                                flags="C_CONTIGUOUS"),
                                                         c_int, c_int,
                                                         np.ctypeslib.ndpointer(dtype=c_double, ndim=1,
                                                                                flags="C_CONTIGUOUS")]
            expanded_set.GenerateExpandedSet.restype = None
            feature_weight = np.round(np.array(scale * self.weighted_set[feature_id, j_sample].todense())[:, 0])
            expanded_feature_id = expanded_feature_id[0, :]

            expanded_set.GenerateExpandedSet(expanded_set_predefined_size, feature_weight, feature_id,
                                             feature_id_num, scale, expanded_feature_id)

            expanded_feature_id = expanded_feature_id[expanded_feature_id != 0]
            expanded_feature_id_num = expanded_feature_id.shape[0]
            k_hash = np.mod(
                np.dot(np.transpose(np.array([expanded_feature_id])), np.array([np.transpose(hash_parameters[:, 1])])) +
                np.dot(np.ones((expanded_feature_id_num, 1)), np.array([np.transpose(hash_parameters[:, 1])])),
                self.C_PRIME)
            min_position = np.argmin(k_hash, axis=0)
            fingerprints[j_sample, :] = expanded_feature_id[min_position]

        elapsed = time.time() - start

        return fingerprints, elapsed

    def haeupler(self, repeat=1, scale=1000):
        """[Haeupler et. al., 2014] preserves the remaining float part with probability
           after each weight is multiplied by a large constant.
           B. Haeupler, M. Manasse, and K. Talwar, "Consistent Weighted Sampling Made Fast, Small, and Easy",
           arXiv preprint arXiv: 1410.4266, 2014

        Parameters
        ----------
        scale: int, default: 1000
            a large constant to transform real-valued weights into integer ones

        repeat: int, default: 1
            the number of repeating the algorithm as the part of the seed of the random number generator

        Returns
        ----------
        fingerprints: ndarray, shape (n_instances, dimension_num)
            hash codes for data matrix, where row represents a data instance

        elapsed: float
            time of hashing data matrix

        Notes
        ----------
        The operation of expanding the original weighted set by scaling the weights is implemented by C++
        due to low efficiency of Python. The operations cannot be vectorized in Python so that it would be
        very slow.
        """
        fingerprints = np.zeros((self.instance_num, self.dimension_num))

        np.random.seed(self.seed * np.power(2, repeat - 1))

        expanded_set_predefined_size = np.ceil(np.max(np.sum(self.weighted_set * scale, axis=0)) * 100).astype(int)
        start = time.time()
        hash_parameters = np.random.randint(1, self.C_PRIME, (self.dimension_num, 2))

        for j_sample in range(0, self.instance_num):

            expanded_feature_id = np.zeros((1, expanded_set_predefined_size))
            feature_id = sparse.find(self.weighted_set[:, j_sample] > 0)[0]
            feature_id_num = feature_id.shape[0]

            expanded_set = CDLL('./cpluspluslib/haeupler_expandset.so')
            expanded_set.GenerateExpandedSet.argtypes = [c_int,
                                                         np.ctypeslib.ndpointer(dtype=c_double, ndim=1,
                                                                                flags="C_CONTIGUOUS"),
                                                         np.ctypeslib.ndpointer(dtype=c_int, ndim=1,
                                                                                flags="C_CONTIGUOUS"),
                                                         c_int, c_int, c_int,
                                                         np.ctypeslib.ndpointer(dtype=c_double, ndim=1,
                                                                                flags="C_CONTIGUOUS")]
            expanded_set.GenerateExpandedSet.restype = None
            feature_weight = np.array(scale * self.weighted_set[feature_id, j_sample].todense())[:, 0]
            expanded_feature_id = expanded_feature_id[0, :]
            expanded_set.GenerateExpandedSet(expanded_set_predefined_size, feature_weight, feature_id, feature_id_num,
                                             scale, self.seed * repeat, expanded_feature_id)

            expanded_feature_id = expanded_feature_id[expanded_feature_id != 0]
            expanded_feature_id_num = expanded_feature_id.shape[0]
            k_hash = np.mod(
                np.dot(np.transpose(np.array([expanded_feature_id])), np.array([np.transpose(hash_parameters[:, 1])])) +
                np.dot(np.ones((expanded_feature_id_num, 1)), np.array([np.transpose(hash_parameters[:, 1])])),
                self.C_PRIME)
            min_position = np.argmin(k_hash, axis=0)
            fingerprints[j_sample, :] = expanded_feature_id[min_position]

        elapsed = time.time() - start

        return fingerprints, elapsed

    def gollapudi1(self, repeat=1, scale=1000):
        """[Gollapudi et. al., 2006](1) is an integer weighted MinHash algorithm,
           which skips much unnecessary hash value computation by employing the idea of "active index".
           S. Gollapudi and R. Panigraphy, "Exploiting Asymmetry in Hierarchical Topic Extraction",
           in CIKM, 2006, pp. 475-482.

        Parameters
        -----------
        repeat: int, default: 1
            the number of repeating the algorithm as the part of the seed of the random number generator

        scale: int, default: 1000
            a large constant to transform real-valued weights into integer ones

        Returns
        -----------
        fingerprints_k: ndarray, shape (n_instances, dimension_num)
            one component of hash codes $(k, y_k)$ for data matrix, where row represents a data instance

        fingerprints_y: ndarray, shape (n_instances, dimension_num)
            one component of hash codes $(k, y_k)$ for data matrix, where row represents a data instance

        elapsed: float
            time of hashing data matrix

        Notes
        ----------
        The operations of seeking "active indices" and computing hashing values are implemented by C++
        due to low efficiency of Python. The operations cannot be vectorized in Python so that it would be
        very slow.
        """
        fingerprints_k = np.zeros((self.instance_num, self.dimension_num))
        fingerprints_y = np.zeros((self.instance_num, self.dimension_num))

        start = time.time()

        for j_sample in range(0, self.instance_num):

            #feature_id = sparse.find(self.weighted_set[:, j_sample] > 0)[0]  # kafeng error
            feature_id = sparse.find(self.weighted_set[:, j_sample] > 0)[1]
            feature_id_num = feature_id.shape[0]

            fingerprints = CDLL('./cpluspluslib/gollapudi1_fingerprints.so')
            fingerprints.GenerateFingerprintOfInstance.argtypes = [c_int,
                                                                   np.ctypeslib.ndpointer(dtype=c_int, ndim=1,
                                                                                          flags="C_CONTIGUOUS"),
                                                                   np.ctypeslib.ndpointer(c_int, ndim=1,
                                                                                          flags="C_CONTIGUOUS"),
                                                                   c_int, c_int,
                                                                   np.ctypeslib.ndpointer(dtype=c_double, ndim=1,
                                                                                          flags="C_CONTIGUOUS"),
                                                                   np.ctypeslib.ndpointer(dtype=c_double, ndim=1,
                                                                                          flags="C_CONTIGUOUS")]
            fingerprints.GenerateFingerprintOfInstance.restype = None
            #feature_weight = np.array(scale * self.weighted_set[feature_id, j_sample].todense())[:, 0]  # orig
            feature_weight = np.array(scale * self.weighted_set[feature_id, j_sample]).astype(np.int32)  # float64 ?
            '''
            #print(self.weighted_set[feature_id, j_sample])
            #print(type(self.weighted_set[feature_id, j_sample]))
            #print(scale * self.weighted_set[feature_id, j_sample])
            tmp = np.array(scale * self.weighted_set[feature_id, j_sample])
            #print(type(tmp))
            #print(tmp.shape)
            #feature_weight = tmp[:, 0]  # 1st column
            feature_weight = tmp.astype(np.int32)
            #print(type(feature_weight[0]))
            #feature_weight = np.int32(np.array(scale * coo_matrix(self.weighted_set[feature_id, j_sample]).todense())[:, 0])  # kafeng
            '''
            fingerprint_k = np.zeros((1, self.dimension_num))[0]
            fingerprint_y = np.zeros((1, self.dimension_num))[0]

            fingerprints.GenerateFingerprintOfInstance(self.dimension_num,
                                                       feature_weight, feature_id, feature_id_num, self.seed * repeat,
                                                       fingerprint_k, fingerprint_y)

            fingerprints_k[j_sample, :] = fingerprint_k
            fingerprints_y[j_sample, :] = fingerprint_y

        elapsed = time.time() - start

        return fingerprints_k, fingerprints_y, elapsed

    def cws(self, repeat=1):
        """The Consistent Weighted Sampling (CWS) algorithm, as the first of the Consistent Weighted Sampling scheme,
           extends "active indices" from $[0, S]$ in [Gollapudi et. al., 2006](1) to $[0, +\infty]$.
           M. Manasse, F. McSherry, and K. Talwar, "Consistent Weighted Sampling", Unpublished technical report, 2010.

        Parameters
        -----------
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

        Notes
        ----------
        The operations of seeking "active indices" and computing hashing values are implemented by C++
        due to low efficiency of Python. The operations cannot be vectorized in Python so that it would be
        very slow.
        """

        fingerprints_k = np.zeros((self.instance_num, self.dimension_num))
        fingerprints_y = np.zeros((self.instance_num, self.dimension_num))

        start = time.time()

        for j_sample in range(0, self.instance_num):

            #feature_id = sparse.find(self.weighted_set[:, j_sample] > 0)[0]  # kafeng error
            feature_id = sparse.find(self.weighted_set[:, j_sample] > 0)[1]
            feature_id_num = feature_id.shape[0]

            fingerprints = CDLL('./cpluspluslib/cws_fingerprints.so')
            fingerprints.GenerateFingerprintOfInstance.argtypes = [c_int,
                                                                   np.ctypeslib.ndpointer(dtype=c_double, ndim=1,
                                                                                          flags="C_CONTIGUOUS"),
                                                                   np.ctypeslib.ndpointer(dtype=c_int, ndim=1,
                                                                                          flags="C_CONTIGUOUS"),
                                                                   c_int, c_int,
                                                                   np.ctypeslib.ndpointer(dtype=c_double, ndim=1,
                                                                                          flags="C_CONTIGUOUS"),
                                                                   np.ctypeslib.ndpointer(dtype=c_double, ndim=1,
                                                                                          flags="C_CONTIGUOUS")]
            fingerprints.GenerateFingerprintOfInstance.restype = None
            #weights = np.array(self.weighted_set[feature_id, j_sample].todense())[:, 0]
            #print('np.array(self.weighted_set[feature_id, j_sample]) = ', np.array(self.weighted_set[feature_id, j_sample]))
            #weights = np.array(self.weighted_set[feature_id, j_sample])
            weights = np.array(self.weighted_set[feature_id, j_sample]).astype(np.float64)
            fingerprint_k = np.zeros((1, self.dimension_num))[0]
            fingerprint_y = np.zeros((1, self.dimension_num))[0]

            fingerprints.GenerateFingerprintOfInstance(self.dimension_num,
                                                       weights, feature_id, feature_id_num, self.seed * repeat,
                                                       fingerprint_k, fingerprint_y)

            fingerprints_k[j_sample, :] = fingerprint_k
            fingerprints_y[j_sample, :] = fingerprint_y

        elapsed = time.time() - start

        return fingerprints_k, fingerprints_y, elapsed

    def icws(self, repeat=1):
        """The Improved Consistent Weighted Sampling (ICWS) algorithm, directly samples the two special "active indices",
           $y_k$ and $z_k$.
           S. Ioffe, "Improved Consistent Weighted Sampling, Weighted Minhash and L1 Sketching",
           in ICDM, 2010, pp. 246-255.

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

        beta = np.random.uniform(0, 1, (self.feature_num, self.dimension_num))
        v1 = np.random.uniform(0, 1, (self.feature_num, self.dimension_num))
        v2 = np.random.uniform(0, 1, (self.feature_num, self.dimension_num))
        u1 = np.random.uniform(0, 1, (self.feature_num, self.dimension_num))
        u2 = np.random.uniform(0, 1, (self.feature_num, self.dimension_num))

        for j_sample in range(0, self.instance_num):
            #print('j_sample = ', j_sample)
            #print(self.weighted_set[:, j_sample])
            #print(self.weighted_set[:, j_sample] > 0)
            #print(sparse.find(self.weighted_set[:, j_sample] > 0))
            #feature_id = sparse.find(self.weighted_set[:, j_sample] > 0)[0]   #  kafeng error
            feature_id = sparse.find(self.weighted_set[:, j_sample] > 0)[1]
            gamma = - np.log(np.multiply(u1[feature_id, :], u2[feature_id, :]))
            #t_matrix = np.floor(np.divide(
            #    np.matlib.repmat(np.log(self.weighted_set[feature_id, j_sample].todense()), 1, self.dimension_num),
            #    gamma) + beta[feature_id, :])
            oneDim = np.log(self.weighted_set[feature_id, j_sample])
            #print('oneDim.shape = ', oneDim.shape)
            #temp = np.matlib.repmat(np.log(self.weighted_set[feature_id, j_sample]), 1, self.dimension_num)
            #temp = np.reshape(np.log(self.weighted_set[feature_id, j_sample]), self.feature_num, self.dimension_num)
            if self.dimension_num >= 2:
                myrepeat = np.vstack((oneDim,oneDim))
                for i in range(2, self.dimension_num):
                    myrepeat = np.vstack((myrepeat,oneDim))
            t_matrix = np.floor(np.divide(
                #np.matlib.repmat(np.log(self.weighted_set[feature_id, j_sample]), 1, self.dimension_num),
                np.transpose(myrepeat),
                gamma) + beta[feature_id, :])
            y_matrix = np.exp(np.multiply(gamma, t_matrix - beta[feature_id, :]))  # kafeng formula 7  
            a_matrix = np.divide(np.multiply(-np.log(np.multiply(v1[feature_id, :], v2[feature_id, :])),
                                             np.multiply(u1[feature_id, :], u2[feature_id, :])), y_matrix)  # kafeng formula 8 ? in pcws

            min_position = np.argmin(a_matrix, axis=0)
            fingerprints_k[j_sample, :] = feature_id[min_position]
            fingerprints_y[j_sample, :] = y_matrix[min_position, np.arange(a_matrix.shape[1])]

        elapsed = time.time() - start

        return fingerprints_k, fingerprints_y, elapsed

    def icws_pytorch(self, repeat=1):

        fingerprints_k = torch.zeros(self.instance_num, self.dimension_num).type(torch.FloatTensor)
        fingerprints_y = torch.zeros(self.instance_num, self.dimension_num).type(torch.FloatTensor)

        np.random.seed(self.seed * np.power(2, repeat - 1))
        start = time.time()

        beta = torch.tensor(np.random.uniform(0, 1, (self.feature_num, self.dimension_num))).type(torch.FloatTensor)
        v1 = torch.tensor(np.random.uniform(0, 1, (self.feature_num, self.dimension_num))).type(torch.FloatTensor)
        v2 = torch.tensor(np.random.uniform(0, 1, (self.feature_num, self.dimension_num))).type(torch.FloatTensor)
        u1 = torch.tensor(np.random.uniform(0, 1, (self.feature_num, self.dimension_num))).type(torch.FloatTensor)
        u2 = torch.tensor(np.random.uniform(0, 1, (self.feature_num, self.dimension_num))).type(torch.FloatTensor)

        for j_sample in range(0, self.instance_num):
            
            feature_id = sparse.find(self.weighted_set[:, j_sample] > 0)[1]
            gamma = - torch.log(torch.mul(u1[feature_id, :], u2[feature_id, :]))
           
            oneDim = torch.log(torch.tensor(self.weighted_set[feature_id, j_sample]).type(torch.FloatTensor))
            '''
            if self.dimension_num >= 2:
                myrepeat = np.vstack((oneDim,oneDim))
                for i in range(2, self.dimension_num):
                    myrepeat = np.vstack((myrepeat,oneDim))
            '''
            myrepeat = oneDim.repeat(self.dimension_num, 1).t()
            t_matrix = torch.floor(torch.div(
                #np.transpose(myrepeat),
                myrepeat,
                gamma) + beta[feature_id, :])
            y_matrix = torch.exp(torch.mul(gamma, t_matrix - beta[feature_id, :]))  # kafeng formula 7  
            a_matrix = torch.div(torch.mul(-torch.log(torch.mul(v1[feature_id, :], v2[feature_id, :])),
                                             torch.mul(u1[feature_id, :], u2[feature_id, :])), y_matrix)  # kafeng formula 8 ? in pcws

            min_position = np.argmin(a_matrix, axis=0)
            fingerprints_k[j_sample, :] = torch.from_numpy(feature_id[min_position])
            fingerprints_y[j_sample, :] = y_matrix[min_position, np.arange(a_matrix.shape[1])]

        elapsed = time.time() - start

        return fingerprints_k, fingerprints_y, elapsed

    def licws(self, repeat=1):
        """The 0-bit Consistent Weighted Sampling (0-bit CWS) algorithm generates the original hash code $(k, y_k)$
           by running ICWS, but finally adopts only $k$ to constitute the fingerprint.
           P. Li, "0-bit Consistent Weighted Sampling", in KDD, 2015, pp. 665-674.

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

        beta = np.random.uniform(0, 1, (self.feature_num, self.dimension_num))  # 7, 3
        v1 = np.random.uniform(0, 1, (self.feature_num, self.dimension_num))
        v2 = np.random.uniform(0, 1, (self.feature_num, self.dimension_num))
        u1 = np.random.uniform(0, 1, (self.feature_num, self.dimension_num))
        u2 = np.random.uniform(0, 1, (self.feature_num, self.dimension_num))

        for j_sample in range(0, self.instance_num):
            #feature_id = sparse.find(self.weighted_set[:, j_sample] > 0)[0]  # kafeng error
            feature_id = sparse.find(self.weighted_set[:, j_sample] > 0)[1]
            gamma = - np.log(np.multiply(u1[feature_id, :], u2[feature_id, :]))
            #t_matrix = np.floor(np.divide(
            #    np.matlib.repmat(np.log(self.weighted_set[feature_id, j_sample].todense()), 1, self.dimension_num),
            #    gamma) + beta[feature_id, :])   # kafeng
            oneDim = np.log(self.weighted_set[feature_id, j_sample])
            #print('oneDim.shape = ', oneDim.shape)  # (7,)    beta[feature_id, :]  (7, 3)
            if self.dimension_num >= 2:
                myrepeat = np.vstack((oneDim,oneDim))
                for i in range(2, self.dimension_num):
                    myrepeat = np.vstack((myrepeat,oneDim))

            t_matrix = np.floor(np.divide(
                #np.matlib.repmat(np.log(self.weighted_set[feature_id, j_sample]), 1, self.dimension_num),
                np.transpose(myrepeat),
                gamma) + beta[feature_id, :])
            y_matrix = np.exp(np.multiply(gamma, t_matrix - beta[feature_id, :]))
            a_matrix = np.divide(np.multiply(-np.log(np.multiply(v1[feature_id, :], v2[feature_id, :])),
                                             np.multiply(u1[feature_id, :], u2[feature_id, :])), y_matrix)

            min_position = np.argmin(a_matrix, axis=0)
            fingerprints[j_sample, :] = feature_id[min_position]

        elapsed = time.time() - start

        return fingerprints, elapsed

    def licws_pytorch(self, repeat=1):

        #fingerprints = np.zeros((self.instance_num, self.dimension_num))
        fingerprints = torch.zeros(self.instance_num, self.dimension_num).type(torch.FloatTensor)

        np.random.seed(self.seed * np.power(2, repeat - 1))
        start = time.time()

        beta = torch.tensor(np.random.uniform(0, 1, (self.feature_num, self.dimension_num))).type(torch.FloatTensor)  # 7, 3
        v1 = torch.tensor(np.random.uniform(0, 1, (self.feature_num, self.dimension_num))).type(torch.FloatTensor)
        v2 = torch.tensor(np.random.uniform(0, 1, (self.feature_num, self.dimension_num))).type(torch.FloatTensor)
        u1 = torch.tensor(np.random.uniform(0, 1, (self.feature_num, self.dimension_num))).type(torch.FloatTensor)
        u2 = torch.tensor(np.random.uniform(0, 1, (self.feature_num, self.dimension_num))).type(torch.FloatTensor)

        for j_sample in range(0, self.instance_num):
            #feature_id = sparse.find(self.weighted_set[:, j_sample] > 0)[0]  # kafeng error
            feature_id = sparse.find(self.weighted_set[:, j_sample] > 0)[1]
            #gamma = - np.log(np.multiply(u1[feature_id, :], u2[feature_id, :]))
            gamma = - torch.log(torch.mul(u1[feature_id, :], u2[feature_id, :]))

            #t_matrix = np.floor(np.divide(
            #    np.matlib.repmat(np.log(self.weighted_set[feature_id, j_sample].todense()), 1, self.dimension_num),
            #    gamma) + beta[feature_id, :])   # kafeng
            '''
            oneDim = np.log(self.weighted_set[feature_id, j_sample])
            #print('oneDim.shape = ', oneDim.shape)  # (7,)    beta[feature_id, :]  (7, 3)
            if self.dimension_num >= 2:
                myrepeat = np.vstack((oneDim,oneDim))
                for i in range(2, self.dimension_num):
                    myrepeat = np.vstack((myrepeat,oneDim))
            '''
            oneDim = torch.log(torch.tensor(self.weighted_set[feature_id, j_sample]).type(torch.FloatTensor))
            myrepeat = oneDim.repeat(self.dimension_num, 1).t()
            t_matrix = torch.floor(torch.div(
                myrepeat,
                gamma) + beta[feature_id, :])
            y_matrix = torch.exp(torch.mul(gamma, t_matrix - beta[feature_id, :]))
            a_matrix = torch.div(torch.mul(-torch.log(np.multiply(v1[feature_id, :], v2[feature_id, :])),
                                             torch.mul(u1[feature_id, :], u2[feature_id, :])), y_matrix)

            min_position = np.argmin(a_matrix, axis=0)
            fingerprints[j_sample, :] = torch.from_numpy(feature_id[min_position])

        elapsed = time.time() - start

        return fingerprints, elapsed


    def pcws_0(self, repeat=1):
        
        np.set_printoptions(threshold=np.inf)
        np.set_printoptions(precision=3, suppress=True, linewidth=150)

        fingerprints_k = np.zeros((self.instance_num, self.dimension_num))  # after compress all instances
        fingerprints_y = np.zeros((self.instance_num, self.dimension_num))

        np.random.seed(self.seed * np.power(2, repeat - 1))
        start = time.time()

        beta = np.random.uniform(0, 1, (self.feature_num, self.dimension_num))  #  compress each instance
        x = np.random.uniform(0, 1, (self.feature_num, self.dimension_num))
        u1 = np.random.uniform(0, 1, (self.feature_num, self.dimension_num))
        u2 = np.random.uniform(0, 1, (self.feature_num, self.dimension_num))

        print('self.feature_num = ', self.feature_num , 'self.instance_num = ', self.instance_num, 'self.dimension_num = ', self.dimension_num)
        for j_sample in range(0, self.instance_num):  # kafeng calc very instance
            #print('self.weighted_set[:, j_sample] = ', self.weighted_set[:, j_sample].shape)
            #feature_id = sparse.find(self.weighted_set[:, j_sample] > 0)[1]  # features > 0 , No.
            #feature_id = self.weighted_set[:, j_sample]
            #print('feature_id = ', feature_id)  # > 0 
            #print('type(feature_id) = ', type(feature_id))
            gamma = - np.log(np.multiply(u1[:, :], u2[:, :]))
            oneDim = np.log(self.weighted_set[:, j_sample])  # (7,)   # cpp error 
           
            if self.dimension_num >= 2:
                myrepeat = np.vstack((oneDim,oneDim))
                for i in range(2, self.dimension_num):
                    myrepeat = np.vstack((myrepeat,oneDim))
        

            t_matrix = np.floor(np.divide(
                np.transpose(myrepeat),     
                gamma) + beta[:, :])         
            
            y_matrix = np.exp(np.multiply(gamma, t_matrix - beta[:, :]))   
            a_matrix = np.divide(-np.log(x[:, :]), np.divide(y_matrix, u1[:, :]))  

            min_position = np.argmin(a_matrix, axis=0)
            #fingerprints_k[j_sample, :] = feature_id[min_position]   # colect all intances reuslt
            fingerprints_k[j_sample, :] = min_position
            #print('fingerprints_k = ', fingerprints_k)  # > 0
            fingerprints_y[j_sample, :] = y_matrix[min_position, np.arange(a_matrix.shape[1])]

        elapsed = time.time() - start

        return fingerprints_k, fingerprints_y, elapsed


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
        np.set_printoptions(threshold=np.inf)
        np.set_printoptions(precision=3, suppress=True, linewidth=150)

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
            #feature_id = self.weighted_set[:, j_sample]
            #print('feature_id = ', feature_id)  # > 0 
            #print('type(feature_id) = ', type(feature_id))
            #print(self.weighted_set[:, j_sample])
            #print('len(self.weighted_set[:, j_sample]) = ', len(feature_id))
            gamma = - np.log(np.multiply(u1[feature_id, :], u2[feature_id, :]))
            #print('feature_id = ', feature_id)  # you 0 hui shanchu
            #print('gamma = ', gamma)
            #print('gamma.flatten = ', gamma.flatten())
            #t_matrix = np.floor(np.divide(
            #    np.matlib.repmat(np.log(self.weighted_set[feature_id, j_sample].todense()), 1, self.dimension_num),
            #    gamma) + beta[feature_id, :])   # kafeng
            
            #print(self.dimension_num)
            oneDim = np.log(self.weighted_set[feature_id, j_sample])  # (7,)   # cpp error 
            #temp = self.weighted_set[feature_id, j_sample]  # (7,)    winder * 3 = (7, 3)
            #print('temp = ', temp)
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
            #print(np.transpose(myrepeat))

            t_matrix = np.floor(np.divide(
                #mb.repmat(np.log(self.weighted_set[feature_id, j_sample]), 1, self.dimension_num),  # myrepeat 
                np.transpose(myrepeat),    # kafeng (len(feature_id), j_sample * self.dimension_num)
                gamma) + beta[feature_id, :])         # kafeng formula 7 
            #print('t_matrix = ', t_matrix)
            #print('t_matrix.flatten = ', t_matrix.flatten())

            y_matrix = np.exp(np.multiply(gamma, t_matrix - beta[feature_id, :]))   # kafeng formula 7
            a_matrix = np.divide(-np.log(x[feature_id, :]), np.divide(y_matrix, u1[feature_id, :]))  # kafeng formula 8 ?? improve ?

            #print('a_matrix = ', a_matrix)  # nonzeros, 32
            #print(a_matrix.flatten())
            min_position = np.argmin(a_matrix, axis=0)
            #print('min_position.shape = ', min_position.shape)
            #print('len(min_position) = ', len(min_position))
            #print('min_position = ', min_position)   # ndarry  1d  > 0
            #print('type(min_position) = ', type(min_position))  # ndarray
            #print(' len(feature_id[min_position]) = ', len(feature_id[min_position]))  # ndarry 1d
            #print('feature_id[min_position] = ', feature_id[min_position])  > 0
            fingerprints_k[j_sample, :] = feature_id[min_position]   # colect all intances reuslt
            #print('fingerprints_k = ', fingerprints_k)  # > 0
            fingerprints_y[j_sample, :] = y_matrix[min_position, np.arange(a_matrix.shape[1])]

        elapsed = time.time() - start

        return fingerprints_k, fingerprints_y, elapsed

    def pcws_cpp(self, repeat=1):

        start = time.time()
        i = fingerprints_k = np.zeros((self.instance_num, self.dimension_num))
        t = fingerprints_y = np.zeros((self.instance_num, self.dimension_num))
        runtime = np.array([])

        d = self.weighted_set

        np.random.seed(self.seed * np.power(2, repeat - 1))

        beta = np.random.uniform(0, 1, (self.feature_num, self.dimension_num))  #  compress each instance
        x = np.random.uniform(0, 1, (self.feature_num, self.dimension_num))
        u1 = np.random.uniform(0, 1, (self.feature_num, self.dimension_num))
        u2 = np.random.uniform(0, 1, (self.feature_num, self.dimension_num))

        #print('beta.dtype = ', beta.dtype)

        #_doublepp = ndpointer(dtype=np.uintp, ndim=1, flags='C')  # pass error data
        #c_doulbe_p = POINTER(c_double)  # kafeng
        #c_doulbe_p = POINTER(c_double) 
        #c_doulbe_p = np.float64
        _doublepp = ndpointer(dtype=np.intp, ndim=1, flags='C')
        _dll = ctypes.cdll.LoadLibrary("WeightedMinHashToolbox/cpluspluslib/pcws_cpp.so")
        # 
        fingerprints = _dll
        fingerprints.GenerateFingerprintOfInstance.argtypes = [_doublepp,
                                                                c_int, 
                                                                _doublepp,
                                                                _doublepp,
                                                                c_int, c_int,  
                                                                np.ctypeslib.ndpointer(dtype=c_double, ndim=1,
                                                                                        flags="C_CONTIGUOUS"),
                                                                _doublepp,
                                                                _doublepp,
                                                                _doublepp,
                                                                _doublepp]
        fingerprints.GenerateFingerprintOfInstance.restype = None

        dpp = (d.__array_interface__['data'][0] + np.arange(d.shape[0])*d.strides[0]).astype(np.intp)  
        ipp = (i.__array_interface__['data'][0] + np.arange(i.shape[0])*i.strides[0]).astype(np.intp) 
        tpp = (t.__array_interface__['data'][0] + np.arange(t.shape[0])*t.strides[0]).astype(np.intp) 

        betapp = (beta.__array_interface__['data'][0] + np.arange(beta.shape[0])*beta.strides[0]).astype(np.intp) 
        xpp = (x.__array_interface__['data'][0] + np.arange(x.shape[0])*x.strides[0]).astype(np.intp) 
        u1pp = (u1.__array_interface__['data'][0] + np.arange(u1.shape[0])*u1.strides[0]).astype(np.intp) 
        u2pp = (u2.__array_interface__['data'][0] + np.arange(u2.shape[0])*u2.strides[0]).astype(np.intp) 

        fingerprints.GenerateFingerprintOfInstance(dpp, self.feature_num, ipp, tpp, self.instance_num, self.dimension_num, runtime,
                                                    betapp, xpp, u1pp, u2pp)



        #print('fingerprintsI = ', i)
        #print('len(fingerprintsI) = ', len(i))
        
        elapsed = time.time() - start

        return fingerprints_k, fingerprints_y, elapsed
        #return 0 ,0, elapsed


    def pcws_gpu(self, repeat=1):
    
        fingerprints_k = torch.zeros(self.instance_num, self.dimension_num).type(torch.FloatTensor)
        fingerprints_y = torch.zeros(self.instance_num, self.dimension_num).type(torch.FloatTensor)

        np.random.seed(self.seed * np.power(2, repeat - 1))
        start = time.time()

        #beta = torch.rand(self.feature_num, self.dimension_num).type(torch.FloatTensor)
        #x = torch.rand(self.feature_num, self.dimension_num).type(torch.FloatTensor)
        #u1 = torch.rand(self.feature_num, self.dimension_num).type(torch.FloatTensor)
        #u2 = torch.rand(self.feature_num, self.dimension_num).type(torch.FloatTensor)
        beta = np.random.uniform(0, 1, (self.feature_num, self.dimension_num))  #  compress each instance
        x = np.random.uniform(0, 1, (self.feature_num, self.dimension_num))
        u1 = np.random.uniform(0, 1, (self.feature_num, self.dimension_num))
        u2 = np.random.uniform(0, 1, (self.feature_num, self.dimension_num))

        beta = torch.tensor(beta).type(torch.FloatTensor)
        x = torch.tensor(x).type(torch.FloatTensor)
        u1 = torch.tensor(u1).type(torch.FloatTensor)
        u2 = torch.tensor(u2).type(torch.FloatTensor)

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
            min_position = torch.argmin(a_matrix, dim=0)
            fingerprints_k[j_sample, :] = torch.from_numpy(feature_id[min_position])    
            fingerprints_y[j_sample, :] = y_matrix[min_position, torch.arange(a_matrix.shape[1])]

        elapsed = time.time() - start

        return fingerprints_k, fingerprints_y, elapsed

    def pcws_pytorch(self, repeat=1):
        #fingerprints_k = np.zeros((self.instance_num, self.dimension_num))  # after compress all instances
        #fingerprints_y = np.zeros((self.instance_num, self.dimension_num))
        fingerprints_k = torch.zeros(self.instance_num, self.dimension_num).type(torch.FloatTensor)
        fingerprints_y = torch.zeros(self.instance_num, self.dimension_num).type(torch.FloatTensor)

        np.random.seed(self.seed * np.power(2, repeat - 1))
        start = time.time()

        beta = np.random.uniform(0, 1, (self.feature_num, self.dimension_num))  #  compress each instance
        x = np.random.uniform(0, 1, (self.feature_num, self.dimension_num))
        u1 = np.random.uniform(0, 1, (self.feature_num, self.dimension_num))
        u2 = np.random.uniform(0, 1, (self.feature_num, self.dimension_num))

        beta = torch.tensor(beta).type(torch.FloatTensor)
        x = torch.tensor(x).type(torch.FloatTensor)
        u1 = torch.tensor(u1).type(torch.FloatTensor)
        u2 = torch.tensor(u2).type(torch.FloatTensor)

        #print('self.feature_num = ', self.feature_num , 'self.instance_num = ', self.instance_num, 'self.dimension_num = ', self.dimension_num)
        for j_sample in range(0, self.instance_num):  # kafeng calc very instance
            
            feature_id = sparse.find(self.weighted_set[:, j_sample] > 0)[1]  # features > 0 , No.

            gamma = - torch.log(torch.mul(u1[feature_id, :], u2[feature_id, :]))

            #gamma = - np.log(np.multiply(u1[feature_id, :], u2[feature_id, :]))
            #u1_t = torch.tensor(u1[feature_id, :]).type(torch.FloatTensor)
            #u2_t = torch.tensor(u2[feature_id, :]).type(torch.FloatTensor)
            #gamma = -torch.log(torch.mul(u1_t, u2_t)).type(torch.FloatTensor)
            
            #oneDim = np.log(self.weighted_set[feature_id, j_sample])  # (7,)
            #if self.dimension_num >= 2:
            #    myrepeat = np.vstack((oneDim,oneDim))
            #    for i in range(2, self.dimension_num):
            #        myrepeat = np.vstack((myrepeat,oneDim))

            oneDim = torch.log(torch.from_numpy(self.weighted_set[feature_id, j_sample])).type(torch.FloatTensor)  # success
            myrepeat = oneDim.repeat( self.dimension_num, 1).t()

            #t_matrix = np.floor(np.divide(
            #    np.transpose(myrepeat),    
            #    gamma) + beta[feature_id, :])        

            #gamma = torch.tensor(gamma).type(torch.FloatTensor)  # success
            beta_t = torch.tensor(beta[feature_id, :]).type(torch.FloatTensor)
            #myrepeat_t = torch.tensor(np.transpose(myrepeat)).type(torch.FloatTensor)
            t_matrix = torch.floor(torch.div(
                #myrepeat_t, 
                myrepeat,    
                gamma) + beta_t) 

            #y_matrix = np.exp(np.multiply(gamma, t_matrix - beta[feature_id, :]))   
            #a_matrix = np.divide(-np.log(x[feature_id, :]), np.divide(y_matrix, u1[feature_id, :]))  

            #gamma = torch.tensor(gamma).type(torch.FloatTensor)   # success
            t_matrix = torch.tensor(t_matrix).type(torch.FloatTensor)
            #beta_t = torch.tensor(beta[feature_id, :]).type(torch.FloatTensor)
            x_t = torch.tensor(x[feature_id, :]).type(torch.FloatTensor)
            u1_t2 = torch.tensor(u1[feature_id, :]).type(torch.FloatTensor)
            y_matrix = torch.exp(torch.mul(gamma, t_matrix - beta_t))   # kafeng formula 7
            a_matrix = torch.div(-torch.log(x_t), torch.div(y_matrix, u1_t2))  # kafeng formula 8 ?? improve ?

            
            #min_position = np.argmin(a_matrix, axis=0)
            #fingerprints_k[j_sample, :] = feature_id[min_position]   # colect all intances reuslt
            #fingerprints_y[j_sample, :] = y_matrix[min_position, np.arange(a_matrix.shape[1])]
            #y_matrix = torch.tensor(y_matrix).type(torch.FloatTensor)  # success
            #a_matrix = torch.tensor(a_matrix).type(torch.FloatTensor)
            min_position = torch.argmin(a_matrix, dim=0) 
            fingerprints_k[j_sample, :] = torch.from_numpy(feature_id[min_position])    
            fingerprints_y[j_sample, :] = y_matrix[min_position, torch.arange(a_matrix.shape[1])]

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

    def pcws_fullMatrix1D(self, repeat=1):
        np.set_printoptions(threshold=np.inf)
        np.set_printoptions(precision=3, suppress=True, linewidth=150)
        #np.set_printoptions(formatter={'float': '{: 0.3f}'.format})

        start = time.time()
        print('self.instance_num = ', self.instance_num)
        print('self.feature_num = ', self.feature_num)
        print('self.dimension_num = ', self.dimension_num)
        fingerprintsI = np.zeros(self.instance_num * self.dimension_num)
        fingerprintsT = np.zeros(self.instance_num * self.dimension_num)
        runtime = np.array([])

        #print('self.weighted_set = ', self.weighted_set)
        #print(type(self.weighted_set))
        #dataset = self.weighted_set.reshape(-1)
        dataset = self.weighted_set.flatten()
        #print('dataset = ', dataset)

        np.random.seed(self.seed * np.power(2, repeat - 1))

        beta = np.random.uniform(0, 1, (self.feature_num, self.dimension_num)).flatten()  #  compress each instance
        x = np.random.uniform(0, 1, (self.feature_num, self.dimension_num)).flatten()
        u1 = np.random.uniform(0, 1, (self.feature_num, self.dimension_num)).flatten() 
        u2 = np.random.uniform(0, 1, (self.feature_num, self.dimension_num)).flatten()
        #print('u2 = ', u2)

        fingerprints = CDLL('WeightedMinHashToolbox/cpluspluslib/pcws_full1D.so')

        fingerprints.GenerateFingerprintOfInstance.argtypes = [np.ctypeslib.ndpointer(dtype=c_double, ndim=1,
                                                                                        flags="C_CONTIGUOUS"),
                                                                c_int, 
                                                                np.ctypeslib.ndpointer(dtype=c_double, ndim=1,
                                                                                        flags="C_CONTIGUOUS"),
                                                                np.ctypeslib.ndpointer(dtype=c_double, ndim=1,
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
                                                                                        flags="C_CONTIGUOUS")]
        fingerprints.GenerateFingerprintOfInstance.restype = None
        fingerprints.GenerateFingerprintOfInstance(dataset, self.feature_num, fingerprintsI, fingerprintsT,
                                                    self.instance_num, self.dimension_num, runtime, beta, x, u1, u2)   # getHash3



        #print('fingerprintsI = ', fingerprintsI)
        #print('len(fingerprintsI) = ', len(fingerprintsI))
        #print('fingerprintsT = ', fingerprintsT)
        elapsed = time.time() - start
        fingerprints_k = fingerprintsI.reshape(self.instance_num , self.dimension_num)
        fingerprints_y = fingerprintsT.reshape(self.instance_num, self.dimension_num)
        return fingerprints_k, fingerprints_y, elapsed
        #return 0 ,0, elapsed


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

    
    def ccws_pytorch(self, repeat=1, scale=1):
        fingerprints_k = torch.zeros(self.instance_num, self.dimension_num).type(torch.FloatTensor)
        fingerprints_y = torch.zeros(self.instance_num, self.dimension_num).type(torch.FloatTensor)

        np.random.seed(self.seed * np.power(2, repeat - 1))
        start = time.time()

        beta = np.random.uniform(0, 1, (self.feature_num, self.dimension_num))
        gamma = np.random.beta(2, 1, (self.feature_num, self.dimension_num))
        c = np.random.gamma(2, 1, (self.feature_num, self.dimension_num))
        beta = torch.tensor(beta).type(torch.FloatTensor)
        gamma = torch.tensor(gamma).type(torch.FloatTensor)
        c = torch.tensor(c).type(torch.FloatTensor)

        
        for j_sample in range(0, self.instance_num):
            #feature_id = sparse.find(self.weighted_set[:, j_sample] > 0)[0]  # kafeng error
            feature_id = sparse.find(self.weighted_set[:, j_sample] > 0)[1]

            oneDim = torch.tensor(self.weighted_set[feature_id, j_sample]).type(torch.FloatTensor)
            myrepeat = oneDim.repeat( self.dimension_num, 1).t()

            t_matrix = torch.floor(scale * torch.div(
                myrepeat,
                gamma[feature_id, :]) + beta[feature_id, :])
            y_matrix = torch.mul(gamma[feature_id, :], (t_matrix - beta[feature_id, :]))
            a_matrix = torch.div(c[feature_id, :], y_matrix) - 2 * torch.mul(gamma[feature_id, :], c[feature_id, :])

            min_position = torch.argmin(a_matrix, dim=0)
            fingerprints_k[j_sample, :] = torch.from_numpy(feature_id[min_position])
            fingerprints_y[j_sample, :] = y_matrix[min_position, np.arange(a_matrix.shape[1])]

        elapsed = time.time() - start

        return fingerprints_k, fingerprints_y, elapsed

    def ccws_gpu(self, repeat=1, scale=1):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        fingerprints_k = torch.zeros(self.instance_num, self.dimension_num).type(torch.cuda.FloatTensor)
        fingerprints_y = torch.zeros(self.instance_num, self.dimension_num).type(torch.cuda.FloatTensor)

        np.random.seed(self.seed * np.power(2, repeat - 1))
        start = time.time()

        beta = np.random.uniform(0, 1, (self.feature_num, self.dimension_num))
        gamma = np.random.beta(2, 1, (self.feature_num, self.dimension_num))
        c = np.random.gamma(2, 1, (self.feature_num, self.dimension_num))
        beta = torch.tensor(beta).type(torch.cuda.FloatTensor)
        gamma = torch.tensor(gamma).type(torch.cuda.FloatTensor)
        c = torch.tensor(c).type(torch.cuda.FloatTensor)

        
        for j_sample in range(0, self.instance_num):
            #feature_id = sparse.find(self.weighted_set[:, j_sample] > 0)[0]  # kafeng error
            feature_id = sparse.find(self.weighted_set[:, j_sample] > 0)[1]

            oneDim = torch.tensor(self.weighted_set[feature_id, j_sample]).type(torch.cuda.FloatTensor)
            feature_id = torch.tensor(feature_id).type(torch.cuda.LongTensor)
            myrepeat = oneDim.repeat( self.dimension_num, 1).t()

            t_matrix = torch.floor(scale * torch.div(
                myrepeat,
                gamma[feature_id, :]) + beta[feature_id, :])
            y_matrix = torch.mul(gamma[feature_id, :], (t_matrix - beta[feature_id, :]))
            a_matrix = torch.div(c[feature_id, :], y_matrix) - 2 * torch.mul(gamma[feature_id, :], c[feature_id, :])

            min_position = torch.argmin(a_matrix, dim=0)
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

