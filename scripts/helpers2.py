#!/usr/bin/env python
# coding: utf-8

import numpy as np

def split_data(x, y, ids, ratio, seed=1):
    """split the dataset based on the split ratio."""
    # set seed
    np.random.seed(seed)
    indices = np.arange(x.shape[0])
    np.random.shuffle(indices)

    n          = x.shape[0]
    index_split = int(n*ratio)
    x_train    = x[indices[0:index_split]]
    y_train    = y[indices[0:index_split]] 
    ids_train = ids[indices[0:index_split]]
    x_test     = x[indices[index_split:]]
    y_test     = y[indices[index_split:]] 
    ids_test = ids[indices[index_split:]] 
    
    return x_train, y_train, x_test, y_test, ids_train, ids_test

def standardize(x):
    """Standardize the original data set."""
    mean_x = np.mean(x)
    x = x - mean_x
    std_x = np.std(x)
    x = x / std_x
    return x, mean_x, std_x

def compute_mse(y, tx, w):
    """Calculate the mse for error vector e."""
    e = y - tx.dot(w)
    return 1/2*np.mean(e**2)


def compute_gradient(y, tx, w): 
    """Compute the gradient."""
    e = y - tx.dot(w)
    G = -tx.T.dot(e) / len(e)
    return G


def batch_iter(y, tx, batch_size=1, num_batches=1, shuffle=True):
    """
    Generate a minibatch iterator for a dataset.
    Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
    Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
    Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.
    Example of use :
    for minibatch_y, minibatch_tx in batch_iter(y, tx, 32):
        <DO-SOMETHING>
    """
    data_size = len(y)

    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]
            
def build_poly(x, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree."""
    poly = np.ones((x.shape[0], 1))
    for deg in range(1, degree+1):
        poly = np.c_[poly, np.power(x, deg)]
    return poly