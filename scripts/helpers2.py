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
            
def build_poly(tx, degree):
    """polynomial basis functions for input data matrix tx, for j=0 up to j=degree."""
    poly = np.ones(tx.shape[0])
    for deg in range(1, degree+1):
        poly = np.c_[poly, np.power(tx, deg)]
    return poly

def sigmoid(t):
    """apply the sigmoid function on t."""
    sig = (1 + np.exp(-t))**(-1)
    return sig

def calculate_loss(y, tx, w):
    """compute the loss: negative log likelihood."""
    
    """print('TEST1')
    print(np.shape(y))
    print(np.shape(y.T))
    print(np.shape(tx))
    print(np.shape(w))
    y = np.reshape(y, (-1, 1))
    print('TEST2')
    print(np.shape(y))
    print(np.shape(y.T))"""
    
    #y = np.reshape(y, (-1,1))
    pred = sigmoid(tx.dot(w))
    #pred = sigmoid(y.T.dot(tx).dot(w))
    #loss = np.log(pred)
    #loss = y.T.dot(np.log(pred)) + (1 - y).T.dot(np.log(1 - pred))
    loss = np.sum(np.log(1 + np.exp(tx.dot(w))) - y.dot(tx.dot(w)))
    #return np.squeeze(- loss)
    return loss

    
def calculate_gradient(y, tx, w):
    """compute the gradient of loss."""
    y = np.reshape(y, (-1, 1))
    grad = tx.T.dot(1/(1 + np.exp(-tx.dot(w))) - y)
    #grad = tx.T.dot(sigmoid(tx.dot(w))-y)
    #grad = -1*np.shape(y) + sigmoid(y.T.dot(tx).dot(w))
    return grad

def calculate_hessian(y, tx, w):
    """return the Hessian of the loss function."""
    pred = sigmoid(tx.dot(w))
    pred = np.diag(pred.T[0])
    r = np.multiply(pred, (1-pred))
    return tx.T.dot(r).dot(tx)

def learning_by_gradient_descent(y, tx, w, gamma):
    """
    Do one step of gradient descent using logistic regression.
    Return the loss and the updated w.
    """
    loss = calculate_loss(y, tx, w)
    grad = calculate_gradient(y, tx, w)
    w = w - gamma*grad
    return loss, w

def penalized_logistic_regression(y, tx, w, lambda_):
    """return the loss, gradient"""
    num_samples = y.shape[0]
    loss = calculate_loss(y, tx, w) + lambda_ * np.squeeze(w.T.dot(w))
    grad = calculate_gradient(y, tx, w) + 2 * lambda_ * w
    hess = calculate_hessian(y, tx, w) + 2 * lambda_
    return loss, grad, hess

def learning_by_penalized_gradient(y, tx, w, gamma, lambda_):
    """
    Do one step of gradient descent, using the penalized logistic regression.
    Return the loss and updated w.
    """
    loss, grad, hess = penalized_logistic_regression(y, tx, w, lambda_)
    w = w - gamma * grad
    return loss, w