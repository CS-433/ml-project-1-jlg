#!/usr/bin/env python
# coding: utf-8

import numpy as np
from helpers2 import *


def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    """Linear regression using gradient descent"""
    # GD donne loss et w 
    w = initial_w
    for n_iter in range(max_iters):
        # compute gradient and loss
        grad = compute_gradient(y, tx, w)
        loss = compute_loss(y, tx, w)
        # update w by gradient
        w = w - gamma*grad
        print("Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1 {w1}".format(
              bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))
    return w, loss


def least_squares_SGD(y, tx, initial_w, max_iters, gamma):
    """Linear regression using stochastic gradient descent"""
    w = initial_w
    for n_iter in range(max_iters):
        for minibatch_y, minibatch_tx in batch_iter(y, tx):
            G = compute_stoch_gradient(minibatch_y, minibatch_tx, w)
            loss = compute_loss(y, tx, w)
            w = w - gamma*G
        
        print("SGD({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
              bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))
    return w, loss


def least_squares(y, tx):
    """Least squares regression using normal equations"""
    w_star = np.linalg.solve(tx.T.dot(tx), tx.T.dot(y))
    loss = compute_mse(y, tx, w_star)
    return w, loss


def ridge_regression(y, tx, lambda_):
    """Ridge regression using normal equations"""
    N = tx.shape[1]
    lamb = 2*N*lambda_*np.identity(N)
    a = tx.T.dot(tx) + lamb
    b = tx.T.dot(y)
    w = np.linalg.solve(a, b)
    
    e = y - tx.dot(w)
    loss = e.dot(e) / (2 * len(e))
    return w, loss


def logistic_regression(y, tx, initial_w, max iters, gamma):
    """Logistic regression using gradient descent or SGD"""
    return w, loss


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """Logistic regression using gradient descent or SGD"""
    return w, loss

