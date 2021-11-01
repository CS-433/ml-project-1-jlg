#!/usr/bin/env python
# coding: utf-8

import numpy as np
from helpers2 import *


def least_squares(y, tx):
    """Least squares regression using normal equations"""
    w = np.linalg.solve(tx.T.dot(tx), tx.T.dot(y))
    loss = compute_mse(y, tx, w)
    return w, loss


def ridge_regression(y, tx, lambda_):
    """Ridge regression using normal equations"""
    lambda_2 = lambda_*2*tx.shape[0]
    I = np.identity(tx.shape[1])
    w = np.linalg.solve(tx.T.dot(tx) + I.dot(lambda_2), tx.T.dot(y))
    loss = compute_mse(y, tx, w)
    return w, loss


def least_squares_GD(y, tx, gamma, max_iters=50):
    """Linear regression using gradient descent"""
    # GD donne loss et w
    w=np.zeros(tx.shape[1])
    for n_iter in range(max_iters):
        # compute gradient and loss
        grad = compute_gradient(y, tx, w)
        loss = compute_mse(y, tx, w)
        # update w by gradient
        w = w - gamma*grad
        print("Gradient Descent({bi}/{ti}): loss={l}".format(bi=n_iter, ti=max_iters - 1, l=loss))
    return w, loss


def least_squares_SGD(y, tx,  gamma, max_iters=50):
    """Linear regression using stochastic gradient descent"""
    w=np.zeros(tx.shape[1])
    for n_iter in range(max_iters):
        for minibatch_y, minibatch_tx in batch_iter(y, tx):
            # compute gradient and loss
            G = compute_gradient(minibatch_y, minibatch_tx, w)
            loss = compute_mse(y, tx, w)
            # update w by gradient
            w = w - gamma*G
        print("SGD({bi}/{ti}): loss={l}".format(
              bi=n_iter, ti=max_iters - 1, l=loss))
    return w, loss


def logistic_regression(y, tx, initial_w=0, max_iters=10, gamma=0.01):
    """Logistic regression using gradient descent or SGD"""
    threshold = 1e-8
    losses = []
    tx = np.c_[np.ones((y.shape[0], 1)), tx]
    w = np.zeros((tx.shape[1], 1))

    for iter in range(max_iters):
        # get loss and update w
        loss, w = learning_by_gradient_descent(y, tx, w, gamma)
        if iter % 2 == 0:
            print("Current iteration={i}, loss={l}".format(i=iter, l=loss))
        # converge criterion
        losses.append(loss)
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break
    print("loss={l}".format(l=calculate_loss(y, tx, w)))
    return w, loss


def reg_logistic_regression(y, tx, lambda_=0.1, initial_w=0, max_iters=5, gamma=0.01):
    """Logistic regression using gradient descent or SGD"""
    threshold = 1e-8
    losses = []
    tx = np.c_[np.ones((y.shape[0], 1)), tx]
    if np.all(initial_w==0):
        initial_w = np.zeros((tx.shape[1], 1))
    w = initial_w

    # start the logistic regression
    for iter in range(max_iters): 
        # get loss and update w.
        loss, w = learning_by_penalized_gradient(y, tx, w, gamma, lambda_)
        if iter % 2 == 0:
            print("Current iteration={i}, loss={l}".format(i=iter, l=loss))
        # converge criterion
        losses.append(loss)
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break
    print("loss={l}".format(l=calculate_loss(y, tx, w)))
    return w, loss

