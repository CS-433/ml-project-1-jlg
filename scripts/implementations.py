#!/usr/bin/env python
# coding: utf-8

from helpers2 import *

def gradient_descent(y, tx, initial_w, max_iters, gamma):
    """Gradient descent algorithm."""
    ws = [initial_w]
    losses = []
    w = initial_w
    for n_iter in range(max_iters):
        # compute gradient and loss
        grad = compute_gradient(y, tx, w)
        loss = compute_loss(y, tx, w)
        # update w by gradient
        w = w - gamma*grad
        # store w and loss
        ws.append(w)
        losses.append(loss)
        print("Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
              bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))
    return losses, ws

def stochastic_gradient_descent(y, tx, initial_w, max_iters, gamma):
    """Stochastic gradient descent algorithm.""".
    ws = [initial_w]
    losses = []
    w = initial_w
    for n_iter in range(max_iters):
        for minibatch_y, minibatch_tx in batch_iter(y, tx, 1):
            G = compute_stoch_gradient(minibatch_y, minibatch_tx, w)
            loss = compute_loss(y, tx, w)
            w = w - gamma*G

        ws.append(w)
        losses.append(loss)
        
        print("SGD({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
              bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))
    return losses, ws


def least squares GD(y, tx, initial w, max_iters, gamma):
    """Linear regression using gradient descent"""
    # GD donne loss et w 
    return w, loss


def least squares SGD(y, tx, initial w, max_iters, gamma):
    """Linear regression using stochastic gradient descent"""
    return w, loss


def least squares(y, tx):
    """Least squares regression using normal equations"""
    w_star = np.linalg.solve(tx.T.dot(tx), tx.T.dot(y))
    loss = compute_mse(y, tx, w_star)
    return w, loss


def ridge regression(y, tx, lambda_):
    """Ridge regression using normal equations"""
    N = tx.shape[1]
    lamb = 2*N*lambda_*np.identity(N)
    a = tx.T.dot(tx) + lamb
    b = tx.T.dot(y)
    w = np.linalg.solve(a, b)
    
    e = y - tx.dot(w)
    loss = e.dot(e) / (2 * len(e))
    return w, loss


def logistic regression(y, tx, initial w, max iters, gamma):
    """Logistic regression using gradient descent or SGD"""
    return w, loss


def reg logistic regression(y, tx, lambda_, initial w, max iters, gamma):
    """Logistic regression using gradient descent or SGD"""
    return w, loss

