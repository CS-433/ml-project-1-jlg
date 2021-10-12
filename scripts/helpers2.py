#!/usr/bin/env python
# coding: utf-8

def compute_mse_loss(y, tx, w):
    """Calculate the mse for error vector e."""
    e = y - tx.dot(w)
    return 1/2*np.mean(e**2)

def compute_gradient(y, tx, w): 
    """Compute the gradient."""
    e = y - tx.dot(w)
    G = -tx.T.dot(e) / len(e)
    return G, e

