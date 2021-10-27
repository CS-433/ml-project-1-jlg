from implementations import *
import numpy as np
from helpers2 import *


def build_k_indices(y, k_fold, seed=1):
    """build k indices for k-fold."""
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval]
                 for k in range(k_fold)]
    return np.array(k_indices)


def k_fold_regression(y, x, k_indices, k, par, degree=0, fonction=1):
    """return the loss for k_fold cross validation.
    default regression is ridge 
    set fonction=0 for least squares with normal equations
        fonction=1 for ridge regression
        fonction=2 for gradient descent
        fonction=3 for stochastic gradient decsent"""
    
    # get k'th subgroup in test, others in train: 
    te_indice = k_indices[k]
    tr_indice = k_indices[~(np.arange(k_indices.shape[0]) == k)]
    tr_indice = tr_indice.reshape(-1)
    y_te = y[te_indice]
    y_tr = y[tr_indice]
    x_te = x[te_indice]
    x_tr = x[tr_indice]

    # form data with polynomial degree
    if degree==0:
        tx_tr = x_tr
        tx_te = x_te
    else: 
        tx_tr = build_poly(x_tr, degree)
        tx_te = build_poly(x_te, degree)
    
    # Regression
    if fonction == 0:
        w, loss_tr = least_squares(y_tr, tx_tr)
    elif fonction == 1: 
        w, loss_tr = ridge_regression(y_tr, tx_tr, par)
    elif fonction == 2: 
        w, loss_tr = least_squares_GD(y_tr, tx_tr, par)
    elif fonction == 3: 
        w, loss_tr = least_squares_SGD(y_tr, tx_tr, par)
    elif fonction == 4:
        w, loss_tr = logistic_regression(y=y_tr, tx=tx_tr, gamma=par)
    
    # calculate the loss for train and test data: 
    loss_tr = np.sqrt(2*loss_tr)
    loss_te = np.sqrt(2*compute_mse(y_te, tx_te, w))
    
    return loss_tr, loss_te


def cross_validation(y, x, k_fold, parameters, degree=0, fonction=1, seed=1):
    """returns the result of a k_fold cross_validation 
    on a certain parameter. """
    # split data in k fold
    k_indices = build_k_indices(y, k_fold, seed)
    # define lists to store the loss of training data and test data
    rmse_tr = []
    rmse_te = []
    for ind, par in enumerate(parameters):
        loss_tr = 0
        loss_te = 0
        for k in range(k_fold):
            loss_tr_k, loss_te_k = k_fold_regression(y, x, k_indices, k, par, fonction=fonction)
            loss_tr = loss_tr + loss_tr_k
            loss_te = loss_te + loss_te_k
        rmse_tr.append(loss_tr/k_fold)
        rmse_te.append(loss_te/k_fold)
        
    best_par = parameters[np.argmin(rmse_te)]
    return best_par


def best_degree_selection(y, x, degrees, k_fold, lambdas, fonction=1, seed=1):
    """return the best degree and lambda association."""
    # split data in k fold
    k_indices = build_k_indices(y, k_fold, seed)
    
    #for each degree, we compute the best lambdas and the associated rmse
    best_lambdas = []
    best_rmses = []
    
    #vary degree
    for degree in degrees:
        rmse_te = []
        
        if np.all(lambdas == 0): #we do not have to find the best lambda
            rmse_te_tmp = []
            for k in range(k_fold):
                _, loss_te = k_fold_regression(y, x, k_indices, k, lambdas, degree, fonction)
                rmse_te_tmp.append(loss_te)
            best_rmses.append(np.mean(rmse_te_tmp))
        
        else: # vary lambda
            for lambda_ in lambdas:
                rmse_te_tmp = []
                for k in range(k_fold):
                    _, loss_te = k_fold_regression(y, x, k_indices, k, lambda_, degree, fonction)
                    rmse_te_tmp.append(loss_te)
                rmse_te.append(np.mean(rmse_te_tmp))
        
            ind_lambda_opt = np.argmin(rmse_te)
            best_lambdas.append(lambdas[ind_lambda_opt])
            best_rmses.append(rmse_te[ind_lambda_opt])
    
    print("best rmses", best_rmses)
    best_degree =  degrees[np.argmin(best_rmses)]  
    if np.all(lambdas == 0):
        best_lambda = 0
    else: best_lambda =  lambdas[np.argmin(best_rmses)]
    
    return  best_degree, best_lambda