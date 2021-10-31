#!/usr/bin/env python
# coding: utf-8

import numpy as np
from implementations import *
from proj1_helpers import *
from helpers2 import *
from preprocessing import * 
from cross_validation import *

print('You are running the script to generate the prediction of our best model : Ridge regression')


#### DOWNLOAD DATA ####

DATA_TRAIN_PATH = "../data/train.csv"
y, tX, ids = load_csv_data(DATA_TRAIN_PATH)


#### PREPROCESSING ####

print('\nThe preprocessing consists of separating the train data into 3 sets according to PRI_jet_num feature and running a Ridge regression for each set')
# Separation into sets according to PRI_jet_num feature
set1_x, set1_y, set1_ids, set2_x, set2_y, set2_ids, set3_x, set3_y, set3_ids = separate_sets(tX, y, ids)

# The filtering method that gives the best results for Ridge regression
def best_filtering_rr(set_x, set_y) :
        set_x = outliers(set_x, -999)
        set_x = filtering_with_mean_bis(set_x, set_y)
        return set_x
    
print('\nSet 1 : Preprocessing')
set1_x_rr = best_filtering_rr(set1_x, set1_y)

print('\nSet 2 : Preprocessing')
set2_x_rr = best_filtering_rr(set2_x, set2_y)

print('\nSet 3 : Preprocessing')
set3_x_rr = best_filtering_rr(set3_x, set3_y)


#### BEST MODEL APPLICATION : RIDGE REGRESSION #####


# Ridge regression of set 1
print('\nSet 1 : Ridge regression')
# Cross-validation on the degrees and the lambdas
degree_rr_set1, lambda_rr_set1 = best_degree_selection(set1_y, set1_x_rr, degrees=np.arange(1,8), k_fold=4, lambdas=np.logspace(-4, 0, 30), fonction=1)
print("Cross validation finished: optimal lambda {l} and degree {d}".format(l=lambda_rr_set1, d=degree_rr_set1))
# Best degree and lambda model
set1_x_rr = build_poly(set1_x_rr, degree_rr_set1)
w_rr_set1, loss_rr_set1 = ridge_regression(set1_y, set1_x_rr, lambda_rr_set1)
print("Ridge regression loss {loss}".format(loss=loss_rr_set1))


# Ridge regression of set 2
print('\nSet 2 : Ridge regression')
# Cross-validation on the degrees and the lambdas
degree_rr_set2, lambda_rr_set2 = best_degree_selection(set2_y, set2_x_rr, degrees=np.arange(1,8), k_fold=4, lambdas=np.logspace(-4, 0, 30), fonction=1)
print("Cross validation finished: optimal lambda {l} and degree {d}".format(l=lambda_rr_set2, d=degree_rr_set2))
# Best degree and lambda model
set2_x_rr = build_poly(set2_x_rr, degree_rr_set2)
w_rr_set2, loss_rr_set2 = ridge_regression(set2_y, set2_x_rr, lambda_rr_set2)
print("Ridge regression loss {loss}".format(loss=loss_rr_set2))


# Ridge regression of set 3
print('\nSet 3 : Ridge regression')
# Cross-validation on the degrees and the lambdas
degree_rr_set3, lambda_rr_set3 = best_degree_selection(set3_y, set3_x_rr, degrees=np.arange(1,8), k_fold=4, lambdas=np.logspace(-4, 0, 30), fonction=1)
print("Cross validation finished: optimal lambda {l} and degree {d}".format(l=lambda_rr_set3, d=degree_rr_set3))
# Best degree and lambda model
set3_x_rr = build_poly(set3_x_rr, degree_rr_set3)
w_rr_set3, loss_rr_set3 = ridge_regression(set3_y, set3_x_rr, lambda_rr_set3)
print("Ridge regression loss {loss}".format(loss=loss_rr_set3))


#### CREATE SUBMISSION ####

print('\nThe csv file best_pred.csv is being prepared...')
OUTPUT_PATH = '../data/best_pred.csv' 
DATA_TEST_PATH = '../data/test.csv'
y_test , tX_test, ids_test = load_csv_data(DATA_TEST_PATH)

set1_x, _, set1_ids, set2_x, _, set2_ids, set3_x, _, set3_ids = separate_sets(tX_test, y_test, ids_test)

def filtering_test_rr (set_x, degree_rr):
    set_x = outliers(set_x, -999)
    set_x = build_poly(set_x, degree_rr)
    return set_x

set1_x = filtering_test_rr(set1_x, degree_rr_set1)
set2_x = filtering_test_rr(set2_x, degree_rr_set2)
set3_x = filtering_test_rr(set3_x, degree_rr_set3)
y_pred1 = predict_labels(w_rr_set1, set1_x)
y_pred2 = predict_labels(w_rr_set2, set2_x)
y_pred3 = predict_labels(w_rr_set3, set3_x)

y_pred_rr, ids_rr = concatenate_sets(y_pred1, set1_ids, y_pred2, set2_ids, y_pred3, set3_ids)
create_csv_submission(ids_rr, y_pred_rr, OUTPUT_PATH)
print('\nThe run is finished. best_pred.csv is ready and it is located in the data folder which is on the same level as the scripts folder.')