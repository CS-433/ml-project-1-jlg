#!/usr/bin/env python
# coding: utf-8

from implementions import *
import numpy as np
from proj1_helpers import *
from helpers2 import *
from preprocessing import * 

#### DOWNLOAD DATA ####
DATA_TRAIN_PATH = "../data/train.csv"
y, tX, ids = load_csv_data(DATA_TRAIN_PATH)

#### PREPROCESSING ####


#### BEST MODEL APPLICATION #####

#### Least squares ####

#### Ridge regression ####

#### Gradient descent ####

#### Stochastic gradient descent ####

#### Logistic regression ####

#### Logistic #####



#### CREATE SUBMISSION ####
DATA_TEST_PATH = '../data/test.csv' # TODO: download train data and supply path here 
_, tX_test, ids_test = load_csv_data(DATA_TEST_PATH)


OUTPUT_PATH = '../data/best_pred_JLG.csv'
create_csv_submission(ids_test, y_pred, OUTPUT_PATH)