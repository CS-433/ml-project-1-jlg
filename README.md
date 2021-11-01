# ml-project-1-jlg_project1_ml

The goal of this project is to use regression tasks to predict if the decay signature of a collision event is a Higgs boson (signal) or something else (background) based on some features representing the decay signature. Only numpy and matplotlib libraries are allowed.

Our repository is composed :
- Scripts folder : It contains the code which is organized in one jupyter notebook which contains the best results for each regression function and different scripts of functions. implementations.py contains the regression functions, preprocessing.py contains all functions to preprocess the original matrix, cross_validation.py has different cross-validated and best parameter selection methods, proj1_helpers.py and helpers2.py contain different helper functions, run.py contain the code for the submission of our best result.
- Data folder : It contains the training data that we used to find relevant models, the test data that we used to do our AIcrowd submissions after selecting the best predicted models on the training data. It will also contain the best submission after running run.py script. Be aware that you have to unzip the training and the test data before using it.

Our results can be reproduced by using Python 3.8 and by running in the terminal in the folder where run.py is (scripts folder) the command : python run.py


