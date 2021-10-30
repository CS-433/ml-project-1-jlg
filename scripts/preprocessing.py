import numpy as np

def filtering_with_mean(tX):
    """Replace the -999 of each column of the matrix tX by the mean of the column without the -999"""
    index = np.arange(tX.shape[1])
    tX_filtered = np.copy(tX)
    arr = []
    for ind in index :
        arr = np.delete(tX_filtered[:,ind], np.where(tX_filtered[:,ind]==-999))
        mean = np.mean(arr)
        tX_filtered[np.where(tX_filtered[:,ind]==-999), ind] = mean
    return tX_filtered


def filtering_with_mean_bis(tX, y, lr=0):
    """Compute a mean of each column of the matrix tX according to ids which correpond to y=1 and another 
    mean of each column according to ids which correpond to y=-1 without taking the -999. Then, these means 
    replace the -999 according to the y of the row where the -999 is. It returns the filtered matrix"""
    index = np.arange(tX.shape[1])
    tX_filtered = np.copy(tX)
    
    ind_1 = np.where(y == 1)[0]
    if lr == 0: ind_2 = np.where(y == -1)[0]
    if lr == 1: ind_2 = np.where(y == 0)[0]
    tX_1 = tX[ind_1,:]
    tX_2 = tX[ind_2,:]
    
    ind_3 = np.where(tX[:,0]==-999)[0]
    new_ind_1 = np.intersect1d(ind_3, ind_1)
    new_ind_2 = np.intersect1d(ind_3, ind_2)
    
    arr_1 = []
    arr_2 = []
    for ind in index :
        arr_1 = np.delete(tX_1[:,ind], np.where(tX_1[:,ind]==-999))
        mean_1 = np.mean(arr_1)
        arr_2 = np.delete(tX_2[:,ind], np.where(tX_2[:,ind]==-999))
        mean_2 = np.mean(arr_2)
        tX_filtered[new_ind_1, ind] = mean_1
        tX_filtered[new_ind_2, ind] = mean_2
    return tX_filtered

def std(tX):
    """Standardize each column of the matrix tX if the standard deviation is bigger than 0"""
    mean = np.mean(tX, axis = 0)
    std = np.std(tX, axis = 0)
    tX[:, std>0] = (tX[:, std>0] - mean[std>0])/std[std>0]
    return tX

def cut(tX, to_cut):
    """Remove columns of the matrix tX whose index are given in the array to_cut as parameters"""
    cut_index = 100*np.ones(tX.shape[1])
    index_full = np.arange(tX.shape[1])
    for i in range(tX.shape[1]):
        for j in range(len(to_cut)):
            if index_full[i] == to_cut[j]:
                cut_index[i] = to_cut[j]
    index = index_full[~(index_full == cut_index)]
    index = index.reshape(-1)
    tX_cut = tX[:, index]
    return tX_cut

def keep(tX, to_keep):
    """Keep only the columns of the matrix tX whose index are given in the array to_keep as parameters"""
    keep_index = 100*np.ones(tX.shape[1])
    index_full = np.arange(tX.shape[1])
    for i in range(tX.shape[1]):
        for j in range(len(to_keep)):
            if index_full[i] == to_keep[j]:
                keep_index[i] = to_keep[j]
    index = index_full[index_full == keep_index]
    index = index.reshape(-1)
    tX_kept = tX[:, index]
    return tX_kept

def log_distribution(tX, to_log):
    tX_log = np.copy(tX)
    index = np.arange(tX.shape[1])
    for i in range(tX.shape[1]):
        for j in range(len(to_log)):
            if index[i] == to_log[j]:
                tX_log[:, i] = np.log(1+tX[:, to_log[j]], where=np.all(tX[:, i]>0))  
    return tX_log

def separate_sets(tX, y, ids, col=22):
    index1 = np.where(tX[:, col]==0)
    index2 = np.where(tX[:, col]==1)
    index3 = np.where(tX[:, col]>1)
    
    index = np.arange(0, tX.shape[1])
    tX = tX[:, ~(index==col)] # delete the 22nd column of the original dataset
    
    set1_x = tX[index1]
    set1_x = set1_x[:, ~(np.all(set1_x==set1_x[0], axis = 0))] # delete possible columns with constant value
    set1_y = y[index1]
    set1_ids = ids[index1]
    
    set2_x = tX[index2]
    set2_x = set2_x[:, ~(np.all(set2_x==set2_x[0], axis = 0))] # delete possible columns with constant value
    set2_y = y[index2]
    set2_ids = ids[index2]
    
    set3_x = tX[index3]
    set3_x = set3_x[:, ~(np.all(set3_x==set3_x[0], axis = 0))] # delete possible columns with constant value
    set3_y = y[index3]
    set3_ids = ids[index3]
    
    return set1_x, set1_y, set1_ids, set2_x, set2_y, set2_ids, set3_x, set3_y, set3_ids

def concatenate_sets(set1_y, set1_ids, set2_y, set2_ids, set3_y, set3_ids):
    y = np.concatenate((set1_y, set2_y, set3_y), axis = 0)
    ids = np.concatenate((set1_ids, set2_ids, set3_ids), axis = 0)
    return y, ids

def outliers(tX, outlier):
    outliers = []
    M = np.squeeze(tX.shape[0])
    for col in range(tX.shape[1]) :
        out_col = np.nonzero(tX[:,col] == outlier)[0].shape
        out_col = np.squeeze(out_col)
        outliers.append(out_col/M)
    print('outliers ratio for each feature', outliers)
    
    index_full = np.arange(tX.shape[1])
    index = index_full[~(outliers==np.ones(len(outliers)))]
    index = index.reshape(-1)
    X_without_outliers = tX[:, index]
    return X_without_outliers 