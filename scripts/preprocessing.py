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


def filtering_with_mean_bis(tX, y):
    """Compute a mean of each column of the matrix tX according to ids which correpond to y=1 and another 
    mean of each column according to ids which correpond to y=-1 without taking the -999. Then, these means 
    replace the -999 according to the y of the row where the -999 is. It returns the filtered matrix"""
    index = np.arange(tX.shape[1])
    tX_filtered = np.copy(tX)
    
    ind_1 = np.where(y == 1)[0]
    ind_2 = np.where(y == -1)[0]
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