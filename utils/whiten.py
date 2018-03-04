import numpy as np
import pandas as pd

def zca_whitening_matrix(X):
    '''
    input: X:[M x N] matrix, needn't sub means, implented by np.conv, 
            accroding to the cov formular, mean is the image mean, 
            not the dataset's mean, shold be unified?
            M: image pixel numer
            N: sample num
    output: ZCAMatrix [M x M] matrix
    '''
    # covariance matrix, sigma = 1/N * \sum (X_i -mu)*(X_i-mu)' 
    sigma = np.cov(X, rowvar=True)
    # Singular Value Decomposition. X = U * np.diag(S) * V
    # u: [M x M], eigenvector of sigma
    # s: [M], eigenvalues of sigma  
    # v: [N x N], transpose of u  
    u,s,v = np.linalg.svd(sigma)
    # print(s)
    # regular part, prevent div zero
    epsilon = 0.1
    # zca whitening u * lambda * u' X
    zca_matrix = np.dot( u, np.dot( 1.0 / np.sqrt(np.diag(s)+epsilon), u.T )) 
    
    return zca_matrix

def zca_com(data, path):
    '''
    save zca matrix to path
    '''
    # data['band_1'] = data['band_1'].apply(lambda x: np.array(x).reshape(75,75))
    # data['band_2'] = data['band_2'].apply(lambda x: np.array(x).reshape(75,75))
    # 1604x75x75
    band_1_tr = np.concatenate([im for im in data['band_1']]).reshape(-1, 75*75)
    band_2_tr = np.concatenate([im for im in data['band_2']]).reshape(-1, 75*75)
    band_3_tr = (band_1_tr + band_2_tr) / 2

    print('band_1_tr', band_1_tr.shape)
    zca_1 = zca_whitening_matrix(band_1_tr.transpose(1, 0))
    zca_2 = zca_whitening_matrix(band_2_tr.transpose(1, 0))
    zca_3 = zca_whitening_matrix(band_3_tr.transpose(1, 0))
    
    np.save(path+'zca_1.npy', zca_1)  # 5625*5625 matrix? too large
    np.save(path+'zca_2.npy', zca_2)  # result pixel is enlarged?
    np.save(path+'zca_3.npy', zca_3)
    print('zca.shape', zca_1.shape)

def zca_transfer(data, path):
    '''
    whiten dataset
    '''
    zca_1 = np.load(path+'zca_1.npy')
    zca_2 = np.load(path+'zca_2.npy')
    data['band_1'] = data['band_1'].apply(lambda x: np.dot(zca_1, np.array(x).reshape(75*75)))
    data['band_2'] = data['band_2'].apply(lambda x: np.dot(zca_2, np.array(x).reshape(75*75)))
    
    data.to_json(path+'train_zca.json')

if __name__ == '__main__':
    path = '/home/lxg/codedata/ice/'
    data = pd.read_json(path+'train.json')
    # zca_com(data, path)
    zca_transfer(data, path)

