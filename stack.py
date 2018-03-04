import os
import pandas as pd
import numpy as np

cutoff_lo = 0.8
cutoff_hi = 0.2

path = os.path.expanduser('~/codedata/ice/')
s_big_0 = pd.read_csv(path+'submit_0_big.csv')
s_big_1 = pd.read_csv(path+'submit_1_big.csv')
s_big_2 = pd.read_csv(path+'submit_2_big.csv')
s_big_3 = pd.read_csv(path+'submit_3_big.csv')
big = pd.concat([s_big_0, s_big_1['is_iceberg'], s_big_2['is_iceberg'], s_big_3['is_iceberg']], axis=1)
# get the data fields ready for stacking
big['is_iceberg_max'] = big.iloc[:, 1:4].max(axis=1)
big['is_iceberg_min'] = big.iloc[:, 1:4].min(axis=1)
big['is_iceberg_mean'] = big.iloc[:, 1:4].mean(axis=1)
big['is_iceberg_median'] = big.iloc[:, 1:4].median(axis=1)

big['is_iceberg_0'] = np.where(np.all(big.iloc[:,1:4] > cutoff_lo, axis=1), 
                                    big['is_iceberg_max'], 
                                    np.where(np.all(big.iloc[:,1:4] < cutoff_hi, axis=1),
                                    big['is_iceberg_min'], 
                                    big['is_iceberg_median']))

s_small_0 = pd.read_csv(path+'submit_0_small.csv')
s_small_1 = pd.read_csv(path+'submit_1_small.csv')
s_small_2 = pd.read_csv(path+'submit_2_small.csv')
s_small_3 = pd.read_csv(path+'submit_3_small.csv')
small = pd.concat([s_small_0, s_small_1['is_iceberg'], s_small_2['is_iceberg'], s_small_3['is_iceberg']], axis=1)

# get the data fields ready for stacking
small['is_iceberg_max'] = small.iloc[:, 1:4].max(axis=1)
small['is_iceberg_min'] = small.iloc[:, 1:4].min(axis=1)
small['is_iceberg_mean'] = small.iloc[:, 1:4].mean(axis=1)
small['is_iceberg_median'] = small.iloc[:, 1:4].median(axis=1)

small['is_iceberg_0'] = np.where(np.all(small.iloc[:,1:4] > cutoff_lo, axis=1), 
                                    small['is_iceberg_max'], 
                                    np.where(np.all(small.iloc[:,1:4] < cutoff_hi, axis=1),
                                    small['is_iceberg_min'], 
                                    small['is_iceberg_median']))

submit = pd.concat([big[['id', 'is_iceberg_0']], small[['id','is_iceberg_0']]])

submit.to_csv(path+'submit_stack.csv', index=False, header=['id', 'is_iceberg'])