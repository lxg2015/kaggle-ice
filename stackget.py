import os
import pandas as pd
import numpy as np

path = os.path.expanduser('~/codedata/ice/vggbn/')
best_base = pd.read_csv('/home/lxg/codedata/ice/submit_get_stack_0.13.csv', index_col=0)

all_files = os.listdir(path)
file_num = len(all_files)
print(len(all_files), all_files)
outs = [pd.read_csv(path+f, index_col=0) for f in all_files ]
outs.append(best_base)

concat_sub = pd.concat(outs, axis=1)
concat_sub.reset_index(inplace=True)

cols = list(map(lambda x: 'is_iceberg_'+str(x)+all_files[x], range(len(outs)-1)))
cols.insert(0,'id')
cols.append('is_iceberg_base')
concat_sub.columns = cols
print(concat_sub.head())

concat_sub['is_iceberg_max'] = concat_sub.iloc[:, 1:25].max(axis=1)
concat_sub['is_iceberg_min'] = concat_sub.iloc[:, 1:25].min(axis=1)
concat_sub['is_iceberg_mean'] = concat_sub.iloc[:, 1:25].mean(axis=1)
concat_sub['is_iceberg_median'] = concat_sub.iloc[:, 1:25].median(axis=1)

# set up cutoff threshold for lower and upper bounds, easy to twist 
cutoff_lo = 0.6 #0.7
cutoff_hi = 0.4 #0.3

# concat_sub['is_iceberg'] = np.where(np.all(concat_sub.iloc[:,1:25] > cutoff_lo, axis=1), concat_sub['is_iceberg_max'],
#                                     np.where(np.all(concat_sub.iloc[:,1:25] < cutoff_hi, axis=1), concat_sub['is_iceberg_min'],
#                                     0.3*concat_sub['is_iceberg_median']+0.7*concat_sub['is_iceberg_base']))
concat_sub['is_iceberg'] = np.where(np.array(concat_sub.iloc[:,1:25] > cutoff_lo).sum(axis=1) > 0.8*25, concat_sub['is_iceberg_max'],
                                    np.where(np.array(concat_sub.iloc[:,1:25] < cutoff_hi).sum(axis=1) > 0.8*25, concat_sub['is_iceberg_min'],
                                    0.3*concat_sub['is_iceberg_median']+0.7*concat_sub['is_iceberg_base']))
concat_sub['is_iceberg'] = np.clip(concat_sub['is_iceberg'].values, 0.001, 0.999)
concat_sub[['id', 'is_iceberg']].to_csv('submit_get_stack.csv', index=False)
