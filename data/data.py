# https://www.kaggle.com/nanigans/pytorch-starter/notebook
import os
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1604 item total
# band_1,2 id, inc_angle, is_ice
path = '/home/lxg/codedata/ice/'
data = pd.read_json(os.path.join(path, 'train.json'))

data['band_1'] = data['band_1'].apply(lambda x: np.array(x).reshape(75,75))
data['band_2'] = data['band_2'].apply(lambda x: np.array(x).reshape(75,75))
# band_1 min-34.715858, max3.98
# band_2 min-35.403362, max-6.934982

data['inc_angle'] = pd.to_numeric(data['inc_angle'], errors='coerce')  # lack data is filled with na
# inc_angle 1604, 1471 notnan 133 nan, min24.75, max45.9, mean39.26
# 753 True, 851 False
# split
# train = data.sample(frac=0.8)
# val = data[~data.isin(train)].dropna()

def plotSample(df, idx):
    c = ('ship', 'ice')
    f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2)
    ax1.imshow(df['band_1'].iloc[idx])
    ax2.imshow(df['band_2'].iloc[idx])
    ax3.hist(df['band_1'].iloc[idx].ravel(), bins=256, fc='k', ec='k')
    ax4.hist(df['band_2'].iloc[idx].ravel(), bins=256, fc='k', ec='k')
    f.set_figheight(10)
    f.set_figwidth(10)
    plt.suptitle(str(df['inc_angle'].iloc[idx])+c[df['is_iceberg'].iloc[idx]])
    plt.show()

def plotMinMax(df):
    min_max = pd.DataFrame()
    min_max['min_1'] = data['band_1'].apply(lambda x: x.min())
    min_max['max_1'] = data['band_1'].apply(lambda x: x.max())
    min_max['min_2'] = data['band_2'].apply(lambda x: x.min())
    min_max['max_2'] = data['band_2'].apply(lambda x: x.max())
    f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2)
    ax1.hist(min_max['min_1'])
    ax2.hist(min_max['max_1'])
    ax3.hist(min_max['min_2'])
    ax4.hist(min_max['max_2'])
    f.set_figheight(20)
    f.set_figwidth(20)
    plt.show()

def splitSave(df):
    train = df.sample(frac=0.8)
    val = df[~df.isin(train)].dropna()
    train.to_json(os.path.join(path, 'train_train.json'))
    val.to_json(os.path.join(path, 'train_val.json'))
    print('split done')

def amplitudeSplit(df):
    '''
    according to angle value, splite the band_1、band_2
    '''
    df['angle'] = df['inc_angle'].apply(lambda x: 45 if math.isnan(x) else x)


def splitAndSaveTest():
    test = pd.read_json(os.path.join(path, 'test.json'))
    test['band_1'] = test['band_1'].apply(lambda x: np.array(x).reshape(75,75))
    test['band_2'] = test['band_2'].apply(lambda x: np.array(x).reshape(75,75))
    test['inc_angle'] = pd.to_numeric(test['inc_angle'], errors='coerce')
    
    length = test.shape[0]
    print('total', test.shape)
    test1 = test[0:length/3]
    test2 = test[length/3:length*2/3]
    test3 = test[length*2/3:]
    print(test1.shape[0]+test2.shape[0]+test3.shape[0])

    test1.to_json(os.path.join(path, 'test1.json'))
    test2.to_json(os.path.join(path, 'test2.json'))
    test3.to_json(os.path.join(path, 'test3.json'))

if __name__ == '__main__':
    # splitAndSaveTest()
    splitSave(data)

    # plotMinMax(data)

    # for i in range(100,200):
    #     plotSample(data, i)
    #     i += 1
    # data.to_json(os.path.join(path, 'train_clean.json'))
    # pass