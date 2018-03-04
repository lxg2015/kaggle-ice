import os
import cv2
import random
import pandas as pd
import numpy as np
import torch.utils.data as data
import torch
from tool import randomCrop
from tool import rotate
from tool import lee_filter

def read_clean(path, file, predicted=False):
    '''
    train and test prepare
    return:
    full_img_tr: numpy
    data['is_iceberg']: numpy
    list(data['id']): list
    '''
    data = pd.read_json(os.path.join(path, file))
    band_1_tr = np.concatenate([im for im in data['band_1']]).reshape(-1, 75, 75)
    band_2_tr = np.concatenate([im for im in data['band_2']]).reshape(-1, 75, 75)
    # band_3_tr = (band_1_tr**2 + band_2_tr**2) / 2
    # full_img_tr = np.stack([band_1_tr, band_2_tr, band_3_tr], axis=1) # 1604,2,75,75
    full_img_tr = np.stack([band_1_tr, band_2_tr], axis=1) # 1604,2,75,75
    full_img_tr = full_img_tr.transpose(0,2,3,1)
    if not predicted:
        return full_img_tr, data['is_iceberg'].values
    else:
        return full_img_tr, list(data['id'])

class train_cross():
    '''
    N folder cross verify
    '''
    def __init__(self, train, label, num):
        '''
        num: split set number
        '''
        self.length = train.shape[0]
        self.num = num
        self.data = train
        self.label = label
        self.image_list = list(range(self.length))
        random.shuffle(self.image_list)  # replace
    
    def getset(self, ids):
        span = self.length / self.num
        first_index = int(ids*span)

        if ids is not self.num-1:
            test_list = self.image_list[first_index:int((ids+1)*span)]
        else:
            test_list = self.image_list[first_index:]
        
        image_test = self.data[test_list]
        lab_test = self.label[test_list]
        
        train_list = list(set(self.image_list) - set(test_list))
        image_train = self.data[train_list]
        lab_train = self.label[train_list]
        return image_train, lab_train, image_test, lab_test

class DataSet(data.Dataset):
    def __init__(self, datap, labelp, train, predicted=False):
        self.image_size = 64
        self.data = datap
        self.predicted = predicted
        self.length = datap.shape[0]
        self.train = train
        if(not predicted):
            self.label = labelp
            self.id = []
        else:
            self.label = []
            self.id = labelp
            
       
    def __getitem__(self, idx):
        img = self.data[idx] # WxHxC
        
        # substract min value, for resnet18
        # img -= img.min() 
        
        # take the opposite
        # img = 0 - img 

        # speckle filter
        # img = lee_filter(img)

        # pca whitening  https://github.com/RobotLiu2015/machine-learning/tree/master/PCA%20and%20Whitening

        if self.train:

            # if random.random() < 0.5:
            #     # add speckle noise(https://stackoverflow.com/questions/22937589/how-to-add-noise-gaussian-salt-and-pepper-etc-to-image-in-python-with-opencv)
            #     row,col,ch = img.shape
            #     gauss = np.random.randn(row,col,ch)
            #     gauss = gauss.reshape(row,col,ch)        
            #     noisy = img + img * gauss
            
            # if random.random() < 0.5:
            # # salter and pepper
            #     row,col,ch = img.shape
            #     s_vs_p = 0.5
            #     amount = 0.004
            #     out = np.copy(img)
            #     # Salt mode
            #     num_salt = np.ceil(amount * img.size * s_vs_p)
            #     coords = [np.random.randint(0, i - 1, int(num_salt))
            #             for i in img.shape]
            #     out[coords] = 1

            #     # Pepper mode
            #     num_pepper = np.ceil(amount* img.size * (1. - s_vs_p))
            #     coords = [np.random.randint(0, i - 1, int(num_pepper))
            #             for i in img.shape]
            #     out[coords] = 0
            #     img = out
        
            if random.random() < 0.5: # np.fliplr, error, tensor not support
                img = rotate(img, 180)

            if random.random() < 0.5:
                angle = random.uniform(-20,20)
                img = rotate(img, angle)
                # print('rotate', angle)

            if random.random() < 0.3:
                img = cv2.resize(img, (85,85))  
                img = randomCrop(img, 75, 75)
            elif random.random() < 0.6:
                img = np.pad(img, ((7,7),(7,7),(0,0)), 'reflect')
                img = randomCrop(img, 75, 75)
            else:
                pass
            # img = randomCrop(img, 60, 60)
            # pass
        else:
            pass

        # back_idx = random.randrange(0, self.length)
        # img_back = self.data[back_idx]
        # img_back_crop = randomCrop(img, 60, 60)
        
        img1 = randomCrop(img, 40, 40)
        img2 = randomCrop(img, 40, 40)
        img3 = img[0:40, 0:40, :]
        # img = randomCrop(img, 60, 60)

        # print(img.shape)
        # img = np.concatenate((img[:,:30,:30], img[:,30:,30:], img[:,:30,30:], img[:,30:,:30], img[:,15:45,15:45]))
        # img = img[:,15:45,15:45]
        img = np.concatenate((img1, img3, img[10:50,10:50,:]), axis=2)
        img = cv2.resize(img, (self.image_size, self.image_size))

        img = img.transpose(2,0,1) 
        img = torch.from_numpy(img).float()

        if not self.predicted:
            return img, self.label[idx]
        else:
            return img, self.id[idx]
    
    def __len__(self):
        return self.length

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from torchvision import transforms
    print('dataset main run')
    transform = transforms.Compose([
        transforms.ToTensor()  # simply typeas float and divide by 255
    ])
    dataset = DataSet(path = '/home/lxg/codedata/ice',
                    file = 'train_train.json',
                    train = True,
                    predicted=True)
    for idx in range(len(dataset)):
        img, label = dataset[idx]
        img = img.numpy()
        print('idx:', idx, 'label:', label, 'shape:', img.shape)
        f, (ax1, ax2) = plt.subplots(1,2)
        ax1.imshow(img[0])
        ax2.imshow(img[1])
        f.suptitle(str(label))
        # plt.show()

        c,w,h = img.shape
        # img = img.transpose(1,2,0)
        # filter_img = img
        filter_img = lee_filter(img)
        print((filter_img[0] == img[0]).sum())
        # img = img.transpose(2,1,0)
        f, (ax1, ax2) = plt.subplots(1,2)
        ax1.imshow(filter_img[0])
        ax2.imshow(filter_img[1])
        f.suptitle('filter_'+str(label))
        plt.show()


