import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch
import models
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from torch.autograd import Variable
from data import DataSet, read_clean, train_cross, getmaxmask
from skimage import morphology, measure
import utils
import torch.nn.functional as F

def processData():
    data = pd.read_json(path+'test.json')
    data['band_1'] = data['band_1'].apply(lambda x: np.array(x).reshape(75,75))
    data['band_2'] = data['band_2'].apply(lambda x: np.array(x).reshape(75,75))
    size = []
    for idx in range(len(data)):
        img1 = data['band_1'].iloc[idx]
        img2 = data['band_2'].iloc[idx]
        # print(type(img1))
        max_area_1, box_1 = getmaxmask(img1)
        max_area_2, box_2 = getmaxmask(img2)
        size.append(max(max_area_1, max_area_2))            
    data['mask_size'] = size
    data.to_json(path+'test_size.json')

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
    

def plotSampleCrop(df, idx):
    c = ('ship', 'ice')
    f, ((ax1, ax2), (ax3, ax4), (ax5,ax6)) = plt.subplots(3,2)
    ax1.imshow(df['band_1'].iloc[idx])
    ax2.imshow(df['band_2'].iloc[idx])
    
    img1 = df['band_1'].iloc[idx]
    mask1 = img1 > img1.mean() +2*img1.std()
    mask1 = morphology.remove_small_objects(mask1, min_size=10, connectivity=2, in_place=False)
    img1 = img1*mask1
    ax3.imshow(img1)
    
    img2 = df['band_2'].iloc[idx]
    mask2 = img2 > img2.mean() +2*img2.std()
    mask2 = morphology.remove_small_objects(mask2, min_size=10, connectivity=2, in_place=False)
    img2 = img2*mask2
    ax4.imshow(img2)

    h,w = img2.shape
    # get object minx miny maxx maxy
    mask = measure.label(mask1+mask2)
    propertity = measure.regionprops(mask)
    max_area = 0
    for region in propertity:
        if region.area > max_area:
            max_area = region.area
            center_x, center_y = region.centroid
            box = region.bbox

    # boxes = np.array(boxes)
    minx = box[0]#boxes[:,0].min()
    miny = box[1]#boxes[:,1].min()
    maxx = box[2]#boxes[:,2].max()
    maxy = box[3]#boxes[:,3].max()

    print(minx, miny, maxx, maxy)
    img1 = df['band_1'].iloc[idx]
    img1 = img1[minx:maxx,miny:maxy]
    ax5.imshow(img1)

    # crop object
    max_size_half = max(maxx-minx, maxy-miny)/2 
    max_size_half = max(max_size_half*1.5, 20)

    # center_x, center_y = (minx+maxx)/2, (miny+maxy)/2
    minx = int(max(center_x - max_size_half, 0))
    miny = int(max(center_y - max_size_half, 0))
    maxx = int(min(center_x + max_size_half, w))
    maxy = int(min(center_y + max_size_half, h))
    
    img1 = df['band_1'].iloc[idx]
    img1 = img1[minx:maxx,miny:maxy]
    ax6.imshow(img1)

    # ax3.hist(df['band_1'].iloc[idx].ravel(), bins=256, fc='k', ec='k')
    # ax4.hist(df['band_2'].iloc[idx].ravel(), bins=256, fc='k', ec='k')
    f.set_figheight(10)
    f.set_figwidth(10)
    plt.suptitle(str(df['inc_angle'].iloc[idx])+c[df['is_iceberg'].iloc[idx]])
    plt.show()
    

def testData():
    data = pd.read_json(path+'train_train.json')
    data['band_1'] = data['band_1'].apply(lambda x: np.array(x).reshape(75,75))
    data['band_2'] = data['band_2'].apply(lambda x: np.array(x).reshape(75,75))
    data['inc_angle'] = pd.to_numeric(data['inc_angle'], errors='coerce')  # lack data is filled with na

    for i in range(0,100):
        plotSample(data, i)
        # plotSampleCrop(data, i)


def testModel(model, x, angles):
    print('input size', x.size())
    out = model(x, angles)
    print('out', out.size())

def teststn(model, x):
    stn_x = model.stn(x)
    stn_x = stn_x.data.cpu().numpy()
    np.save(path+'stn_x.npy', stn_x)

    x_src = x.data.cpu().numpy()
    np.save(path+'stn_xsrc.npy', x_src)
    print('save stn')

def showstn(nimg):
    stn_x = np.load(path+'stn_x.npy')
    x_src = np.load(path+'stn_xsrc.npy')
    for idx in range(nimg):
        f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2)
        ax1.imshow(stn_x[idx,0])
        ax2.imshow(stn_x[idx,1])
        ax3.imshow(x_src[idx,0])
        ax4.imshow(x_src[idx,1])
        f.set_figheight(10)
        f.set_figwidth(10)
        plt.savefig(path+'image/%d.png' % idx)
        plt.show()
        print('show %d' % idx)

def testListData():
    images_all, labels_all, inc_angle = read_clean(path, 'train_clean_size.json') 
    train_dataset = DataSet(images_all, 
                            labels_all, 
                            inc_angle,
                            train=True)
    for idx in range(len(train_dataset)):
        img, label, inc_angle = train_dataset[idx]
        img = img.numpy()
        f, (ax1, ax2) = plt.subplots(1,2)
        ax1.imshow(img[0])
        ax2.imshow(img[1])
        f.suptitle(str(label))
        plt.show()

def analyseResult(model, x, incs, label):
    out = model(x, incs)
    out = out.sigmoid()
    f, (ax1, ax2) = plt.subplots(1,2)
    x = x.numpy() # [batch_size, channel, w,h]
    ax1.imshow(x[0,0])
    ax2.imshow(x[0,1])
    out = out.data.numpy()
    print(out.shape)
    f.suptitle('ice %s out %f' % ('True' if label[0] else 'False', out[0]))
    plt.show()

def testModelMain():
    print('loading data.....')
    images_all, labels_all, inc_angle = read_clean(path, 'train_clean_size.json') 
    train_dataset = DataSet(images_all, 
                            labels_all, 
                            inc_angle,
                            train=False)
    batch_size = 1
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                                batch_size=batch_size,
                                                shuffle=False,
                                                num_workers=2)     
    # model = models.IceVGG(path)
    model = models.smallNet(path)
    # model = models.resModel(path)
    # model = models.lateralNet(path)
    # model = models.convNet(path)
    model.load('0_test_crop.pth')
    model.eval()
    # criterion = utils.CrossEntropy()
    predict_all = []
    for idx, (x, labels, incs) in enumerate(train_loader):
        # x = Variable(torch.randn(3, 2, 75, 75))
        if use_cuda:
            model.cuda()
            x = x.cuda()
            labels = labels.cuda()
            incs = incs.cuda()
        x = Variable(x, volatile=True)
        labels = Variable(labels, volatile=True)
        incs = Variable(incs, volatile=True)
        
        # testModel(model, x, angles)
        # teststn(model, x)
        # showstn(batch_size)
        # analyseResult(model, x, incs, labels)
        out = model(x, incs)
        # loss = criterion(out, labels)
        # import pdb; pdb.set_trace()
        # print(out.data, labels.data)
        # print('loss', loss.data[0])
        # labels = labels.float() #[batch_size, 1]
        # print('out', out.shape)
        # input = out.squeeze()
        # target = labels
        # max_val = (-input).clamp(min=0)
        # loss = input - input * target + max_val + ((-max_val).exp() + (-input - max_val).exp()).log()
        # print('function loss', loss.mean().data[0])

        # out = out.squeeze().sigmoid() #[batch_size, 1]
        # loss = -((out.log()*labels) + (1-out).log()*(1-labels)) # 2x2 so bad 
        # print('shape test',out.log().shape, labels.shape, type(out.log()), type(labels), (out.log()*labels).shape)
        # print('compute loss ',  loss.mean().data[0])
        # print('shape', out.shape, labels.shape, loss.shape)

        out = out.squeeze().sigmoid()
        out = out.data.cpu().numpy()
        predict_all.extend(out)
    data = pd.read_json(path+'train_clean_size.json')
    data['predict'] = predict_all
    data.to_json(path+'train_clean_predict_small.json')

if __name__ == '__main__':
    path = os.path.expanduser('~/codedata/ice/')
    use_cuda = torch.cuda.is_available()
    # testModelMain()
    # showstn(30)
    # testData()
    # testListData()
    processData()
