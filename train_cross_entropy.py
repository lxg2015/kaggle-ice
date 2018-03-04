import os
os.environ['CUDA_VISIBLE_DEVICES'] = "1"

import torch
import torchvision
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import transforms
import utils
from data import DataSet, read_clean, train_cross
import models
import copy

# batch_size = 256
batch_size = 128
# batch_size = 64
# batch_size = 32
folder_num = 8
valid_folder_num = 5
num_epoch = 100
use_cuda = torch.cuda.is_available()
path = os.path.expanduser('~/codedata/ice/')

print('loading data.....')
images_all, labels_all, inc_angle_all = read_clean(path, 'train_clean_size.json') 
train_set_folders = train_cross(images_all, labels_all, inc_angle_all, folder_num)

best_test_loss_stl = [np.inf] * folder_num
best_train_loss_stl = [np.inf] * folder_num

vis = utils.Visualizer(env='lxg')

for folder in range(folder_num):
    if folder is valid_folder_num:
        break
    
    train_data, train_label, train_inc, test_data, test_label, test_inc = \
                        train_set_folders.getset(folder)
    train_dataset = DataSet(train_data, 
                            train_label, 
                            train_inc,
                            train=True)
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                                batch_size=batch_size,
                                                shuffle=True,
                                                num_workers=5)
    test_dataset = DataSet(test_data,
                            test_label,
                            test_inc,
                            train=False)
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                            batch_size=batch_size,
                                            shuffle=False,
                                            num_workers=5)
    print('define model.......')
    # model = models.convNet(path)
    # model = models.smallNet(path)
    model = models.outsModel(path)
    # model = models.fcnNet(path)
    # model = models.IceVGG(path)
    # model = models.Res18(); 
    # model = models.VGG16(); learning_rate = 0.01; optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=0.005)
    # model = models.resModel(path)
    # model = models.lateralNet(path)
    
    if use_cuda:
        model.cuda()

    learning_rate = 0.01; 
    # learning_rate *= 0.1
    optimizer = torch.optim.SGD(model.parameters(), 
                                lr=learning_rate,
                                momentum=0.9,
                                weight_decay=5e-3) 
                                # weight_decay=5e-5)
    # optimizer = torch.optim.Adam(model.parameters(),
    #                             lr=learning_rate,
    #                             weight_decay=5e-5)

    # criterion = utils.fcnLoss()
    criterion = utils.CrossEntropy()  # one out
    # criterion = utils.CrossEntropyWeight() # two out

    print('batch_size: %d' % (batch_size))                                    
    print('train_dataset: %d idx: %d' % (len(train_dataset), len(train_loader)))
    print('test_dataset: %d idx: %d' % (len(test_dataset), len(test_loader)))
    print('begin to train.....')
    num_iter = 0

    for epoch in range(num_epoch):
        # train
        print('\n')
        model.train()
        train_loss = 0
        for batch_idx, (images, labels, incs) in enumerate(train_loader):
            if use_cuda:
                images = images.cuda()
                labels = labels.cuda()
                incs = incs.cuda()
            images = Variable(images)
            labels = Variable(labels)
            incs = Variable(incs)
            optimizer.zero_grad()
            outputs = model(images, incs)
            if isinstance(criterion, utils.fcnLoss):
                loss = criterion(outputs, incs)
            else:
                loss = criterion(outputs, labels)
            # print(type(loss), type(loss.data), type(loss.data[0]), loss.data)
            loss.backward()
            nn.utils.clip_grad_norm(model.parameters(), 2.0)
            optimizer.step()
            train_loss += loss.data[0]
            if (batch_idx+1) % 3 == 0:
                print ('train Epoch [%d/%d], Iter [%d/%d] lr: %8f Loss: %.4f ' 
                       %(epoch+1, num_epoch, batch_idx+1, len(train_loader), learning_rate, loss.data[0]))
        train_loss /= len(train_loader)
        if train_loss < best_train_loss_stl[folder]:
            best_train_loss_stl[folder] = train_loss

        # test
        print('\n')
        model.eval()
        test_loss = 0
        for batch_idx, (images, labels, incs) in enumerate(test_loader):
            if use_cuda:
                images = images.cuda()
                labels = labels.cuda()
                incs = incs.cuda()
            images = Variable(images, volatile=True)
            labels = Variable(labels, volatile=True)
            incs = Variable(incs, volatile=True)
            outputs = model(images, incs)
            if isinstance(criterion, utils.fcnLoss):
                loss = criterion(outputs, incs)
            else:
                loss = criterion(outputs, labels)
            test_loss += loss.data[0]
            _, predicted = torch.max(outputs.data, 1)
            if (batch_idx+1) % 2 == 0:
                print ('test Epoch [%d/%d], Iter [%d/%d] Loss: %.4f' 
                       %(epoch+1, num_epoch, batch_idx+1, len(test_loader), loss.data[0]))
        print('\n')
        test_loss /= len(test_loader)
        if test_loss < best_test_loss_stl[folder]:
            best_test_loss_stl[folder] = test_loss
            print('best loss %.5f' % best_test_loss_stl[folder])
            # if folder == 0:
            best_model = copy.deepcopy(model.state_dict())

        print('fold %d, Epoch %d, lr: %.8f best_test_loss %.5f, train_loss:%.5f, test_loss %.5f' % (
                folder, epoch, learning_rate, best_test_loss_stl[folder], train_loss, test_loss))

        ## learning rate decay
        if epoch == 50 or epoch == 90 or epoch == 100:
            learning_rate *= 0.1
            for param_group in optimizer.param_groups:
                param_group['lr'] = learning_rate
        # early stop, model with less loss has been saved, so this is not so useful
        
        ## visdom display
        train_loss = np.clip(train_loss, a_min=0, a_max=0.5)
        test_loss = np.clip(test_loss, a_min=0, a_max=0.5)
        vis.plot_train_val(loss_train=train_loss, loss_val=test_loss)
    model.save('%d_100_%.4f_%.4f.pth' % (folder, best_test_loss_stl[folder],
                         best_train_loss_stl[folder]))

    del model


test_loss_sum = 0.
train_loss_sum = 0.
for i in range(valid_folder_num):
    print('folder:%d best test loss:%.5f best train loss:%.5f' %(i, 
                    best_test_loss_stl[i], best_train_loss_stl[i]))
    test_loss_sum += best_test_loss_stl[i]
    train_loss_sum += best_train_loss_stl[i]
    
print('average test loss:%f, average train loss:.%5f' % (test_loss_sum/valid_folder_num, 
                    train_loss_sum/valid_folder_num))