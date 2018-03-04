import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import torch
import pandas as pd
import numpy as np
import torchvision
from torch.autograd import Variable
from torchvision import transforms
from data import DataSet
from data import read_clean
import models

use_cuda = torch.cuda.is_available()

path = os.path.expanduser('~/codedata/ice/')
print('define model....')
# model = models.ResModel(path)
# model = models.convNet(path)
# model = models.smallNet(path)
# model = models.lateralNet(path)
# model = models.resModel(path)
model = models.outsModel(path)

if use_cuda:
    model.cuda()

print('loading model...')
model.load('1_100_0.2150_0.0284.pth')
model.eval()

id_total = []
predicted_total = []

print('loading data...')
data_test_src, id_test, inc_test_angle = read_clean(path, 'test_size.json', predicted=True)
data_test = DataSet(data_test_src,
                id_test,
                inc_test_angle,
                train=False,
                predicted=True)
print('dataset', data_test_src.shape)

print('predict.....')
data_loader = torch.utils.data.DataLoader(data_test, 
                                batch_size=32,
                                shuffle=False,
                                num_workers=2)

for batch_idx, (images, ids, incs) in enumerate(data_loader):
    if use_cuda:
        images = images.cuda()
        incs = incs.cuda()
    images = Variable(images, volatile=True)
    incs = Variable(incs, volatile=True)
    outputs = model(images, incs)
    outputs = outputs.data.sigmoid() # because sigmoid has been moved to binary_cross_entropy_with_logits
    out_array = outputs.cpu().numpy()
    # out_array = outputs.data.cpu().numpy()
    # probability = np.exp(out_array[:,1]) / np.exp(out_array).sum(1)
    probability = out_array.squeeze()
    predicted_total.extend(probability)
    id_total.extend(ids)
print('predict:', len(predicted_total))

predict_dict = {'id':id_total, 'is_iceberg':predicted_total}
predict_series = pd.DataFrame(predict_dict)
submit_name = 'vggbn/submit_1_outsmodel.csv'
predict_series.to_csv(os.path.join(path, submit_name), index=False)
print('save %s done' % submit_name)
# check submit validity
print('check if submit.csv is validity......')
submit = pd.read_csv(os.path.join(path, submit_name))
assert (submit['is_iceberg'] > 0).all()
assert (submit['is_iceberg'] <= 1).all()
print('all right....')