backscatter coefficient 后向散射系数
[极化](https://zhidao.baidu.com/question/433786349728507724.html)

# convNet
<!-- weight_decay=0.01, no data augmentation -->
test Epoch 21 train_loss:0.14493, test_loss 0.19278, best_test_loss 0.19278, accuracy 90.34483    0.3309 
<!-- weight_decay=0.01, resize random crop -->
test Epoch 34, lr: 0.00000 best_test_loss 0.19360, test_accuracy 86.20690, train_loss:0.18985, test_loss 0.19360  0.2705 
<!-- weight_decay=0.01, pad, reiseze random crop -->
test Epoch 43, lr: 0.00000 best_test_loss 0.16528, test_accuracy 93.10345, train_loss:0.25169, test_loss 0.16528 0.1945  **rotate is awesome?**
<!-- weight_decay=0.01, flip, pad, resize random crop -->
test Epoch 37, lr: 0.00000 best_test_loss 0.22386, test_accuracy 91.03448, train_loss:0.27736, test_loss 0.22386  0.2854
<!-- weight_decay=0.01, rotate, flip, pad, random crop-->
test Epoch 29, lr: 0.00000 best_test_loss 0.19650, test_accuracy 88.96552, train_loss:0.31724, test_loss 0.19650  0.2399
<!-- batch_size=64, weight_decay=0.01, speckle noise, pad, resize random crop -->
test Epoch 99, lr: 0.00156250 best_test_loss 0.15662, test_accuracy 86.89655, train_loss:0.17001, test_loss 0.32298   0.1978
<!-- batch_size=64, weight_decay=0.01,  salter and pepper noise, speckle noise, pad, resize random crop -->
test Epoch 99, lr: 0.00004883 best_test_loss 0.15658, test_accuracy 87.58621, train_loss:0.17496, test_loss 0.33669
<!-- batch_size=64, weight_decay=0.001, salter and pepper noise, speckle noise, pad, resize random crop -->
test Epoch 99, lr: 0.00039063 best_test_loss 0.15901, test_accuracy 80.68966, train_loss:0.40502, test_loss 0.48465
<!-- batch_size=128, weight_decay=0.001, salter and pepper noise, speckle noise, pad, resize random crop -->
test Epoch 99, lr: 0.00004883 best_test_loss 0.15368, test_accuracy 85.51724, train_loss:0.20466, test_loss 0.54482
<!-- add min value, batch_size=128, weight_decay=0.001, salter and pepper noise, speckle noise, pad, resize random crop -->  过拟合？？
test Epoch 99, lr: 0.00019531 best_test_loss 0.13829, test_accuracy 91.72414, train_loss:0.14624, test_loss 0.21511  0.2354
<!-- sub min value, batch_size=128, weight_decay=0.001, salter and pepper noise, speckle noise, pad, resize random crop -->  
test Epoch 99, lr: 0.00039063 best_test_loss 0.15874, test_accuracy 86.89655, train_loss:0.10851, test_loss 0.51559
<!-- sub min value, pad, resize random crop -->
test Epoch 99, lr: 0.00002441 best_test_loss 0.13738, test_accuracy 86.89655, train_loss:0.13139, test_loss 0.61632
<!-- speckle filter 没有帮助-->
test Epoch 99, lr: 0.00039063 best_test_loss 0.23778, test_accuracy 83.56164, train_loss:0.18233, test_loss 0.28511
<!-- band_3 sqrt(band_1**2 + band_2**2) 没有帮助-->
test Epoch 99, lr: 0.00625000 best_test_loss 0.22963, test_accuracy 86.30137, train_loss:0.23748, test_loss 0.34087


# res18 pretrained
<!-- cross entropy -->
average test loss:0.204781
<!-- dropout, fc -->
average test loss:0.223527
<!-- 去掉band_3_tr = (band_1_tr**2 + band_2_tr**2) / 2 -->
average test loss:0.216351
<!-- add two layer overfitting more -->
average test loss:0.272495
<!-- 找到一个bug，对res18.classifier赋值，不会报错，尽管并不存在这一层，应该对res18.fc赋值 -->
<!-- 使用sigmoid+F.binary_cross_entropy() -->
average test loss:0.235534
<!-- 添加dropout -->
average test loss:0.273817
<!-- 添加dropout的同时，添加一层fc -->
average test loss:0.235803

## resModel
<!-- 2channel 30pixel input -->
average test loss:0.214413
<!-- 8channel -->
average test loss:0.237254
<!-- 2channel centeral 30 pixel -->
average test loss:0.171246
average test loss:0.205263
average test loss:0.205097, average train loss:.0.181294  线上0.2127
<!-- 8channel+2ceneral channel 过拟合 -->
average test loss:0.233942
average test loss:0.208232, average train loss:.0.153128
<!-- 4channel random crop 2centeral crop -->
average test loss:0.192417, average train loss:.0.206058
average test loss:0.185343, average train loss:.0.203189
average test loss:0.236775, average train loss:.0.201195 结果太随机了
<!-- 4channel random crop 2centeral crop 2resize channel-->
average test loss:0.223221, average train loss:.0.202647
<!-- ramdom choose ship image as auxiliary channel -->
<!-- 全连接变512通道 -->
average test loss:0.233810, average train loss:.0.200464
<!-- no noise -->
average test loss:0.233780, average train loss:.0.076921
<!-- noise -->
average test loss:0.209401, average train loss:.0.068871 stack


# vgg16

# zca白化
<!-- loss 不下降 -->
fold 0, Epoch 22, lr: 0.01000000 best_test_loss 0.67068, train_loss:0.76683, test_loss 0.85430

# 通过对输入加上Batch Normalization，以实现对输入的预处理
<!-- batch nomalization就相当于白化 -->
average test loss:0.196119, average train loss:.0.211353 线上 0.1960 

# 对最后的输出层添加噪声，以使sigmoid的输出区分开来，而不至于在0.5附近，
在sigmoid处添加噪声
使输出更趋向于二值输出，参见deep learning的autoencoder
average test loss:0.199828, average train loss:.0.085958  线上 0.1608 what happended??

# 在ConvNet中，把dropout层去掉
是否如论文中所说，有了bn层之后，Bias理论上确实多余，而dropout呢
# 写predict的集成方法
# 同时查看ice和ship图片，发现不同
看不出，不过发现亮点位置变化很大，大小也变化很大
# 使用deformable CNN
<!--a conv3 deformable cnn -->
average test loss:0.233655, average train loss:.0.050323
<!--a conv4 deformable cnn -->
average test loss:0.228653, average train loss:.0.051385
<!--a conv5 deformable cnn -->
average test loss:0.221123, average train loss:.0.060002
<!--a conv5 deformable cnn and rand cxcy for augmentation-->
average test loss:0.222726, average train loss:.0.149231 
<!-- crop 40, no resize -->
average test loss:0.232851, average train loss:.0.064902
average test loss:0.240076, average train loss:.0.007134
<!-- conv5 conv3  deformable cnn-->
average test loss:0.238063, average train loss:.0.021351
<!-- 添加旋转增强 -->
average test loss:0.194955, average train loss:.0.110455  数据增强有用？
<!-- conv5 conv3 conv1 deformable cnn -->
average test loss:0.183261, average train loss:.0.097679  线上0.1932  0.1997


# spatial transform network
<!-- convNet -->
average test loss:0.204144, average train loss:.0.181775  线上 0.1729
average test loss:0.214501, average train loss:.0.100599  stack
<!-- 去掉水平翻转及旋转数据增强后，模型又开始过拟合严重 -->
average test loss:0.183260, average train loss:.0.092487
<!-- 三层卷积 -->
average test loss:0.324347, average train loss:.0.210049  有的好，有的差
average test loss:0.341616, average train loss:.0.258920  是真的不好

# 选出亮度大于均值+2×方差的像素的位置，然后圈定    根本不行
<!-- crop 30 -->
average test loss:0.219151, average train loss:.0.043743 线上 0.1828 不起作用？
<!-- small network -->
average test loss:0.262968, average train loss:.0.121245 线上 0.1999  也不起作用
<!-- crop 40 -->
average test loss:0.205066, average train loss:.0.044473
<!-- crop 50 -->
average test loss:0.240381, average train loss:.0.039588
<!-- crop 60 -->
average test loss:0.219999, average train loss:.0.054592
<!-- zero crop -->
average test loss:0.180112, average train loss:.0.095212 不crop更好
<!-- 按比例crop -->
average test loss:0.237561, average train loss:.0.012233 过拟合严重，应该着手找去除过拟合的方法，而不是继续新网络的尝试
<!-- 去掉inc_angle -->
average test loss:0.246844, average train loss:.0.010427
<!-- 增加旋转数据增强 -->
average test loss:0.221798, average train loss:.0.109564
<!-- 去掉stn -->
average test loss:0.220774, average train loss:.0.076189
<!-- crop 40 shift augmentation -->
average test loss:0.297814, average train loss:.0.061193 过拟合很严重

# 在全连接层添加inc_angle
<!-- inc_angle first fc -->
average test loss:0.159343, average train loss:.0.024036  损失震荡很厉害
<!-- inc_angle normalise -->
average test loss:0.207845, average train loss:.0.090437  归一化后效果和不加相同
<!-- inc_angle zero mean -->
average test loss:0.164736, average train loss:.0.070337  震荡太严重
<!-- inc_angle zero mean *10-->
average test loss:0.212593, average train loss:.0.074800  更糟糕
<!-- replace nan with zero not mean -->                   有可能训练根本就不会收敛？
average test loss:0.150993, average train loss:.0.053462  确实很好，比使用均值填充震荡幅度小
<!-- inc_angle second fc -->
average test loss:0.225410, average train loss:.0.036134
<!-- inc_angle last fc -->  发散
<!-- inc_angle frst fc 加bn层 -->
average test loss:0.373767, average train loss:.0.251022 不好

# 添加laternel结构

# 第一层卷积使用PReLU
<!-- 效果不好，有权重衰减的情况 -->
average test loss:0.232106, average train loss:.0.092780
# 改变参数初始化方法以减少震荡或不收敛的情况
看pytorch默认的初始化方法是什么 xavier-uniform
<!-- 使用kaiming-normal，初始参数很好，但是后来又发散了 -->
average test loss:0.206330, average train loss:.0.221600
<!-- 降低学习率*0.1 -->
average test loss:0.182640, average train loss:.0.088378 好一点
<!-- kaiming-uniform, -->
average test loss:0.181852, average train loss:.0.115198 
<!-- 去掉weight_decay -->
average test loss:0.194272, average train loss:.0.064499
<!-- 加大权重衰减5e-5->5e-4 -->
average test loss:0.201961, average train loss:.0.081981
<!-- 加大权重衰减5e-5->5e-3 -->
average test loss:0.148888, average train loss:.0.074843 加大权重衰减可以减轻过拟合
average test loss:0.179505, average train loss:.0.074203 结果还是不太稳定，有一个0.13 
                                                         线上0.1861
<!-- 加大权重衰减5e-5->5e-2 -->
average test loss:0.208659, average train loss:.0.089634 有一个不行

# 使用$2\times 2$的卷积核，以保持细节
# 阐释fractional pool
# 将dropout换成conv2d的步长
<!-- 应该会好一点 -->
average test loss:0.173561, average train loss:.0.077523  线上 0.2195

# resModel
<!-- resize 40 -->
average test loss:0.228612, average train loss:.0.202333
<!-- 去掉左右翻转 -->
average test loss:0.247735, average train loss:.0.149113
<!-- 去掉旋转 -->
average test loss:0.275267, average train loss:.0.185022
<!-- 75, add layer5 -->
average test loss:0.233755, average train loss:.0.178425
<!-- 去掉旋转，75, add layer5 -->
average test loss:0.254026, average train loss:.0.149570
<!-- crop 40 than resize 75 -->
average test loss:0.355445, average train loss:.0.066586 crop就过拟合
<!-- 把数据增强条件去掉，完全随机 -->
average test loss:0.266209, average train loss:.0.181895
<!-- mse_loss -->  loss不同，输出的分布也不同？
<!-- 加/3噪声 -->
average test loss:0.242483, average train loss:.0.180673 
<!-- 加/5噪声 -->
average test loss:0.252653, average train loss:.0.172899?
<!-- 2输出 -->
average test loss:0.242949, average train loss:.0.169888
<!-- 1输出 -->
average test loss:0.238308, average train loss:.0.178249
<!-- last -->
average test loss:0.232003, average train loss:.0.077536

# lateral model
average test loss:0.224093, average train loss:.0.185493 线上 0.222
<!-- 引入噪声 -->
average test loss:0.238502, average train loss:.0.183955  
<!-- 加conv6 -->
average test loss:0.201659, average train loss:.0.100682
<!-- 添加laternal -->
average test loss:0.214560, average train loss:.0.105047

<!-- smallNet mask_size<100.001, 885+11=896 -->
average test loss:0.426968, average train loss:.0.335662
<!-- 去掉layer4 crop 40-->
average test loss:0.418621, average train loss:.0.132600
<!-- 去掉layer4 crop 30 -->
average test loss:0.409492, average train loss:.0.137095
<!-- 去掉tool内的数据增强 -->
average test loss:0.368851, average train loss:.0.151549
<!-- 添加layer4 去掉tool内的数据增强 -->
average test loss:0.393786, average train loss:.0.111288
<!-- 去掉layer4 去掉tool内的数据增强 -->
average test loss:0.371570, average train loss:.0.126198
<!-- crop20 -->
average test loss:0.353302, average train loss:.0.113255
<!-- laternal layer4 -->
average test loss:0.372462, average train loss:.0.110490
<!-- laternal layer2 -->
average test loss:0.369433, average train loss:.0.123947
<!-- kaimining init, conv5, laternal layer2 -->
average test loss:0.354283, average train loss:.0.146388
<!-- kaimining init, conv5, laternal layer4 -->
average test loss:0.377611, average train loss:.0.099520
<!-- layer3 32channel -->
average test loss:0.388231, average train loss:.0.141772
<!-- stn -->
average test loss:0.412088, average train loss:.0.292780
<!-- incs -->
average test loss:0.337630, average train loss:.0.125791 可以
<!-- 去掉旋转 -->
average test loss:0.369177, average train loss:.0.022962
<!-- 旋转-50,50 -->
average test loss:0.363353, average train loss:.0.200878
<!-- sigmoid 旋转-50,50 -->
average test loss:0.340855, average train loss:.0.187562 
<!-- 降低学习率 -->
average test loss:0.398085, average train loss:.0.345799
<!-- 完全sigmoid --> 不收敛
<!-- leakyReLU 旋转-20,20-->
average test loss:0.317586, average train loss:.0.135877 厉害
average test loss:0.366345, average train loss:.0.126594 结果很随机
<!-- 完全替换为LeakyReLU -->
average test loss:0.348928, average train loss:.0.136806
<!-- 去inc -->
average test loss:0.390565, average train loss:.0.132121
<!-- 去输出噪声 -->
average test loss:0.358188, average train loss:.0.133969
<!-- inception -->
<!-- PReLU 16-->
average test loss:0.348081, average train loss:.0.134984
<!-- conv kernel 5 pad 2  leakyReLU-->
average test loss:0.337618, average train loss:.0.145867
<!-- conv kernel 7 pad 3 LeakyReLU -->
average test loss:0.307901, average train loss:.0.173047 
average test loss:0.317608, average train loss:.0.159305
average test loss:0.390299, average train loss:.0.300357
<!-- conv4 -->
average test loss:0.365219, average train loss:.0.245349
<!-- conv kernel 9 pad 4 LeakyReLU -->
average test loss:0.379116, average train loss:.0.318695
<!-- 尝试inceptionA -->
average test loss:0.342046, average train loss:.0.189783
<!-- 学习率x0.1 InceptionA 加ReLU-->
average test loss:0.353239, average train loss:.0.299657
<!-- 学习率x0.1 InceptionA 加ReLU 加BN-->
average test loss:0.376087, average train loss:.0.318330
<!-- 学习率x0.1 InceptionA 加ReLU 加BN inceptionI-->
average test loss:0.388909, average train loss:.0.298582
<!-- inceptionI -->  不靠谱
average test loss:0.408406, average train loss:.0.330217
<!-- fractional pool -->
average test loss:0.324028, average train loss:.0.270278
average test loss:0.307347, average train loss:.0.240791
average test loss:0.343076, average train loss:.0.243124

<!-- maxout -->

<!-- grad clip 2.0 -->
average test loss:0.307148, average train loss:.0.192440
average test loss:0.305122, average train loss:.0.190761
<!-- grad clip 1.0 -->
average test loss:0.326102, average train loss:.0.228217
<!-- grad clip 0.5 -->


<!-- smallNet mask_size>100.001, 708 -->
average test loss:0.080338, average train loss:.0.015431
average test loss:0.044228, average train loss:.0.007685
<!-- laternal net -->
average test loss:0.078167, average train loss:.0.024920
average test loss:0.073738, average train loss:.0.016546
<!-- resModel -->
folder:0 best test loss:0.03260 best train loss:0.02173
folder:1 best test loss:0.05762 best train loss:0.01672
folder:2 best test loss:0.00517 best train loss:0.01049
folder:3 best test loss:0.02128 best train loss:0.02577
folder:4 best test loss:0.11475 best train loss:0.01656
average test loss:0.046285, average train loss:.0.018255  线上 0.208


<!-- get network stack -->
线上 0.1535
线上 0.1446 convNet(stn)+getNet
线上 0.1341 resModel+convNet(stn)+getNet
线上 0.1347 +laternel 0.1626
线上 0.1423 +outsModel