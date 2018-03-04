import dataset
from torchvision import transforms
import matplotlib.pyplot as plt

path = '/home/lxg/codedata/ice/'

images, labels = dataset.read_clean(path, 'train_val.json')
train_set = dataset.train_cross(images, labels, 5)
train,label_train, test, label_test = train_set.getset(0)
print('train',train.shape, 'test', test.shape)


transform = transforms.Compose([
    transforms.ToTensor()  # simply typeas float and divide by 255
])
test_data = dataset.DataSet(test, label_test, 
                        transform=transform, train=False)
img, lab = test_data[0]
img = img.numpy()
print('img:',img.shape, 'lab', lab)

for i in range(len(test_data)):
    img,lab = test_data[i]
    img = img.numpy()
    f, (ax1,ax2) = plt.subplots(1,2)
    ax1.imshow(img[0])
    ax2.imshow(img[1])
    f.suptitle(str(lab))
    plt.show()