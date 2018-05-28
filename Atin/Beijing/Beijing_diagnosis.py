from Atin.Russian.helper import *

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch.utils.data.sampler import WeightedRandomSampler
import torch.optim as optim
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
import torch.nn.init as init
from pytorch_monitor import monitor_module, init_experiment


import random
import os
import matplotlib.pyplot as plt
from scipy.misc import imresize
from sklearn import metrics

import sys
import warnings

from collections import Counter

os.environ["CUDA_VISIBLE_DEVICES"]="1"


def beijing_pre_process(path):

    patient_id = list({x.split('_')[0] for x in os.listdir(path)})


class Beijing_dataset(Dataset):


    def __init__(self,root_dir, patient_id_G, patient_id_H,slices_id="6",transform=None):
        self.root_dir = root_dir
        self.patient_id_G = patient_id_G
        self.patient_id_H = patient_id_H
        self.slices = slices_id
        self.transform = transform

    def __len__(self):

        return len(self.patient_id_G) + len(self.patient_id_H)

    def __getitem__(self, idx):
        if idx < len(self.patient_id_G):
            file_name = "glaucoma/"+self.patient_id_G[idx]+"_"+self.slices+"_G.png"
            train_y = 1
        else:
            idx_new = idx-len(self.patient_id_G)
            file_name = "healthy/"+ self.patient_id_H[idx_new] + "_"+self.slices+"_H.png"
            train_y = 0

        loc = os.path.join(self.root_dir, file_name)
        im = plt.imread(loc)
        #im = imresize(im, (143, 400),interp='nearest') #original image size is
        #im = imresize(im, (200, 400), interp='nearest')
        im = imresize(im, (125, 350), interp='nearest')
        im = im/255
        im = np.expand_dims(im,0)
        im = im.transpose((1,2,0))


        if self.transform is not None:
            im = self.transform(im)

        return im, train_y


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size= (3,3), padding=(2,2), dilation=2)
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=8, kernel_size= (3,3), padding=(2,2), dilation=2)
        self.conv3 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3,3), padding=(2,2), dilation=2)
        self.conv4 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3,3), padding=(2,2), dilation=2)
        self.conv5 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3,3), padding=(2,2), dilation=2)
        self.conv5_drop = nn.Dropout(p=.2)
        self.fc1 = nn.Linear(in_features=32, out_features=10)
        self.fc2 = nn.Linear(in_features=10, out_features=2)

        for m in self.modules():

            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                #init.xavier_normal(m.weight.data, gain=nn.init.calculate_gain('relu'))
                init.xavier_uniform(m.weight.data, gain=nn.init.calculate_gain('relu'))
                #init.kaiming_uniform(m.weight.data)
                init.constant(m.bias, .1)

            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()




    def forward(self, x):
        x = F.avg_pool2d(x, (2,2))
        x = F.elu(self.conv1(x))
        x = F.elu(self.conv2(x))
        x = F.max_pool2d(x, (2,2))
        x = F.elu(self.conv3(x))
        x = F.max_pool2d(x, (2,2))
        x = F.elu(self.conv4(x))
        x = F.max_pool2d(x, (2,2))
        x = F.elu(self.conv5(x))
        x = F.adaptive_avg_pool2d(x,output_size=1)
        x = self.conv5_drop(x)
        x = F.elu(x)
        x = x.view(-1, 32)
        x = F.elu(self.fc1(x))
        x = self.fc2(x)
        return x
        #return F.log_softmax(x, dim=1)





def train(epoch,model,dataloader,optimizer):
    model.train()

    train_loss = 0
    #correct = 0

    kont = 0
    for batch_id, (data, target) in enumerate(dataloader):
        kont += data.size(0)
        data, target = data.type(torch.FloatTensor).cuda(), target.type(torch.LongTensor).cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)

        #loss = F.cross_entropy(input=output, target=target)

        loss = F.cross_entropy(input=output, target=target)
        train_loss = train_loss + loss.data[0]* data.size(0)
        #train_loss = train_loss + loss.data[0]

        loss.backward()
        optimizer.step()


        if batch_id % 50 ==0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_id * len(data), len(dataloader.dataset),100. * batch_id / len(dataloader), loss.data[0]))

    train_loss /= kont
    print('\nTrain set: Average loss: {:.4f}'.format(train_loss))


def test(model,testloader):
    model.eval()
    test_loss = 0
    correct = 0
    for _, (data, target) in enumerate(testloader):
        data, target = data.type(torch.FloatTensor).cuda(), target.type(torch.LongTensor).cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        test_loss = test_loss + F.cross_entropy(output, target, size_average=False).data[0]
        pred = output.data.max(1, keepdim=True)[1]
        correct = correct + pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(testloader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(testloader.dataset),
        100. * correct / len(testloader.dataset)))

def auc_cal(model, testloader):
    model.eval()
    test_loss = 0
    correct = 0

    final_pred = []
    for _, (data, target) in enumerate(testloader):
        data, target = data.type(torch.FloatTensor).cuda(), target.type(torch.LongTensor).cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)

        test_loss = test_loss + F.cross_entropy(output, target, size_average=False).data[0]
        pred = output.data.max(1, keepdim=True)[1]
        new_pred = pred.cpu().numpy().reshape(len(testloader.dataset), )
        final_pred.append(new_pred)

        correct = correct + pred.eq(target.data.view_as(pred)).cpu().sum()

        output = np.exp(output.data.cpu().numpy())
        predicted = output / np.sum(output, axis=1).reshape(len(testloader.dataset),1)
        target = target.data.cpu().numpy()
        fpr, tpr, thresholds = metrics.roc_curve(1 + target, predicted[:, 1], pos_label=2)



    test_loss /= len(testloader.dataset)

    glaucoma_pred = np.count_nonzero(final_pred)
    print("Predicted Healthy: {} and Predicted Glaucoma: {}".format(len(testloader.dataset)-glaucoma_pred,glaucoma_pred))

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(testloader.dataset),
        100. * correct / len(testloader.dataset)))
    return metrics.auc(fpr, tpr)



def prediction(im_array, c=2):
    predicted = np.zeros((im_array.shape[0], c))
    i = 0
    for im in im_array:
        tr = transforms.Compose([ToTensor()])
        im = tr(im)
        im = im.unsqueeze(0)
        output = model(Variable(im.type(torch.FloatTensor).cuda()))
        output = np.exp(output.data.cpu().numpy())

        predicted[i, :] = output / np.sum(output)
        # predicted[i,:] = np.exp(output.data.cpu().numpy())

        i += 1

    return predicted


if __name__ == "__main__":

    if not sys.warnoptions:
        warnings.simplefilter("ignore")

    # root_dir = '/home/atin/DeployedProjects/data/Beijing'
    root_dir = '/home/atin/rescaled_march_2018'
    #Read the patient id from disk for glaucoma and healthy
    patient_id_G = [x.split('_')[0] for x in os.listdir(root_dir + "/glaucoma/")]
    patient_id_H = [x.split('_')[0] for x in os.listdir(root_dir + "/healthy/")]
    freq_G = Counter(patient_id_G)
    freq_H = Counter(patient_id_H)
    #len(freq_G), len(freq_H)= (164, 2742)
    #Collect those patients which have all 6 slices
    G_exclude = [k for k, v in freq_G.items() if v != 6]
    H_exclude = [k for k, v in freq_H.items() if v != 6]
    #len(freq_G_exclude), len(freq_H_exclude) =(9, 76)

    #Remove those patients which don't have 6 slices.
    patient_id_G = set(patient_id_G) - set(G_exclude)
    patient_id_H = set(patient_id_H) - set(H_exclude)

    print(len(patient_id_G), len(patient_id_H))





    # patient_id_G = {x.split('_')[0] for x in os.listdir(root_dir+"/glaucoma/")}
    # patient_id_H = {x.split('_')[0] for x in os.listdir(root_dir+"/healthy/")}

    # random.seed(982)
    c = 593
    random.seed(c)
    print("random seed: {}".format(c))

    patient_id_G_test = random.sample(patient_id_G,20)
    patient_id_H_test = random.sample(patient_id_H,200)

    patient_id_G_train = list(patient_id_G.difference(patient_id_G_test))
    patient_id_H_train = list(patient_id_H.difference(patient_id_H_test))



    #transformed_images = Beijing_dataset(root_dir,patient_id_G_train,patient_id_H_train)
    transformed_images = Beijing_dataset(root_dir,patient_id_G_train,patient_id_H_train,transform=transforms.Compose([Horizontalflip(),
                                                                       RandomRotate(angle_range=(0, 5), prob=0,
                                                                                    mode="constant"),
                                                                       Translation(h_shift_range=(0, 4),
                                                                                   w_shift_range=(0, 8), prob=0,
                                                                                   mode="constant"),
                                                                       ToTensor()]))


    prob = np.array([len(patient_id_G_train),len(patient_id_H_train)])
    prob = 1/prob

    weight = [prob[0]]*len(patient_id_G_train) + [prob[1]]*len(patient_id_H_train)
    sampler = WeightedRandomSampler(weight,len(weight))

    dataloader = DataLoader(transformed_images,batch_size=16, num_workers=10,sampler=sampler)
    #dataloader = DataLoader(transformed_images, batch_size=16, num_workers=10, shuffle=True)

    testdataset = Beijing_dataset(root_dir,patient_id_G_test,patient_id_H_test, transform= transforms.Compose([ToTensor()]))
    #testdataset = Beijing_dataset(root_dir,patient_id_G_test,patient_id_H_test)

    testloader = DataLoader(testdataset, batch_size=220, shuffle=True, num_workers=20)

    # config = dict(
    #     title="An Experiment",
    #     description="Testing out a NN",
    #     log_dir='logs')
    # writer = init_experiment(config)

    model = Net()
    # monitor_module(model, writer)
    print(model)

    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    nb_params = sum([np.prod(p.size()) for p in model_parameters])
    print("no. of trainable parametes is: {}".format((nb_params)))

    model.cuda()

    # optimizer = optim.SGD(model.parameters(), lr=.01, momentum=.5)
    optimizer = optim.Adam(model.parameters(), lr=.001)
    # optimizer = optim.Adadelta(model.parameters(),weight_decay=.5)

    nb_epoch = 25
    for epoch in range(1, nb_epoch + 1):
        train(epoch, model, dataloader,optimizer)
        print("Test AUC:", auc_cal(model, testloader))
        # test(model, testloader)

    # predicted_test_prob = prediction(X_test)
    # predicted_train_prob = prediction(X_train)
    #
    # fpr, tpr, thresholds = metrics.roc_curve(1 + y_test, predicted_test_prob[:, 1], pos_label=2)
    # print("test AUC:\n")
    # print(metrics.auc(fpr, tpr))
    #
    # fpr1, tpr1, thresholds = metrics.roc_curve(1 + y_train, predicted_train_prob[:, 1], pos_label=2)
    # print("train AUC:\n")
    # print(metrics.auc(fpr1, tpr1))
























