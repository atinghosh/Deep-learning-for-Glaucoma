# Adds new class of Augmentation like zoom, brigtness etc.
# import torchsample.transforms as tr is the new library
# Its the most basic single input CNN
# Adding attention to this

# Works Best so far

from sys import path


path.insert(0, '/home/atin/DeployedProjects/Glaucoma-DL/')

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
import torchsample.transforms as tr

import random
import os
import matplotlib.pyplot as plt
from scipy.misc import imresize
from sklearn import metrics

import sys
import warnings
import math

from collections import Counter
from random import shuffle

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

def make_balance(small_list, big_list):
    ratio = math.floor(len(big_list) / len(small_list))
    if ratio > .5:
        final_list = small_list * (ratio + 1) + big_list
    else:
        final_list = small_list * ratio + big_list

    shuffle(final_list)
    return final_list



def beijing_pre_process(path):
    patient_id = list({x.split('_')[0] for x in os.listdir(path)})


# class Beijing_dataset(Dataset):
#
#     def __init__(self, root_dir, patient_id_G, patient_id_H, slices_id="6", transform=None):
#         self.root_dir = root_dir
#         self.patient_id_G = patient_id_G
#         self.patient_id_H = patient_id_H
#         self.slices = slices_id
#         self.transform = transform
#
#     def __len__(self):
#
#         return len(self.patient_id_G) + len(self.patient_id_H)
#
#     def __getitem__(self, idx):
#         if idx < len(self.patient_id_G):
#             file_name = "glaucoma/" + self.patient_id_G[idx] + "_" + self.slices + "_G.png"
#             train_y = 1
#             id = self.patient_id_G[idx]
#         else:
#             idx_new = idx - len(self.patient_id_G)
#             file_name = "healthy/" + self.patient_id_H[idx_new] + "_" + self.slices + "_H.png"
#             train_y = 0
#             id = self.patient_id_H[idx_new]
#
#         loc = os.path.join(self.root_dir, file_name)
#         im = plt.imread(loc)
#         # im = imresize(im, (143, 400),interp='nearest') #original image size is
#         # im = imresize(im, (200, 400), interp='nearest')
#         im = imresize(im, (125, 350), interp='nearest')
#         im = im / 255
#         # im = np.expand_dims(im,0)
#         # im = im.transpose((1,2,0))
#
#         if self.transform is not None:
#             im = self.transform(im)
#
#         return im, train_y, id


class Beijing_dataset(Dataset):

    def __init__(self, root_dir, patient_id_G, patient_id_H, balanced_patient_id, slices_id="6", transform=None):
        self.root_dir = root_dir
        self.patient_id_G = patient_id_G
        self.patient_id_H = patient_id_H
        self.patient_id = balanced_patient_id
        self.slices = slices_id
        self.transform = transform

    def __len__(self):

        return len(self.patient_id_G) + len(self.patient_id_H)

    def __getitem__(self, idx):

        current_id = self.patient_id[idx]
        if current_id in self.patient_id_H:
            file_name = "healthy/" + self.patient_id[idx] + "_" + self.slices + "_H.png"
            train_y = 0
            id = self.patient_id[idx]
        else:
            file_name = "glaucoma/" + self.patient_id[idx] + "_" + self.slices + "_G.png"
            train_y = 1
            id = self.patient_id[idx]


        loc = os.path.join(self.root_dir, file_name)
        im = plt.imread(loc)
        # im = imresize(im, (143, 400),interp='nearest') #original image size is
        # im = imresize(im, (200, 400), interp='nearest')
        im = imresize(im, (125, 350), interp='nearest')
        im = im / 255
        # im = np.expand_dims(im,0)
        # im = im.transpose((1,2,0))

        if self.transform is not None:
            im = self.transform(im)

        return im, train_y, id


class Attention_nonlinear(nn.Module):

    def __init__(self, in_channel, out_channel, att_dim):
        super(Attention_nonlinear, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.att_dim = att_dim
        self.Linear_g = nn.Linear(in_features=out_channel, out_features=att_dim)
        self.Linear_x = nn.Linear(in_features=self.in_channel, out_features=att_dim)
        self.Linear_final = nn.Linear(in_features=att_dim, out_features=1)

        for m in self.modules():

            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                # init.xavier_normal(m.weight.data, gain=nn.init.calculate_gain('relu'))
                init.xavier_uniform(m.weight.data, gain=nn.init.calculate_gain('relu'))
                # init.kaiming_uniform(m.weight.data)
                init.constant(m.bias, .1)

            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x, g):
        g = self.Linear_g(g)
        dim0, dim1 = x.size()[2], x.size()[3]
        x = x.view(-1, self.in_channel, dim0 * dim1)
        changed_x = torch.transpose(x, 1, 2)
        new_x = [self.Linear_x(el).unsqueeze(0) for el in changed_x]
        new_x = torch.cat(new_x)
        new_x = g.unsqueeze(1) + new_x
        new_x = F.relu(new_x)

        att_map = [self.Linear_final(el).unsqueeze(0) for el in new_x]
        att_map = torch.cat(att_map)
        att_map = att_map.squeeze(2)

        att_map = att_map - torch.min(att_map, 1)[0].unsqueeze(1)
        att_map = att_map / (torch.sum(att_map, 1).unsqueeze(1))

        # att_map = F.softplus(att_map)
        # att_map = att_map / (torch.sum(att_map, 1).unsqueeze(1) + .0001)

        final_output = torch.bmm(x, att_map.unsqueeze(2))

        return final_output.squeeze(2), att_map.data.cpu().numpy().reshape(-1, dim0, dim1)


class Attention_linear(nn.Module):

    def __init__(self, in_channel, out_channel):
        super(Attention_linear, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.one_x_one_conv = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=(1, 1))

    def forward(self, x, g):
        changed_x = self.one_x_one_conv(x)  # Make x as same dimension as g by 1x1 conv
        dim0, dim1 = changed_x.size()[2], changed_x.size()[3]
        changed_x = changed_x.view(-1, self.out_channel, dim0 * dim1)
        changed_x = torch.transpose(changed_x, 1, 2)

        compatibility_score = torch.bmm(changed_x, g.unsqueeze(2))  # batch_size x h*w x 1
        compatibility_score = F.softplus(compatibility_score)
        weights = compatibility_score / (torch.sum(compatibility_score, 1).unsqueeze(1) + .0001)
        # weights = (compatibility_score - torch.min(compatibility_score, 1)[0].unsqueeze(1))/torch.sum(compatibility_score,1).unsqueeze(1)
        # weights = F.softmax(compatibility_score,1)  # h*w x 1
        original_x = x.view(-1, self.in_channel, dim0 * dim1)  # B x self.in_channel x h*w
        final_output = torch.bmm(original_x, weights)

        return final_output.squeeze(2), weights.data.cpu().numpy().reshape(-1, dim0, dim1)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(3, 3), padding=(2, 2), dilation=2)
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=8, kernel_size=(3, 3), padding=(2, 2), dilation=2)
        self.conv3 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 3), padding=(2, 2), dilation=2)
        # self.conv_attention1 = Attention_linear(16, 64)
        self.conv_attention1 = Attention_nonlinear(16, 64, 64)
        self.conv4 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), padding=(2, 2), dilation=2)
        self.conv5 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), padding=(2, 2), dilation=2)
        # self.conv_attention2 = Attention_linear(32, 64)
        self.conv_attention2 = Attention_nonlinear(32, 64, 64)

        self.conv6 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), padding=(2, 2), dilation=2)
        # self.conv_attention3 = Attention_linear(32, 64)
        self.conv_attention3 = Attention_nonlinear(32, 64, 64)

        self.conv7 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding=(2, 2), dilation=2)
        self.conv5_drop = nn.Dropout(p=.2)
        self.fc1 = nn.Linear(in_features=80, out_features=32)
        self.fc2 = nn.Linear(in_features=32, out_features=10)
        self.fc3 = nn.Linear(in_features=10, out_features=2)

        for m in self.modules():

            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                # init.xavier_normal(m.weight.data, gain=nn.init.calculate_gain('relu'))
                init.xavier_uniform(m.weight.data, gain=nn.init.calculate_gain('relu'))
                # init.kaiming_uniform(m.weight.data)
                init.constant(m.bias, .1)

            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        x = F.avg_pool2d(x, (2, 2))

        x = F.elu(self.conv1(x))
        x = F.elu(self.conv2(x))
        x = F.max_pool2d(x, (2, 2))

        x = F.elu(self.conv3(x))
        x = F.max_pool2d(x, (2, 2))
        attention_x1 = x

        x = F.elu(self.conv4(x))
        x = F.elu(self.conv5(x))
        x = F.max_pool2d(x, (2, 2))
        attention_x2 = x

        x = F.elu(self.conv6(x))
        attention_x3 = x

        x = F.elu(self.conv7(x))

        x = F.adaptive_avg_pool2d(x, output_size=1)
        x = self.conv5_drop(x)
        x = F.elu(x)

        g = x.view(-1, 64)

        at1, at_map1 = self.conv_attention1(attention_x1, g)
        at2, at_map2 = self.conv_attention2(attention_x2, g)
        at3, at_map3 = self.conv_attention3(attention_x3, g)
        g_new = torch.cat((at1, at2, at3), 1)

        x = F.elu(self.fc1(g_new))
        x = F.dropout(x, p=.2)
        x = F.elu(self.fc2(x))
        x = self.fc3(x)
        return x


def train(epoch, model, dataloader, optimizer):
    model.train()

    train_loss = 0
    # correct = 0

    kont = 0
    for batch_id, (data, target) in enumerate(dataloader):
        kont += data.size(0)
        data, target = data.type(torch.FloatTensor).cuda(), target.type(torch.LongTensor).cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)

        # loss = F.cross_entropy(input=output, target=target)

        loss = F.cross_entropy(input=output, target=target)
        train_loss = train_loss + loss.data[0] * data.size(0)
        # train_loss = train_loss + loss.data[0]

        loss.backward()
        optimizer.step()

        if batch_id % 50 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_id * len(data), len(dataloader.dataset), 100. * batch_id / len(dataloader), loss.data[0]))

    train_loss /= kont
    print('\nTrain set: Average loss: {:.4f}'.format(train_loss))


def temporal_ensemble_train(nb_epoch, model, dataloader, optimizer, testloader, alpha = .6):
    model.train()
    x = np.linspace(0, 1, nb_epoch)
    auc_history = []
    for ep in range(nb_epoch):
        if ep ==0:
            old_pred = dict()
        train_loss = 0
        kont = 0

        for batch_id, (data, target, id) in enumerate(dataloader):
            kont += data.size(0)
            data, target= data.type(torch.FloatTensor).cuda(), target.type(torch.LongTensor).cuda()
            data, target = Variable(data), Variable(target)
            optimizer.zero_grad()
            output = model(data)

            current_pred = dict(zip(id, output))
            if ep ==0:
                old_pred.update(current_pred)
                loss = F.cross_entropy(input=output, target=target)
                loss.backward()
                optimizer.step()
                train_loss = train_loss + loss.data[0] * data.size(0)
            else:
                target_output = []
                ind = 0
                for key in id:
                    old_pred_key = old_pred[key]
                    old_pred_key = alpha * old_pred_key + (1-alpha) * output[ind]
                    old_pred[key] = old_pred_key
                    ind +=1
                    old_pred_key = old_pred_key / (1 - alpha ** ep)
                    target_output.append(old_pred_key.unsqueeze(0))

                target_output = torch.cat(target_output)
                target_output = target_output.detach()
                diag_loss = F.cross_entropy(input=output, target=target)
                temporal_loss = F.mse_loss(output,target_output)
                w = math.exp(-5*(1-x[ep-1])**2)
                loss = diag_loss + w * temporal_loss
                loss.backward()
                optimizer.step()
                train_loss = train_loss + loss.data[0] * data.size(0)

            if batch_id % 50 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                ep, batch_id * len(data), len(dataloader.dataset), 100. * batch_id / len(dataloader), loss.data[0]))

        train_loss /= kont
        print('\nTrain set: Average loss: {:.4f}'.format(train_loss))
        auc_history.append(auc_cal(model, testloader))
        print("Test AUC:", auc_history[-1])
    print(auc_history)










    # for batch_id, (data, target) in enumerate(dataloader):
    #     kont += data.size(0)
    #     data, target = data.type(torch.FloatTensor).cuda(), target.type(torch.LongTensor).cuda()
    #     data, target = Variable(data), Variable(target)
    #     optimizer.zero_grad()
    #     output = model(data)
    #
    #     # loss = F.cross_entropy(input=output, target=target)
    #
    #     loss = F.cross_entropy(input=output, target=target)
    #     train_loss = train_loss + loss.data[0] * data.size(0)
    #     # train_loss = train_loss + loss.data[0]
    #
    #     loss.backward()
    #     optimizer.step()
    #
    #     if batch_id % 50 == 0:
    #         print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
    #             epoch, batch_id * len(data), len(dataloader.dataset), 100. * batch_id / len(dataloader), loss.data[0]))
    #
    # train_loss /= kont
    # print('\nTrain set: Average loss: {:.4f}'.format(train_loss))


def test(model, testloader):
    model.eval()
    test_loss = 0
    correct = 0
    for _, (data, target,_) in enumerate(testloader):
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

    final_predicted = []
    final_target = []
    for _, (data, target, _) in enumerate(testloader):
        data, target = data.type(torch.FloatTensor).cuda(), target.type(torch.LongTensor).cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)

        test_loss = test_loss + F.cross_entropy(output, target, size_average=False).data[0]

        test_loss = test_loss + F.cross_entropy(output, target, size_average=False).data[0]
        pred = output.data.max(1, keepdim=True)[1]
        correct = correct + pred.eq(target.data.view_as(pred)).cpu().sum()

        output = np.exp(output.data.cpu().numpy())
        final_predicted.append(output / np.sum(output, axis=1).reshape(data.size(0), 1))
        final_target.append(target.data.cpu().numpy())

        # pred = output.data.max(1, keepdim=True)[1]
        # correct = correct + pred.eq(target.data.view_as(pred)).cpu().sum()
        #
        # output = np.exp(output.data.cpu().numpy())
        # predicted = output / np.sum(output, axis=1).reshape(len(testloader.dataset),1)
        # target = target.data.cpu().numpy()
        #
        # final_predicted.append(output / np.sum(output, axis=1).reshape(len(data), 1))
        # final_target.append(target.data.cpu().numpy())
        # fpr, tpr, thresholds = metrics.roc_curve(1 + target, predicted[:, 1], pos_label=2)

    final_predicted = np.concatenate(final_predicted, 0)
    final_target = np.concatenate(final_target, 0)
    fpr, tpr, thresholds = metrics.roc_curve(1 + final_target, final_predicted[:, 1], pos_label=2)

    test_loss /= len(testloader.dataset)
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
    root_dir = '/home/atin/rescaled_may_2018'
    # Read the patient id from disk for glaucoma and healthy
    patient_id_G = [x.split('_')[0] for x in os.listdir(root_dir + "/glaucoma/")]
    patient_id_H = [x.split('_')[0] for x in os.listdir(root_dir + "/healthy/")]
    freq_G = Counter(patient_id_G)
    freq_H = Counter(patient_id_H)
    # len(freq_G), len(freq_H)= (164, 2742)
    # Collect those patients which have all 6 slices
    G_exclude = [k for k, v in freq_G.items() if v != 6]
    H_exclude = [k for k, v in freq_H.items() if v != 6]
    # len(freq_G_exclude), len(freq_H_exclude) =(9, 76)

    # Remove those patients which don't have 6 slices.
    patient_id_G = set(patient_id_G) - set(G_exclude)
    patient_id_H = set(patient_id_H) - set(H_exclude)

    print(len(patient_id_G), len(patient_id_H))

    # patient_id_G = {x.split('_')[0] for x in os.listdir(root_dir+"/glaucoma/")}
    # patient_id_H = {x.split('_')[0] for x in os.listdir(root_dir+"/healthy/")}

    # random.seed(982)
    c = 12
    # c = 77898
    random.seed(c)
    print("random seed: {}".format(c))

    patient_id_G_test = random.sample(patient_id_G, 50)
    patient_id_H_test = random.sample(patient_id_H, 500)

    patient_id_G_train = list(patient_id_G.difference(patient_id_G_test))
    patient_id_H_train = list(patient_id_H.difference(patient_id_H_test))


    final_train = make_balance(patient_id_G_train, patient_id_H_train)
    final_test = patient_id_G_test + patient_id_H_test

    # transformed_images = Beijing_dataset(root_dir,patient_id_G_train,patient_id_H_train)

    transform_pipeline_train = tr.Compose([tr.ToTensor(), tr.AddChannel(axis=0), tr.TypeCast('float'),
                                           # tr.RangeNormalize(0,1),
                                           tr.RandomBrightness(-.2, .2),
                                           tr.RandomGamma(.8, 1.2),
                                           tr.RandomFlip(),
                                           tr.RandomAffine(rotation_range=5, translation_range=0.2,
                                                           zoom_range=(0.9, 1.1))])

    transform_pipeline_test = tr.Compose([tr.ToTensor(), tr.AddChannel(axis=0), tr.TypeCast('float')
                                          # tr.RangeNormalize(0, 1)
                                          ])

    transformed_images = Beijing_dataset(root_dir, patient_id_G_train, patient_id_H_train,final_train,
                                         transform=transform_pipeline_train)
    dataloader = DataLoader(transformed_images, batch_size=16, num_workers=10)
    testdataset = Beijing_dataset(root_dir, patient_id_G_test, patient_id_H_test, final_test, transform=transform_pipeline_test)
    testloader = DataLoader(testdataset, batch_size=16, shuffle=True, num_workers=20)

    # transformed_images = Beijing_dataset(root_dir, patient_id_G_train, patient_id_H_train,
    #                                      transform=transform_pipeline_train)
    # transformed_images = Beijing_dataset(root_dir,patient_id_G_train,patient_id_H_train,transform=transforms.Compose([Horizontalflip(),
    #                                                                    RandomRotate(angle_range=(0, 5), prob=0,
    #                                                                                 mode="constant"),
    #                                                                    Translation(h_shift_range=(0, 4),
    #                                                                                w_shift_range=(0, 8), prob=0,
    #                                                                                mode="constant"),
    #                                                                    ToTensor()]))

    # prob = np.array([len(patient_id_G_train), len(patient_id_H_train)])
    # prob = 1 / prob
    #
    # weight = [prob[0]] * len(patient_id_G_train) + [prob[1]] * len(patient_id_H_train)
    # sampler = WeightedRandomSampler(weight, len(weight))

    # dataloader = DataLoader(transformed_images, batch_size=32, num_workers=10, sampler=sampler)
    # dataloader = DataLoader(transformed_images, batch_size=16, num_workers=10, shuffle=True)

    # testdataset = Beijing_dataset(root_dir,patient_id_G_test,patient_id_H_test, transform= transforms.Compose([ToTensor()]))
    # testdataset = Beijing_dataset(root_dir, patient_id_G_test, patient_id_H_test, transform=transform_pipeline_test)

    # testdataset = Beijing_dataset(root_dir,patient_id_G_test,patient_id_H_test)

    # testloader = DataLoader(testdataset, batch_size=len(testdataset), shuffle=True, num_workers=20)

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
    optimizer = optim.Adam(model.parameters(), lr=.0001, weight_decay=.000001)
    # optimizer = optim.Adadelta(model.parameters(),weight_decay=.00001)

    nb_epoch = 1000
    temporal_ensemble_train(nb_epoch, model, dataloader, optimizer, testloader)
    # for epoch in range(1, nb_epoch + 1):
    #     train(epoch, model, dataloader, optimizer)
    #     print("Test AUC:", auc_cal(model, testloader))
    #     # test(model, testloader)

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
























