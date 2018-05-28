#Adds new class of Augmentation like zoom, brigtness etc.
#import torchsample.transforms as tr is the new library
#Its the most basic single input CNN
#Adding attention to this

#TRAINING multi input cnn
#ADDING Gated attention insted of max from 6 slices
#Works great so far


from sys import path
path.insert(0,'/home/atin/DeployedProjects/Glaucoma-DL/')

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
import torchsample.transforms as tr


import random
import os
import matplotlib.pyplot as plt
from scipy.misc import imresize

import sys
import warnings

from collections import Counter
from sklearn import metrics

os.environ["CUDA_VISIBLE_DEVICES"]="1"
CUDA = True
LOG_INTERVAL = 10


def beijing_pre_process(path):

    patient_id = list({x.split('_')[0] for x in os.listdir(path)})


def preprocess_image(loc, transform = None):
    im = plt.imread(loc)
    im = imresize(im, (143, 400),interp='nearest') #original image size is
    # im = imresize(im, (200, 400), interp='nearest')
    #im = imresize(im, (125, 350), interp='nearest')
    im = im / 255
    # im = np.expand_dims(im, 0)
    # im = im.transpose((1, 2, 0))

    if transform is not None:
        im = transform(im)

    return im




class Beijing_dataset(Dataset):


    def __init__(self,root_dir, patient_id_G, patient_id_H,transform=None):
        self.root_dir = root_dir
        self.patient_id_G = patient_id_G
        self.patient_id_H = patient_id_H
        self.transform = transform

    def __len__(self):

        return len(self.patient_id_G) + len(self.patient_id_H)

    def __getitem__(self, idx):

        if idx < len(self.patient_id_G):
            file_name = "glaucoma/"+self.patient_id_G[idx]
            train_y = 1
        else:
            idx_new = idx-len(self.patient_id_G)
            file_name = "healthy/"+ self.patient_id_H[idx_new]
            train_y = 0
        slices = ["_1", "_2", "_3", "_4", "_5", "_6"]

        final_image_array = []
        for slice in slices:
            if idx < len(self.patient_id_G):
                slice_file_name = file_name+slice+"_G.png"
            else:
                slice_file_name = file_name + slice + "_H.png"
            loc = os.path.join(self.root_dir, slice_file_name)
            im = preprocess_image(loc, self.transform)
            final_image_array.append(im.unsqueeze(0))



        return torch.cat(final_image_array), train_y


class Attention_linear(nn.Module):

    def __init__(self,in_channel,out_channel):
        super(Attention_linear, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.one_x_one_conv = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=(1, 1))


    def forward(self, x, g):
        changed_x = self.one_x_one_conv(x) #Make x as same dimension as g by 1x1 conv
        dim0, dim1 = changed_x.size()[2], changed_x.size()[3]
        changed_x = changed_x.view(-1, self.out_channel, dim0 * dim1)
        changed_x = torch.transpose(changed_x, 1, 2)

        compatibility_score = torch.bmm(changed_x, g.unsqueeze(2))  #batch_size x h*w x 1
        compatibility_score = F.softplus(compatibility_score)
        weights = compatibility_score/(torch.sum(compatibility_score,1).unsqueeze(1)+.0001)
        # weights = (compatibility_score - torch.min(compatibility_score, 1)[0].unsqueeze(1))/torch.sum(compatibility_score,1).unsqueeze(1)
        # weights = F.softmax(compatibility_score,1)  # h*w x 1
        original_x = x.view(-1, self.in_channel, dim0 * dim1)  # B x self.in_channel x h*w
        final_output = torch.bmm(original_x, weights)

        return final_output.squeeze(2), weights.data.cpu().numpy().reshape(-1, dim0, dim1)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size= (3,3), padding=(2,2), dilation=2)
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=8, kernel_size= (3,3), padding=(2,2), dilation=2)
        self.conv3 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3,3), padding=(2,2), dilation=2)
        self.conv_attention1 = Attention_linear(16, 64)
        self.conv4 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3,3), padding=(2,2), dilation=2)
        self.conv5 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3,3), padding=(2,2), dilation=2)
        self.conv_attention2 = Attention_linear(32, 64)
        self.conv6 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), padding=(2, 2), dilation=2)
        self.conv_attention3 = Attention_linear(32, 64)
        self.conv7 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding=(2, 2), dilation=2)
        self.conv5_drop = nn.Dropout(p=.2)
        self.fc1 = nn.Linear(in_features=80, out_features=10)
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
        attention_x1 = x

        x = F.elu(self.conv4(x))
        x = F.elu(self.conv5(x))
        x = F.max_pool2d(x, (2, 2))
        attention_x2 = x

        x = F.elu(self.conv6(x))
        attention_x3 = x

        x = F.elu(self.conv7(x))

        x = F.adaptive_avg_pool2d(x,output_size=1)
        x = self.conv5_drop(x)
        x = F.elu(x)

        g = x.view(-1, 64)

        at1, at_map1 = self.conv_attention1(attention_x1, g)
        at2, at_map2 = self.conv_attention2(attention_x2, g)
        at3, at_map3 = self.conv_attention3(attention_x3, g)
        g_new = torch.cat((at1, at2, at3), 1)

        # x = F.elu(self.fc1(g_new))
        # x = self.fc2(x)
        return g_new
        #return F.log_softmax(x, dim=1)



class Net_final(nn.Module):

    def __init__(self):
        super(Net_final, self).__init__()
        self.nn_block = Net()
        self.fc1 = nn.Linear(in_features=80, out_features=10)
        self.fc2 = nn.Linear(in_features=10, out_features=2)
        self.sigmoid_att = nn.Linear(in_features=80, out_features=10, bias=False)
        self.tanh_att = nn.Linear(in_features=80, out_features=10, bias= False)
        self.combine = nn.Linear(in_features=10, out_features=1, bias= False)

        for m in self.modules():

            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                #init.xavier_normal(m.weight.data, gain=nn.init.calculate_gain('relu'))
                init.xavier_uniform(m.weight.data, gain=nn.init.calculate_gain('relu'))
                #init.kaiming_uniform(m.weight.data)
                if m.bias is not None:
                    init.constant(m.bias, .1)

            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self,y):
        output = []
        weight = []
        for x in y:
            ind_output = self.nn_block(x)
            h = F.sigmoid(self.sigmoid_att(ind_output)) * F.tanh(self.tanh_att(ind_output))
            weight.append(self.combine(h).unsqueeze(0))
            output.append(ind_output.unsqueeze(0))

        # weight = torch.cat(weight)
        # weight = torch.transpose(weight,0,1).squeeze(2)
        # weight = F.softmax(weight,1).unsqueeze(2)
        final_output = torch.cat(output)
        # final_output = torch.bmm(final_output.permute(1,2,0), weight).squeeze(2)

        # final_output, _ = torch.max(final_output,0)
        final_output = torch.mean(final_output, 0)

        # final_output = F.dropout(final_output, p =.1)
        x = F.elu(self.fc1(final_output))
        x = self.fc2(x)

        return x







def train(epoch,model,dataloader,optimizer):
    model.train()

    train_loss = 0
    #correct = 0

    kont = 0
    for batch_id, (data, target) in enumerate(dataloader):
        kont += data.size(0)

        data = torch.transpose(data,0,1)
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
        data = torch.transpose(data, 0, 1)
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

    for _, (data, target) in enumerate(testloader):
        data = torch.transpose(data, 0, 1)
        data, target = data.type(torch.FloatTensor).cuda(), target.type(torch.LongTensor).cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)

        test_loss = test_loss + F.cross_entropy(output, target, size_average=False).data[0]
        pred = output.data.max(1, keepdim=True)[1]
        correct = correct + pred.eq(target.data.view_as(pred)).cpu().sum()

        output = np.exp(output.data.cpu().numpy())
        predicted = output / np.sum(output, axis=1).reshape(220,1)
        target = target.data.cpu().numpy()
        fpr, tpr, thresholds = metrics.roc_curve(1 + target, predicted[:, 1], pos_label=2)



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

    #This is the location of the data in delta machine
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

    # random.seed(828989)
    # random.seed(82)
    # random.seed(672) #Bad sometimes
    # # random.seed(67672)
    # # random.seed(312)
    # # random.seed(31256)
    # random.seed(675906)

    c = 675906
    # c = 77898
    random.seed(c)
    print("random seed: {}".format(c))






    patient_id_G_test = random.sample(patient_id_G,20)
    patient_id_H_test = random.sample(patient_id_H,200)

    patient_id_G_train = list(patient_id_G.difference(patient_id_G_test))
    patient_id_H_train = list(patient_id_H.difference(patient_id_H_test))

    transform_pipeline_train = tr.Compose([tr.ToTensor(), tr.AddChannel(axis=0), tr.TypeCast('float'),
                                           # tr.RangeNormalize(0,1),
                                           tr.RandomBrightness(-.2, .2),
                                           tr.RandomGamma(.9, 1.1),
                                           tr.RandomFlip(),
                                           tr.RandomAffine(rotation_range=5, translation_range=0.2,
                                                           zoom_range=(0.9, 1.1))])

    transform_pipeline_test = tr.Compose([tr.ToTensor(), tr.AddChannel(axis=0), tr.TypeCast('float')
                                          # tr.RangeNormalize(0, 1)
                                          ])

    transformed_images = Beijing_dataset(root_dir, patient_id_G_train, patient_id_H_train,
                                         transform=transform_pipeline_train)

    #transformed_images = Beijing_dataset(root_dir,patient_id_G_train,patient_id_H_train)
    # transformed_images = Beijing_dataset(root_dir,patient_id_G_train,patient_id_H_train,transform=transforms.Compose([Horizontalflip(),
    #                                                                    RandomRotate(angle_range=(0, 5), prob=0,
    #                                                                                 mode="constant"),
    #                                                                    Translation(h_shift_range=(0, 4),
    #                                                                                w_shift_range=(0, 8), prob=0,
    #                                                                                mode="constant"),
    #                                                                    ToTensor()]))


    prob = np.array([len(patient_id_G_train),len(patient_id_H_train)])
    prob = 1/prob

    weight = [prob[0]]*len(patient_id_G_train) + [prob[1]]*len(patient_id_H_train)
    sampler = WeightedRandomSampler(weight,len(weight))

    dataloader = DataLoader(transformed_images,batch_size=16, num_workers=10,sampler=sampler)
    #dataloader = DataLoader(transformed_images, batch_size=16, num_workers=10, shuffle=True)

    testdataset = Beijing_dataset(root_dir, patient_id_G_test, patient_id_H_test, transform=transform_pipeline_test)



    # testdataset = Beijing_dataset(root_dir,patient_id_G_test,patient_id_H_test, transform= transforms.Compose([ToTensor()]))
    #testdataset = Beijing_dataset(root_dir,patient_id_G_test,patient_id_H_test)

    testloader = DataLoader(testdataset, batch_size=len(testdataset), shuffle=True, num_workers=20)

    model = Net_final()
    print(model)

    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    nb_params = sum([np.prod(p.size()) for p in model_parameters])
    print("no. of trainable parametes is: {}".format((nb_params)))

    model.cuda()

    # optimizer = optim.SGD(model.parameters(), lr=.01, momentum=.5)
    # optimizer = optim.Adam(model.parameters(), lr=.001)
    optimizer = optim.Adam(model.parameters(), lr=.001, weight_decay=.000001)

    # optimizer = optim.Adadelta(model.parameters(),weight_decay=.00001)

    nb_epoch = 50
    for epoch in range(1, nb_epoch + 1):
        train(epoch, model, dataloader,optimizer)
        print("Test AUC:", auc_cal(model, testloader))
        #test(model, testloader)

    # optimizer = optim.Adam(model.parameters(), lr=.0001, weight_decay=.000001)
    #
    # # optimizer = optim.Adadelta(model.parameters(),weight_decay=.5)
    # nb_epoch = 100
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
























