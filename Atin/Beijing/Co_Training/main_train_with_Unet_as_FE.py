from sys import path
path.insert(0,'/home/atin//New_deployed_projects/Glaucoma-diagnosis/')
from Atin.Beijing.Co_Training.diag_train import Net_final as Diag_model, train as diag_train, Beijing_dataset as Beijing_diag_dataset, AddGaussian, auc_cal
from Atin.Beijing.Co_Training.Seg_train import Seg_Model, seg_train, Beijing_Seg_Dataset, create_dark_mask, create_elastic_indices, train_transform, JaccardCriterion
from torchvision.transforms import ToPILImage

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
import numpy as np

import sys
import warnings

from collections import Counter
from sklearn import metrics

os.environ["CUDA_VISIBLE_DEVICES"]="1"
CUDA = True
LOG_INTERVAL = 10
Force_train = False


class Combined_Model(nn.Module):

    def __init__(self):
        super(Combined_Model, self).__init__()
        self.diag = Diag_model()
        # self.fc1 = nn.Linear(in_features=80, out_features=32)
        # self.fc2 = nn.Linear(in_features=32, out_features=10)
        # self.fc3 = nn.Linear(in_features=10, out_features=2)

    def forward(self, x, seg_model):
        seg_output = []
        for el in x:
            seg_output.append(seg_model(el).unsqueeze(0))
        seg_output = torch.cat(seg_output)
        # seg_output = seg_model(seg_output)
        final_output = self.diag(seg_output)
        return final_output

def combined_train(epoch, combined_model, seg_model, dataloader, optimizer):
    combined_model.train()
    seg_model.eval()

    train_loss = 0
    # correct = 0

    kont = 0
    for batch_id, (data, target) in enumerate(dataloader):
        kont += data.size(0)

        data = torch.transpose(data, 0, 1)
        data, target = data.type(torch.FloatTensor).cuda(), target.type(torch.LongTensor).cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = combined_model(data, seg_model)
        # loss = F.cross_entropy(input=output, target=target)

        loss = F.cross_entropy(input=output, target=target)
        train_loss = train_loss + loss.data[0] * data.size(1)
        # train_loss = train_loss + loss.data[0]

        loss.backward()
        optimizer.step()

        if batch_id % 50 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_id * len(data), len(dataloader.dataset), 100. * batch_id / len(dataloader), loss.data[0]))

    train_loss /= kont
    print('\nTrain set: Average loss: {:.4f}'.format(train_loss))



def combined_auc_cal(combined_model, seg_model, testloader):
    combined_model.eval()
    seg_model.eval()
    test_loss = 0
    correct = 0

    final_predicted = []
    final_target = []
    for _, (data, target) in enumerate(testloader):
        data = torch.transpose(data, 0, 1)
        data, target = data.type(torch.FloatTensor).cuda(), target.type(torch.LongTensor).cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data, seg_model)

        test_loss = test_loss + F.cross_entropy(output, target, size_average=False).data[0]

        test_loss = test_loss + F.cross_entropy(output, target, size_average=False).data[0]
        pred = output.data.max(1, keepdim=True)[1]
        correct = correct + pred.eq(target.data.view_as(pred)).cpu().sum()

        output = np.exp(output.data.cpu().numpy())
        final_predicted.append(output / np.sum(output, axis=1).reshape(data.size(1), 1))
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







if __name__ == "__main__":

    if not sys.warnoptions:
        warnings.simplefilter("ignore")

    #This is the location of the data in delta machine
    # root_dir = '/home/atin/DeployedProjects/data/Beijing'
    old_root_dir = '/data02/Atin/DeployedProjects/data'
    root_dir = '/home/atin/rescaled_march_2018'
    seg_loc = '/home/atin/DeployedProjects/Glaucoma-DL/Atin/Beijing/segmentation_data'
    model_repo_dir = '/home/atin/New_deployed_projects/Glaucoma-diagnosis/Atin/Beijing/Model_repo'

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

    c = 9856123
    c = 11345345
    # c = 77898
    random.seed(c)
    print("random seed: {}".format(c))

    patient_id_G_test = random.sample(patient_id_G,20)
    patient_id_H_test = random.sample(patient_id_H,200)

    patient_id_G_train = list(patient_id_G.difference(patient_id_G_test))
    patient_id_H_train = list(patient_id_H.difference(patient_id_H_test))





    #Training Unet START
    seg_patient_id = [x.split('_')[:3] for x in os.listdir(seg_loc)]
    seg_id = [p[0] for p in seg_patient_id]
    dark_mask = create_dark_mask()
    indices_x_clipped, indices_y_clipped = create_elastic_indices()

    seg_train_data = Beijing_Seg_Dataset(seg_patient_id, old_root_dir, seg_loc, (96, 288),
                                         train_transform(dark_mask, indices_x_clipped, indices_y_clipped))

    seg_train_loader = DataLoader(seg_train_data, batch_size=16, num_workers=10, shuffle=True)


    if os.path.exists(os.path.join(model_repo_dir,'unet.pt')) and not Force_train:
        unet_model = torch.load(os.path.join(model_repo_dir,'unet.pt'))

    else:
        unet_model = Seg_Model()
        model_parameters = filter(lambda p: p.requires_grad, unet_model.parameters())
        nb_params = sum([np.prod(p.size()) for p in model_parameters])
        print("no. of trainable parameters in Unet: {}".format((nb_params)))

        if CUDA: unet_model.cuda()

        optimizer = optim.Adam(unet_model.parameters(), lr=.001, weight_decay=.000001)
        criterion = JaccardCriterion()
        if CUDA: criterion.cuda()

        nb_epoch = 300

        for epoch in range(1, nb_epoch + 1):
            seg_train(epoch, unet_model, seg_train_loader, criterion, optimizer)
            # print("Test AUC:", auc_cal(model, testloader))
            # test(model, testloader)
        torch.save(unet_model, os.path.join(model_repo_dir, 'unet.pt'))

    #Training UNET END


    #Training Diag network START
    transform_pipeline_train = tr.Compose(
        [
         # AddGaussian(),
         # AddGaussian(ismulti=False),
         tr.ToTensor(), tr.AddChannel(axis=0), tr.TypeCast('float'),
         # Attenuation((-.001, .1)),
         # tr.RangeNormalize(0,1),
         tr.RandomBrightness(-.2, .2),
         tr.RandomGamma(.9, 1.1),
         tr.RandomFlip(),
         tr.RandomAffine(rotation_range=5, translation_range=0.2
                         # zoom_range=(0.9, 1.1)
                         )])

    transform_pipeline_test = tr.Compose([tr.ToTensor(), tr.AddChannel(axis=0), tr.TypeCast('float')
                                          # tr.RangeNormalize(0, 1)
                                          ])

    transformed_images = Beijing_diag_dataset(root_dir, patient_id_G_train, patient_id_H_train, resize_dim= (96,288),
                                         transform=transform_pipeline_train)

    prob = np.array([len(patient_id_G_train), len(patient_id_H_train)])
    prob = 1 / prob

    weight = [prob[0]] * len(patient_id_G_train) + [prob[1]] * len(patient_id_H_train)
    sampler = WeightedRandomSampler(weight, len(weight))

    dataloader = DataLoader(transformed_images, batch_size=16, num_workers=10, sampler=sampler)
    # dataloader = DataLoader(transformed_images, batch_size=16, num_workers=10, shuffle=True)

    testdataset = Beijing_diag_dataset(root_dir, patient_id_G_test, patient_id_H_test, resize_dim= (96,288),transform=transform_pipeline_test)
    # testloader = DataLoader(testdataset, batch_size=len(testdataset), shuffle=True, num_workers=20)
    testloader = DataLoader(testdataset, batch_size=16, shuffle=True, num_workers=20)


    model = Combined_Model()
    print(model)

    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    nb_params = sum([np.prod(p.size()) for p in model_parameters])
    print("no. of trainable parametes is: {}".format((nb_params)))

    if CUDA: model.cuda()

    # optimizer = optim.SGD(model.parameters(), lr=.01, momentum=.5)
    # optimizer = optim.Adam(model.parameters(), lr=.001)
    optimizer = optim.Adam(model.parameters(), lr=.001, weight_decay=.000001)

    # optimizer = optim.Adadelta(model.parameters(),weight_decay=.00001)

    nb_epoch = 50
    for epoch in range(1, nb_epoch + 1):
        combined_train(epoch, model,unet_model, dataloader, optimizer)
        print("Test AUC:", combined_auc_cal(model, unet_model, testloader))






















