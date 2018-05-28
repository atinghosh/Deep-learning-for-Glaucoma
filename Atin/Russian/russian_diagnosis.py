from Atin.Russian.helper import *

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
#cuda.set_device(1)
import torch.nn.init as init
from sklearn import metrics


class GlaucomaDataset(Dataset):

    def __init__(self, train_x, train_y, transform=None):
        self.train_x = train_x
        self.train_y = train_y
        self.transform = transform

    def __len__(self):
        return len(self.train_y)

    def __getitem__(self, idx):
        sample = self.train_x[idx]
        if self.transform:
            sample = self.transform(sample)
        return sample, self.train_y[idx]



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



if __name__ == '__main__':


    X_train = np.load('/home/atin/DeployedProjects/pytorch_delta/Russian_data/X_train.npy')
    y_train = np.load('/home/atin/DeployedProjects/pytorch_delta/Russian_data/y_train.npy')

    X_test = np.load("/home/atin/DeployedProjects/pytorch_delta/Russian_data/X_test.npy")
    y_test = np.load("/home/atin/DeployedProjects/pytorch_delta/Russian_data/y_test.npy")

    transformed_images = GlaucomaDataset(X_train,
                                         y_train,
                                         transform=transforms.Compose([Horizontalflip(),
                                                                       RandomRotate(angle_range=(0, 5), prob=0,
                                                                                    mode="constant"),
                                                                       Translation(h_shift_range=(0, 4),
                                                                                   w_shift_range=(0, 8), prob=0,
                                                                                   mode="constant"),
                                                                       ToTensor()]))

    dataloader = DataLoader(transformed_images, batch_size=32, shuffle=True, num_workers=20)

    testdtaset = GlaucomaDataset(X_test, y_test, transform=transforms.Compose([ToTensor()]))
    testloader = DataLoader(testdtaset, batch_size=148, shuffle=True, num_workers=20)

    model = Net()
    print(model)

    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    nb_params = sum([np.prod(p.size()) for p in model_parameters])
    print("no. of trainable parametes is: {}".format((nb_params)))

    model.cuda()

    #optimizer = optim.SGD(model.parameters(), lr=.01, momentum=.5)
    optimizer = optim.Adam(model.parameters(), lr=.0001)
    #optimizer = optim.Adadelta(model.parameters(),weight_decay=.5)

    nb_epoch = 20
    for epoch in range(1, nb_epoch+1):
        train(epoch,model,dataloader,optimizer)
        test(model,testloader)






    predicted_test_prob = prediction(X_test)
    predicted_train_prob = prediction(X_train)

    fpr, tpr, thresholds = metrics.roc_curve(1 + y_test, predicted_test_prob[:, 1], pos_label=2)
    print("test AUC:\n")
    print(metrics.auc(fpr, tpr))

    fpr1, tpr1, thresholds = metrics.roc_curve(1 + y_train, predicted_train_prob[:, 1], pos_label=2)
    print("train AUC:\n")
    print(metrics.auc(fpr1, tpr1))


    #
    # plt.rcParams['figure.figsize'] = (20, 15)
    #
    # plt.subplot(2, 2, 1)
    # fpr, tpr, thresholds = metrics.roc_curve(1 + y_test, predicted_test_prob[:, 1], pos_label=2)
    # plt.plot(1 - fpr, tpr, "b-", linewidth=2, alpha=0.7,
    #          label='AUC %0.2f' % (metrics.auc(fpr, tpr)))
    # plt.plot(1 - fpr, fpr, "r--", linewidth=1, alpha=0.5)
    # plt.xlabel("false negative (specificity)")
    # plt.ylabel("true positive rate (sensitivity)")
    # plt.title("Test: ROC curve")
    # plt.legend(loc="lower left")
    # plt.grid(True)
    #
    # plt.subplot(2, 2, 2)
    # fpr, tpr, thresholds = metrics.roc_curve(1 + y_train, predicted_train_prob[:, 1], pos_label=2)
    # plt.plot(1 - fpr, tpr, "b-", linewidth=2, alpha=0.7,
    #          label='AUC %0.2f' % (metrics.auc(fpr, tpr)))
    # plt.plot(1 - fpr, fpr, "r--", linewidth=1, alpha=0.5)
    # plt.xlabel("false negative (specificity)")
    # plt.ylabel("true positive (sensitivity)")
    # plt.title("Train: ROC curve")
    # plt.legend(loc="lower left")
    # plt.grid(True)
    #
    # plt.subplot(2, 2, 3)
    # _, _, _ = plt.hist(predicted_test_prob[:, 1], bins=20, normed=True, alpha=0.5)
    # plt.xlim(0, 1)
    # plt.title("Distribution of Test Probability")
    # plt.xlabel("Glaucoma Probability")
    #
    # plt.subplot(2, 2, 4)
    # _, _, _ = plt.hist(predicted_train_prob[:, 1], bins=20, normed=True, alpha=0.5)
    # plt.xlim(0, 1)
    # plt.title("Distribution of Train Probability")
    # plt.xlabel("Glaucoma Probability")
    #
    # plt.savefig('result.png')



    # testdtaset = GlaucomaDataset(X_test, y_test, transform=transforms.Compose([ToTensor()]))
    # testloader = DataLoader(testdtaset, batch_size=148, shuffle=False, num_workers=20)
    #
    # model.eval()
    # for _, (data, target) in enumerate(testloader):
    #     data, target = data.type(torch.FloatTensor).cuda(), target.type(torch.LongTensor).cuda()
    #     data, target = Variable(data, volatile=True), Variable(target)
    #     output = model(data)
    #     output = output.data.cpu().numpy()
    #     predicted_test_prob = np.exp(output)
    #     break
    #
    # traindtaset = GlaucomaDataset(X_test, y_test, transform=transforms.Compose([ToTensor()]))
    # trainloader = DataLoader(traindtaset, batch_size=X_train.shape[0], shuffle=False, num_workers=20)
    # for _, (data, target) in enumerate(trainloader):
    #     data, target = data.type(torch.FloatTensor).cuda(), target.type(torch.LongTensor).cuda()
    #     data, target = Variable(data, volatile=True), Variable(target)
    #     output = model(data)
    #     output = output.data.cpu().numpy()
    #     predicted_train_prob = np.exp(output)
    #     break
    #
    # plt.rcParams['figure.figsize'] = (20, 15)
    #
    # plt.subplot(2, 2, 1)
    # fpr, tpr, thresholds = metrics.roc_curve(1 + y_test, predicted_test_prob[:, 1], pos_label=2)
    # plt.plot(1 - fpr, tpr, "b-", linewidth=2, alpha=0.7,
    #          label='AUC %0.2f' % (metrics.auc(fpr, tpr)))
    # plt.plot(1 - fpr, fpr, "r--", linewidth=1, alpha=0.5)
    # plt.xlabel("false negative (specificity)")
    # plt.ylabel("true positive rate (sensitivity)")
    # plt.title("Test: ROC curve")
    # plt.legend(loc="lower left")
    # plt.grid(True)
    #
    # plt.subplot(2, 2, 2)
    # fpr, tpr, thresholds = metrics.roc_curve(1 + y_train, predicted_train_prob[:, 1], pos_label=2)
    # plt.plot(1 - fpr, tpr, "b-", linewidth=2, alpha=0.7,
    #          label='AUC %0.2f' % (metrics.auc(fpr, tpr)))
    # plt.plot(1 - fpr, fpr, "r--", linewidth=1, alpha=0.5)
    # plt.xlabel("false negative (specificity)")
    # plt.ylabel("true positive (sensitivity)")
    # plt.title("Train: ROC curve")
    # plt.legend(loc="lower left")
    # plt.grid(True)
    #
    # plt.subplot(2, 2, 3)
    # _, _, _ = plt.hist(predicted_test_prob[:, 1], bins=20, normed=True, alpha=0.5)
    # plt.xlim(0, 1)
    # plt.title("Distribution of Test Probability")
    # plt.xlabel("Glaucoma Probability")
    #
    # plt.subplot(2, 2, 4)
    # _, _, _ = plt.hist(predicted_train_prob[:, 1], bins=20, normed=True, alpha=0.5)
    # plt.xlim(0, 1)
    # plt.title("Distribution of Train Probability")
    # plt.xlabel("Glaucoma Probability")
    #
    # plt.savefig("AUC curve")















