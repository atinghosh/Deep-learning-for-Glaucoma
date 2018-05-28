import argparse
import time
import os
import gc
import glob
import csv
import itertools
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.utils as nnUtils
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch import cuda
from torchvision import transforms as T
from visdom import Visdom
from torchnet import meter
from torchnet.logger import VisdomLogger, VisdomPlotLogger, VisdomSaver
from PIL import Image
from scipy import misc
import pickle

import Huy.data as data
import Huy.model as model
from Huy.model import Model, JaccardCriterion, JaccardMinMaxCriterion

parser = argparse.ArgumentParser(description='Option for Glaucoma')
#------------------------------------------------------------------- data-option
parser.add_argument('--trainSet',  type=list, default=['./data/SERIsegmentedImagesT.npy', './data/SERIsegmentedLabelT.npy'],
                    help='location of training data')
parser.add_argument('--valSet',  type=list, default=['./data/SERIsegmentedImagesV.npy', './data/SERIsegmentedLabelV.npy'],
                    help='location of validation data')
parser.add_argument('--nthreads',  type=int, default=4,
                    help='number of threads for data loader')
parser.add_argument('--batch_size', type=int, default=5, metavar='N',
                    help='train batch size')
parser.add_argument('--val_batch_size', type=int, default=5, metavar='N',
                    help='val batch size')
#------------------------------------------------------------------ model-option
parser.add_argument('--ispretrain', type=bool, default=False,
                    help='is pretrain mode')
parser.add_argument('--pretrained_model', type=str, default='',
                    help='pretrain model location')
parser.add_argument('--depth', type=int, default=3,
                    help='encoder and decoder depth')
parser.add_argument('--dims', type=int, default=2,
                    help='number of data dimension')
parser.add_argument('--block', type=str, default='ResBasicBlock',
                    help='block type of network')
parser.add_argument('--inplanes', type=int, default=1,
                    help='number of input channels')
parser.add_argument('--planes', type=int, default=16,
                    help='number of 1st output block channels')
parser.add_argument('--dilation', type=int, default=1,
                    help='dilation of conv layer')
parser.add_argument('--bias', type=bool, default=True,
                    help='use bias for conv layer')
parser.add_argument('--relu_type', type=str, default='ELU',
                    help='activation unit type')
parser.add_argument('--nClasses', type=int, default=5,
                    help='number of classes')
parser.add_argument('--up_mode', type=str, default='upsample',
                    help='up mode of up sample module')
parser.add_argument('--merge_mode', type=str, default='concat',
                    help='merge mode of decoder module')
parser.add_argument('--planes_expansion', type=int, default=1,
                    help='increase number of planes with follow number')
#--------------------------------------------------------------- training-option
parser.add_argument('--seed', type=int, default=1234,
                    help='random seed')
parser.add_argument('--gpus', type=list, default=[0],
                    help='list of GPUs in use')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA, in case of gups=0, it must be False')
#optimizer-option
parser.add_argument('--grad_clip', type=float, default=0,
                    help='gradient clipping, should use only in case of rnn')
parser.add_argument('--optim_algor', type=str, default='Adam',
                    help='optimization algorithm: Adam or Adadelta')
parser.add_argument('--lr', type=float, default=1e-4,
                    help='learning rate (Adam, 1e-3) or coef that scale delta before it applied to the params (Adadelta, 1.0)')
parser.add_argument('--betas', type=tuple, default=(0.9, 0.999),
                    help='coefficients used for computing running averages of gradient and its square (Adam)')
parser.add_argument('-rho', type=float, default=0.9,
                    help='coef for computing a running average squared gradients (Adadelta)')
parser.add_argument('--eps', type=float, default=1e-8,
                    help='numerical stability')
parser.add_argument('--weight_decay', type=float, default=0,
                    help='coef for L2 regularization')
#saving-option
parser.add_argument('--epochs', type=int, default=2000,
                    help='number of epochs')
parser.add_argument('--test_interval', type=int, default=1,
                    help='epoch interval of evaluation on test set')
parser.add_argument('--checkpoint_interval', type=int, default=1,
                    help='epoch interval of saving checkpoint')
parser.add_argument('--save_path', type=str, default='checkpoint',
                    help='directory for saving checkpoint')
parser.add_argument('--resume_checkpoint', type=str, default='',
                    help='location of saved checkpoint')
#only prediction-option
parser.add_argument('--trained_model', type=str, default='',
                    help='location of trained checkpoint')

opt = parser.parse_args()
opt.block = getattr(model, opt.block)
if len(opt.gpus) > 0:
    opt.cuda = True
    cuda.set_device(opt.gpus[0])
else:
    opt.cuda = False

# Set the random seed manually for reproducibility.
torch.manual_seed(opt.seed)
if opt.cuda:
    torch.cuda.manual_seed(opt.seed)

class Main:
    
    def __init__(self, opt):
        self.opt = opt
        os.makedirs(self.opt.save_path, exist_ok=True)
        self.trainLoader, self.valLoader= self.make_data(opt)
        self.model, self.criterion = self.make_model_criterion(opt)
        self.optimizer = self.make_optimizer(opt, self.model)
        self.lossmeter = meter.AverageValueMeter()
        self.accmeter = meter.AverageValueMeter()
        
    def make_model_criterion(self, opt):
        "create model, criterion"
        #if use pretrained model (load pretrained weight)
        if opt.pretrained_model:
            print("Loading pretrained model {} \n".format(opt.pretrained_model))
            model = Model(opt)
            pretrained_state = torch.load(opt.pretrained_model, map_location=lambda storage, loc: storage, pickle_module=pickle)['modelstate']
            model.load_state_dict(pretrained_state)
        else:
            model = Model(opt)
        #criterion
        criterion = JaccardCriterion(size_average=opt.size_average, reduce=opt.reduce) #if not opt.ispretrain else nn.BCELoss()
        return model, criterion
    
    def make_optimizer(self, opt, model, param_groups=[]):
        parameters = filter(lambda p: p.requires_grad, model.parameters())
        if len(param_groups) > 0:
            lr = param_groups[0]['lr']
            betas = param_groups[0]['betas']
            eps = param_groups[0]['eps']
            weight_decay = param_groups[0]['weight_decay']
        else:
            lr = opt.lr
            betas = opt.betas
            eps = opt.eps
            weight_decay = opt.weight_decay
        optimizer = getattr(optim, self.opt.optim_algor)(parameters,
                                                         lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        return optimizer
    
    def make_data(self, opt):
        #get data
        trainSet = data.get_trainSet(opt.trainSet)
        valSet = data.get_valSet(opt.valSet)
        #dataloader
        trainLoader = DataLoader(dataset=trainSet, num_workers=opt.nthreads, batch_size=opt.batch_size, shuffle=True)
        valLoader = DataLoader(dataset=valSet, num_workers=opt.nthreads, batch_size=opt.val_batch_size, shuffle=False)
        return trainLoader, valLoader
    
    def update_lr(self, epoch):
        lr = self.opt.lr * (0.5 ** (epoch // 500))
        if lr < 5e-5:
            lr = 5e-5
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        return lr
    
    #===========================================================================
    # Training and Evaluating
    #===========================================================================
    
    def scoremeter(self, output, target, isvolatile):
        """ouput: tensor N*Classes*H*W 
        target: N*C*H*W
        """
        jaccscore = self.jaccardscore(output, target)
        self.accmeter.add(jaccscore)
    
    def jaccardscore(self, output, target):
        """ouput tensor: N*Classes*H*W
        target tensor: N*C*H*W
        """
        if target.type() != output.type():
            target.data = target.type(output.type()) #should be ok when target contains binary value
        numerator = torch.min(output, target)
        denominator = torch.max(output, target)
        numerator = numerator.sum(dim=3).sum(dim=2)
        denominator = denominator.sum(dim=3).sum(dim=2)
        jacc_score = numerator/(denominator + 1e-6) #size N*C
        jacc_score = jacc_score.mean(dim=1)
        return jacc_score.mean()
    
    def predictor(self, output):
        "ouput: tensor batch*N_classes*H*W \
        target: tensor batch*H*W"
        mask = output.max(1)[1]
        return mask
    
    def resetmeter(self):
        self.lossmeter.reset()
        self.accmeter.reset()
    
    def evaluate(self, opt, model, criterion, dataloader):
        gc.collect()
        model.eval()
        self.resetmeter()
        
        for batch in dataloader:
            input = Variable(batch[0], volatile=True)
            target = Variable(batch[1], volatile=True)
            if len(opt.gpus) > 1:
                input = input.cuda(opt.gpus[0], True)
                target = target.cuda(opt.gpus[0], True)  
            elif opt.cuda:
                input = input.cuda(opt.gpus[0])
                target = target.cuda(opt.gpus[0])
            output = model(input)
            loss = criterion(output, target)
            
            #compute score
            self.scoremeter(output.cpu().data, target.cpu().data, target.volatile)
            self.lossmeter.add(loss.cpu().data[0])
        
        model.train()
        return self.lossmeter.value()[0], self.accmeter.value()[0]
    
    def train(self, opt, model, criterion, dataloader, optimizer, epoch):
        model.train()
        self.resetmeter()
        
        for iteration, batch in enumerate(dataloader, 1):
            start_time = time.time()
            input = Variable(batch[0])
            target = Variable(batch[1])
            if len(opt.gpus) > 1:
                input = input.cuda(opt.gpus[0], True)
                target = target.cuda(opt.gpus[0], True)
            elif opt.cuda:
                input = input.cuda()
                target = target.cuda()
            #zero gradient first,then forward
            optimizer.zero_grad()
            output = model(input)
            loss = criterion(output, target)
            loss.backward()
            
            optimizer.step()
            
            #compute score
            self.scoremeter(output.cpu().data, target.cpu().data, target.volatile)
            self.lossmeter.add(loss.cpu().data[0])
            
            #print
            eslapsed = time.time() - start_time
            print('| epoch {:3d} | {:3d}/{:3d} ith_batch | time(s) {:5.2f} | loss {:5.2f}'.format(
                                                                                                  epoch, iteration, len(dataloader), eslapsed, loss.data[0]))
        return self.lossmeter.value()[0], self.accmeter.value()[0]
        
    def execute(self):
        print(self.opt)
        print('\n')
        start_epoch = 1
        best_result = 0.
        best_flag = False
        
        #resume from saved checkpoint
        if self.opt.resume_checkpoint:
            print('Resuming checkpoint at {}'.format(self.opt.resume_checkpoint))
            checkpoint = torch.load(self.opt.resume_checkpoint, map_location=lambda storage, loc: storage, pickle_module=pickle)
            model_state = checkpoint['modelstate']
            self.model.load_state_dict(model_state)
            optim_state = checkpoint['optimstate']
            self.optimizer = self.make_optimizer(self.opt, self.model, param_groups=optim_state['param_groups'])
            start_epoch = checkpoint['epoch']+1
            best_result = checkpoint['best_result']
        
        #DataParallel for multiple GPUs:
        if len(self.opt.gpus) > 1:
            self.model = nn.DataParallel(self.model, device_ids=self.opt.gpus, dim=0)
            self.model.cuda()
            self.criterion.cuda()
        elif self.opt.cuda:
            self.model.cuda()
            self.criterion.cuda()
                
        #visualization
        viz = Visdom()
        visdom_saver = VisdomSaver([viz.env])
        train_loss_logger = VisdomPlotLogger('line', opts={'title': 'Train Loss'})
        val_loss_logger = VisdomPlotLogger('line', opts={'title': 'Val Loss'})
        acc_logger = VisdomPlotLogger('line', opts={'title': 'Class Acc', 'legend': ['train_score', 'val_score']})
        
        print('Start training optim {}, cuda {}'.format(self.opt.optim_algor, self.opt.cuda))
        
        for epoch in range(start_epoch, self.opt.epochs+1):
            #update learning rate
            if self.opt.optim_algor is not 'Adadelta':
                lr = self.update_lr(epoch)
            else:
                lr = self.opt.lr
            #let's go
            print('\n')
            print('-' * 65)
            print('{}'.format(time.asctime(time.localtime())))
            print(' **Training epoch {}, lr {}'.format(epoch, lr))
            
            start_time = time.time()
            train_avg_loss, train_avg_acc = self.train(self.opt, self.model,
                                                       self.criterion, self.trainLoader, self.optimizer, epoch)
            print('| finish training on epoch {:3d} | time(s) {:5.2f} | loss {:5.2f} | score {:5.2f}'.format(epoch,
                                                                                                             time.time() - start_time, train_avg_loss, train_avg_acc))
            
            start_time = time.time()
            print(' **Evaluating on validate set')
            val_avg_loss, val_avg_acc = self.evaluate(self.opt, self.model,
                                                      self.criterion, self.valLoader)
            print('| finish validating on epoch {:3d} | time(s) {:5.2f} | loss {:5.2f} | score {:5.2f}'.format(epoch,
                                                                                                               time.time() - start_time, val_avg_loss, val_avg_acc))
            
            if val_avg_acc > best_result:
                best_result = val_avg_acc
                best_flag = True
                print('*' * 10, 'BEST result {} at epoch {}'.format(best_result, epoch), '*' * 10)
            
            if epoch % self.opt.checkpoint_interval == 0 or epoch == self.opt.epochs or best_flag:
                print(' **Saving checkpoint {}'.format(epoch))
                if self.opt.ispretrain:
                    snapshot_prefix = os.path.join(self.opt.save_path, 'pretrain')
                else:
                    snapshot_prefix = os.path.join(self.opt.save_path, 'snapshot')
                snapshot_path = snapshot_prefix + '_epoch_{}.pt'.format(epoch)
                model_state = self.model.module.state_dict() if len(self.opt.gpus) > 1 else self.model.state_dict()
                optim_state = self.optimizer.state_dict()
                checkpoint = {
                              'modelstate':model_state,
                              'optimstate':optim_state,
                              'epoch':epoch,
                              'best_result':best_result,
                              }
                torch.save(checkpoint, snapshot_path, pickle_module=pickle)
                
                #delete old checkpoint
                for f in glob.glob(snapshot_prefix + '*'):
                    if f != snapshot_path:
                        os.remove(f)
                if best_flag:
                    if self.opt.ispretrain:
                        best_prefix = os.path.join(self.opt.save_path, 'BESTpretrain')
                    else:
                        best_prefix = os.path.join(self.opt.save_path, 'BESTsnapshot')
                    best_path = best_prefix + '_epoch_{}.pt'.format(epoch)
                    torch.save(checkpoint, best_path, pickle_module=pickle)
                    best_flag = False
                    for f in glob.glob(best_prefix + '*'):
                        if f != best_path:
                            os.remove(f)
                print('| finish saving checkpoint {}'.format(epoch))
                
            #visualize training and validating process
            train_loss_logger.log(epoch, train_avg_loss)
            val_loss_logger.log(epoch, val_avg_loss)
            acc_logger.log((epoch, epoch), (train_avg_acc, val_avg_acc))
            visdom_saver.save()
        
        print('*' * 65)
        print('Finish train and test on all epoch')

#------------------------------------------------------------------------------ 
#Testing on specific images
#------------------------------------------------------------------------------ 
    
    def test(self, opt, model, dataloader):
        """Segment on random image from dataset
        Support 2D images only
        """
        #colors base
        colors = []
        colors.append([0, 0, 0])
        colors.append([165,42,42])
        colors.append([0,255,0])
        colors.append([0,0,255])
        colors.append([255,0,255])
        colors.append([255,255,0])
        colors.append([0, 255, 255])
        colors.append([255, 255, 255])
        colors = torch.IntTensor(colors)
        
        model.eval()
        
        #run all validation image
        idx = 0
        for iteration, batch in enumerate(dataloader, 1):
            input = Variable(batch[0], volatile=True)
            target = Variable(batch[1], volatile=True)
            if len(opt.gpus) > 1:
                input = input.cuda(opt.gpus[0], True)
                target = target.cuda(opt.gpus[0], True)  
            elif opt.cuda:
                input = input.cuda()
                target = target.cuda()

            #predict output
            output = model(input)
            _, mask = output.cpu().data.max(dim=1)
            predicts = torch.zeros(*mask.size(), 3).int() #color image
            _, label = target.cpu().data.max(dim=1)
            groudtruths = torch.zeros(*label.size(), 3).int()
            for n, i, j in itertools.product(range(mask.size(0)), range(mask.size(1)), range(mask.size(2))):
                predicts[n, i, j, :] = colors[mask[n, i, j]]
                groudtruths[n, i, j, :] = colors[label[n, i, j]]

            #saving image
            os.makedirs('predictAdamNoTransform', exist_ok=True)
            file_prefix = os.path.join('predictAdamNoTransform', 'image')
            for n in range(mask.size(0)):
                input_path = file_prefix + '_input_{}.png'.format(idx+n)
                output_path = file_prefix + '_predict_{}.png'.format(idx+n)
                target_path = file_prefix + '_groundtruth_{}.png'.format(idx+n)
                input_img = T.ToPILImage()(input.cpu().data[n])
                input_img.save(input_path)
                misc.imsave(output_path, predicts[n].numpy())
                misc.imsave(target_path, groudtruths[n].numpy())
            idx += mask.size(0)
        model.train()
        return idx
    
    def validate(self):
        #Load trained model
        print(self.opt, '\n')
        print('Load checkpoint at {}'.format(self.opt.trained_model))
        checkpoint = torch.load(self.opt.trained_model, map_location=lambda storage, loc: storage, pickle_module=pickle)
        model_state = checkpoint['modelstate']
        self.model.load_state_dict(model_state)
        
        #DataParallel for multiple GPUs:
        if len(self.opt.gpus) > 1:
            #dim always is 0 because of input data always is in shape N*W
            self.model = nn.DataParallel(self.model, device_ids=self.opt.gpus, dim=0)
            self.model.cuda()
        elif self.opt.cuda:
            self.model.cuda()
        
        print('Start testing cuda {}'.format(self.opt.cuda))
        start_time = time.time()
        total_sample = self.test(self.opt, self.model, self.valLoader)
        print('| finish testing on {} samples in {} seconds'.format(total_sample, time.time() - start_time))

if __name__ == "__main__":
    main = Main(opt)
    if opt.trained_model:
        main.validate()
    else:
        main.execute()