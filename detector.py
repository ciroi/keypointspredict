from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn import init
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import datasets, transforms

import runpy
import numpy as np
import os
import cv2

from data import get_train_test_set
from predict import validPredict
from predict import videoPredict

torch.set_default_tensor_type(torch.FloatTensor)

class Residual(nn.Module):
    def __init__(self, in_channels, out_channels, use_1x1conv=False, stride=1):
        super(Residual, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        return F.relu(Y + X)

criterion_pts = nn.MSELoss()
criterion_cls = nn.CrossEntropyLoss()

def netLoss(pt, cls, targets):
    tcls = targets[:, -1].long()
    npt = tcls!=0
    
    ptLoss = criterion_pts(pt[npt, :], targets[npt, :-1])
    clsLoss = criterion_cls(cls, targets[:, -1].long())
    return 0.5 * ptLoss + 0.5 * clsLoss

class Residual(nn.Module):
    def __init__(self, in_channels, out_channels, use_1x1conv=False, stride=1):
        super(Residual, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        return F.relu(Y + X)

def netLoss(pt, cls, targets):
    tcls = targets[:, -1].long()
    npt = tcls!=0
    
    ptLoss = criterion_pts(pt[npt, :], targets[npt, :-1])
    clsLoss = criterion_cls(cls, targets[:, -1].long())
    
    return 0.5 * ptLoss + 0.5 * clsLoss

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # # Backbone:
        self.conv1_1 = nn.Sequential(
            nn.Conv2d(1,8,5,2,0),
            nn.BatchNorm2d(8),
            nn.PReLU(),
            nn.MaxPool2d(2,2)
        )
        self.conv2_1 = nn.Sequential(
            nn.Conv2d(8, 16, 3, 1, 0),
            nn.BatchNorm2d(16),
            nn.PReLU()
        )
        
        self.residulal_1 = Residual(16, 16)
        
        self.conv2_2 = nn.Sequential(
            nn.Conv2d(16, 16, 3, 1, 0),
            nn.BatchNorm2d(16),
            nn.PReLU(),
            nn.MaxPool2d(2,2, ceil_mode=True)
        )
        self.conv3_1 = nn.Sequential(
            nn.Conv2d(16, 24, 3, 1, 0),
            nn.BatchNorm2d(24),
            nn.PReLU()
        )

        self.conv3_2 = nn.Sequential(
            nn.Conv2d(24, 24, 3, 1, 0),
            nn.BatchNorm2d(24),
            nn.PReLU(),
            nn.MaxPool2d(2,2, ceil_mode=True)
        )

        self.conv4_1 = nn.Sequential(
            nn.Conv2d(24, 40, 3, 1, 1),
            nn.BatchNorm2d(40),
            nn.PReLU()
        )
        
        self.residulal_2 = Residual(40, 40)
        
        self.conv4_2 = nn.Sequential(
            nn.Conv2d(40, 80, 3, 1, 1),
            nn.BatchNorm2d(80),
            nn.PReLU(),
        )
        
        self.conv4_2_cls = nn.Sequential(
            nn.Conv2d(40, 40, 3, 1, 1),
            nn.BatchNorm2d(40),
            nn.PReLU(),
        )
        
        self.ip1_cls = nn.Sequential(
            nn.Linear(4 * 4 * 40, 512),
            nn.PReLU(),
            nn.Dropout(0.5)
        )
        
        self.ip2_cls = nn.Sequential(
            nn.Linear(512, 128),
            nn.PReLU(),
            nn.Dropout(0.5)
        )
        
        self.ip3_cls = nn.Sequential(
            nn.Linear(128, 2),
            nn.PReLU(),
            nn.Dropout(0.5),
            nn.Sigmoid() 
        )
        
        self.fc1 = nn.Sequential(
            nn.Linear(4 * 4 * 80, 512),
            nn.PReLU(),
            nn.Dropout(0.5)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(512, 128),
            nn.PReLU(),
            nn.Dropout(0.5)
        )
        self.fc3 = nn.Linear(128, 42)

    def forward(self, x, targets=None):
        # print(x.shape)
        x = self.conv1_1(x)
        # print('x after block1 and pool shape should be 32x8x27x27: ', x.shape)
        x = self.conv2_1(x)
        x = self.residulal_1(x)
        
        # print('b2: after conv2_1 and prelu shape should be 32x16x25x25: ', x.shape)
        x = self.conv2_2(x)
        # print('x after block2 and pool shape should be 32x16x12x12: ', x.shape)
        x = self.conv3_1(x)
        # print('b3: after conv3_1 and pool shape should be 32x24x10x10: ', x.shape)
        x = self.conv3_2(x)
        # print('x after block3 and pool shape should be 32x24x4x4: ', x.shape)
        x = self.conv4_1(x)
        x = self.residulal_2(x)
        
        y = self.conv4_2_cls(x)
        y = y.view(y.size(0),-1)
        y = self.ip1_cls(y)
        y = self.ip2_cls(y)
        y = self.ip3_cls(y)
        
        # print('x after conv4_1 and pool shape should be 32x40x4x4: ', x.shape)
        x = self.conv4_2(x)
        # print('pts: ip3 after conv4_2 and pool shape should be 32x80x4x4: ', x.shape)
        x = x.view(x.size(0),-1)
        # print(x.shape)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)

        if targets is None:
            return x, y
        loss = netLoss(x, y, targets)
        
        return loss

def train(args, train_loader, valid_loader, model, criterion, optimizer, device):
    # save model
    if args.save_model:
        if not os.path.exists(args.save_directory):
            os.makedirs(args.save_directory)

    epoch = args.epochs
    pts_criterion = criterion

    train_losses = []
    valid_losses = []

    for epoch_id in range(epoch):
        adjust_learning_rate(optimizer, epoch_id, 0.01)
        train_loss = 0.0
        valid_loss = 0.0
        model.train()
        for batch_idx, batch in enumerate(train_loader):
            img = batch['image']
            landmark = batch['landmarks']
			
            input_img = img.to(device)
            target_pts = landmark.to(device)

            optimizer.zero_grad()

            loss = model(input_img, target_pts)
            
            loss.backward()
            optimizer.step()
			
			# show log info
            if batch_idx % args.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\t pts_loss: {:.6f}'.format(
						epoch_id,
						batch_idx * len(img),
						len(train_loader.dataset),
						100. * batch_idx / len(train_loader),
						loss.item()
					)
				)

        valid_mean_pts_loss = 0.0

        model.eval() 
        with torch.no_grad():
            valid_batch_cnt = 0

            for valid_batch_idx, batch in enumerate(valid_loader):
                valid_batch_cnt += 1
                valid_img = batch['image']
                landmark = batch['landmarks']

                input_img = valid_img.to(device)
                target_pts = landmark.to(device)

                loss = model(input_img, target_pts)
				
                valid_mean_pts_loss += loss.item()
				
            valid_mean_pts_loss /= valid_batch_cnt * 1.0
            print('Valid: pts_loss: {:.6f}'.format(
					valid_mean_pts_loss
				)
			)
        print('====================================================')
        # save model
        if args.save_model:
            saved_model_name = os.path.join(args.save_directory, 'detector_epoch' + '_' + str(epoch_id) + '.pt')
            torch.save(model.state_dict(), saved_model_name)
    return loss, 0.5

def main_test():
    parser = argparse.ArgumentParser(description='Detector')
    parser.add_argument('--batch-size', type=int, default=50, metavar='N',				
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=50, metavar='N',		
                        help='input batch size for testing (default: 64)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001)')
    parser.add_argument('--momentum', type=float, default=0.05, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=True,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=20, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=True,
                        help='save the current Model')
    parser.add_argument('--save-directory', type=str, default='trained_models',
                        help='learnt models are saving here')
    parser.add_argument('--phase', type=str, default='Train',   
                        help='training, test or predict')
    parser.add_argument('--pre_mode', type=str, default='',  
                        help='training, predicting or finetuning')
    args = parser.parse_args()
	###################################################################################
    torch.manual_seed(args.seed)
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")    # cuda:0
	
    print('===> Loading Datasets')
    train_set, test_set = get_train_test_set()
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(test_set, batch_size=args.test_batch_size)

    print('===> Building Model')
    model = Net().to(device)

    ####################################################################
    criterion_pts = nn.MSELoss()
    # optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.0001)
	####################################################################
    if args.phase == 'Train' or args.phase == 'train':
        if args.pre_mode.endswith(".pt"):
            model.load_state_dict(torch.load(args.pre_mode))
        print('===> Start Training')
        train_losses, valid_losses = \
			train(args, train_loader, valid_loader, model, criterion_pts, optimizer, device)
        print('====================================================')
    elif args.phase == 'Test' or args.phase == 'test':
        print('===> Test')
        validPredict(args, "detector_epoch.pt", Net(), valid_loader)
    elif args.phase == 'Predict' or args.phase == 'predict':
        print('===> Predict')
        videoPredict(args, "detector_epoch.pt", Net())

if __name__ == '__main__':
    main_test()










