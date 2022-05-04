from vit import ViT,ClassificationHead,Reduce_channel
import argparse
import numpy as np
import random
from torch.autograd import Variable
import torch.nn as nn
# import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.autograd as autograd
import torch
import torch.distributed as dist
# from torch.utils.tensorboard import SummaryWriter
# import cv2 as cv
import torchvision
import os
import scipy.io as scio
from metrics import ArcMarginProduct,AddMarginProduct,FocalLoss,SphereProduct,ArcFace


parser = argparse.ArgumentParser()
parser.add_argument("--feature_epochs", type=int, default=50, help="number of epochs of training feature")
parser.add_argument("--classifier_epochs", type=int, default=80, help="number of epochs of training classifier")
parser.add_argument("--feature_lr", type=float, default=1e-4, help="learning rate")
parser.add_argument("--classifier_lr", type=float, default=3e-5, help="learning rate")
parser.add_argument("--b1", type=float, default=0.9, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
opt = parser.parse_args()

def train(sub_index, A_data_train, A_label_train, B_data_train, B_label_train, B_data_test, B_label_test,batch_size,gpus,file_name):

    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, gpus))

    features = ViT()
    metric_fc= ArcMarginProduct(15, 4, s=30, m=0.5, easy_margin=True)
    # metric_fc = ArcFace(15, 4)
    # metric_fc=AddMarginProduct(15,4)
    # metric_fc=SphereProduct(15,4)
    classifier_head=ClassificationHead(15,4)
    
    features = features.cuda()
    features = nn.DataParallel(features, device_ids=[i for i in range(len(gpus))])
    features=features.cuda()
    print('features')
    print(features)
    
    metric_fc = metric_fc.cuda()
    metric_fc = nn.DataParallel(metric_fc, device_ids=[i for i in range(len(gpus))])
    metric_fc=metric_fc.cuda()
    print('metric_fc')
    print(metric_fc)
    
    classifier_head = classifier_head.cuda()
    classifier_head = nn.DataParallel(classifier_head, device_ids=[i for i in range(len(gpus))])
    classifier_head=classifier_head.cuda()
    print('classifier_head')
    print(classifier_head)

    # reduce_ch=Reduce_channel(4,4)
    # reduce_ch=reduce_ch.cuda()
    # reduce_ch=nn.DataParallel(reduce_ch,device_ids=[i for i in range(len(gpus))])
    # reduce_ch=reduce_ch.cuda()

    # set the seed for reproducibility
    seed_n = np.random.randint(500)
    print(seed_n)
    random.seed(seed_n)
    np.random.seed(seed_n)
    torch.manual_seed(seed_n)
    torch.cuda.manual_seed(seed_n)

    # A_data_train = A_data_train.transpose([0, 2, 1])
    # B_data_test = B_data_test.transpose([0, 2, 1])
    # A_data_train = np.expand_dims(A_data_train, axis=1)
    # B_data_test = np.expand_dims(B_data_test, axis=1)

    A_data_train = torch.from_numpy(A_data_train)
    B_data_train=torch.from_numpy(B_data_train)
    B_data_test = torch.from_numpy(B_data_test)
    A_label_train = torch.from_numpy(A_label_train)
    B_label_train=torch.from_numpy(B_label_train)
    B_label_test = torch.from_numpy(B_label_test)
    

    A_dataset = torch.utils.data.TensorDataset(A_data_train, A_label_train)
    A_dataloader = torch.utils.data.DataLoader(dataset=A_dataset, batch_size=batch_size, shuffle=True)
    
    B_dataset = torch.utils.data.TensorDataset(B_data_train, B_label_train)
    B_dataloader = torch.utils.data.DataLoader(dataset=B_dataset, batch_size=batch_size, shuffle=True)


    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    B_data_test = B_data_test.cuda()
    B_label_test = B_label_test.cuda()
    B_data_test = Variable(B_data_test.type(Tensor))

    # optimizer = torch.optim.SGD(classifier.parameters(), lr=opt.lr,weight_decay=0.01,momentum=0.9)
    optimizer = torch.optim.Adam([{'params': features.parameters()}, {'params': metric_fc.parameters()}], lr=opt.feature_lr, weight_decay=1e-3)
    classifier_optimizer = torch.optim.Adam(classifier_head.parameters(), lr=opt.classifier_lr, weight_decay=0.01)
    # loss_func = nn.CrossEntropyLoss()
    loss_func=FocalLoss(gamma=2)

    test_acc = []
    pre=[]
    
    
    #train feature
    for epoch in range(opt.feature_epochs):
        for step, data in enumerate(A_dataloader):
            datatrain, labeltrain = data
            datatrain = datatrain.cuda()
            labeltrain = labeltrain.cuda()

            # datatrain = Variable(datatrain.type(Tensor))
            # labeltrain = Variable(labeltrain.type(Tensor))
            datatrain = datatrain.type(Tensor)
            labeltrain = labeltrain.type(Tensor)

            output = features(datatrain)
            output=metric_fc(output,labeltrain)
            loss = loss_func(output, labeltrain.long())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % 1 == 0:
                test_output = features(B_data_test)
                test_output=metric_fc(test_output,B_label_test)

                pred = torch.max(test_output, 1)[1].data.squeeze()
                accuracy = float((pred == B_label_test.type(torch.long)).cpu().numpy().astype(int).sum()) / float(B_label_test.size(0))
                print('train features --- Epoch: ', epoch, '| train loss: %.4f' % loss.data, '| test accuracy: %.6f' % accuracy)
    
    for name,param in features.named_parameters():
        param.requires_grad=False

    for name,param in metric_fc.named_parameters():
        param.requires_grad=False


    #fine tune
    #train classifier
    for epoch in range(opt.classifier_epochs):
        for step, data in enumerate(B_dataloader):
            datatrain, labeltrain = data
            datatrain = datatrain.cuda()
            labeltrain = labeltrain.cuda()

            # datatrain = Variable(datatrain.type(Tensor))
            # labeltrain = Variable(labeltrain.type(Tensor))
            datatrain = datatrain.type(Tensor)
            labeltrain = labeltrain.type(Tensor)

            output = features(datatrain)
            output=classifier_head(output)
            loss = loss_func(output, labeltrain.long())
            classifier_optimizer.zero_grad()
            loss.backward()
            classifier_optimizer.step()

            if step % 1 == 0:
                test_output = features(B_data_test)
                test_output=classifier_head(test_output)

                pred = torch.max(test_output, 1)[1].data.squeeze()
                pre.append(pred)
                accuracy = float((pred == B_label_test.type(torch.long)).cpu().numpy().astype(int).sum()) / float(B_label_test.size(0))
                print('train classifier --- Epoch: ', epoch, '| train loss: %.4f' % loss.data, '| test accuracy: %.6f' % accuracy)
                test_acc.append(accuracy)

    test_acc.append(np.mean(test_acc))
    test_acc.append(np.max(test_acc))
    max_index=test_acc.index(np.max(test_acc))
    final_pre=pre[max_index].cpu().detach().numpy()
    test_acc.append(seed_n)
    save_acc = np.array(test_acc)

    np.savetxt('./result/'+file_name+'_b_' + str(
        sub_index) + '.txt', save_acc, '%.10f')
    scio.savemat('./pre_and_label/'+file_name+'_b_' +str(
        sub_index) + '.mat',{'pre':final_pre,'label':B_label_test.cpu().detach().numpy()})