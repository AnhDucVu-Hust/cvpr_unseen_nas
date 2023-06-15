import logging
import os

import numpy as np
import torch
from torch.autograd import Variable
from torch.backends import cudnn
from torch import nn
from utils import AvgrageMeter
import utils
from architect import Architect
from model_search import *
from Args import Args

np.random.seed(42)
torch.cuda.set_device("cuda:0" if torch.cuda.is_available() else "cpu")
cudnn.benchmark = True
torch.manual_seed(42)
cudnn.enabled = True
torch.cuda.manual_seed(42)
args = Args()


class NAS:
    """
    ====================================================================================================================
    INIT ===============================================================================================================
    ====================================================================================================================
    The NAS class will receive the following inputs
        * train_loader: The train loader created by your DataProcessor
        * valid_loader: The valid loader created by your DataProcessor
        * metadata: A dictionary with information about this dataset, with the following keys:
            'num_classes' : The number of output classes in the classification problem
            'codename' : A unique string that represents this dataset
            'input_shape': A tuple describing [n_total_datapoints, channel, height, width] of the input data
            'time_remaining': The amount of compute time left for your submission
            plus anything else you added in the DataProcessor

        You can modify or add anything into the metadata that you wish,
        if you want to pass messages between your classes,
    """

    def __init__(self, train_loader, valid_loader, metadata):
        # fill this in
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.num_classes = metadata['num_classes']
        self.input_shape = metadata['input_shape']

    """
    ====================================================================================================================
    SEARCH =============================================================================================================
    ====================================================================================================================
    The search function is called with no arguments, and expects a PyTorch model as output. Use this to
    perform your architecture search. 
    """

    def search(self):
        np.random.seed(args.seed)
        torch.cuda.set_device(args.gpu)
        cudnn.benchmark = True
        torch.manual_seed(args.seed)
        cudnn.enabled = True
        torch.cuda.manual_seed(args.seed)
        print('gpu device = %d' % args.gpu)
        print("args = %s", args)
        logging.info('gpu device = %d' % args.gpu)
        logging.info("args = %s", args)

        criterion = nn.CrossEntropyLoss()
        criterion = criterion.cuda()
        model = Network(args.init_channels, self.num_classes, args.layers, criterion)
        model = model.cuda()
        print("param size = %fMB", utils.count_parameters_in_MB(model))
        logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

        # print("Day la model " + str(model.alphas_normal))
        optimizer = torch.optim.SGD(
            model.parameters(),
            args.learning_rate,
            momentum=args.momentum,
            weight_decay=args.weight_decay)

        # train_transform, valid_transform = utils._data_transforms_cifar10(args)
        train_queue = self.train_loader
        valid_queue = self.valid_loader
        #valid_queue = len(train_data)
        #indices = list(range(num_train))
        # split = int(np.floor(args.train_portion * num_train))

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, float(args.epochs), eta_min=args.learning_rate_min)

        architect = Architect(model, args)

        for epoch in range(args.epochs):
            scheduler.step()
            lr = scheduler.get_lr()[0]
            print('epoch %d lr %e' % (epoch, lr))
            logging.info('epoch %d lr %e', epoch, lr)

            genotype = model.genotype()
            print(genotype)
            with open('/kaggle/working/genotype.txt', 'w') as f:
                f.write(str(genotype))
            print(F.softmax(model.alphas_normal, dim=-1))
            print(F.softmax(model.alphas_reduce, dim=-1))

            # training
            train_acc, train_obj = self.train(train_queue, valid_queue, model, architect, criterion, optimizer, lr)
            logging.info('train_acc %f', train_acc)

            # validation
            valid_acc, valid_obj = self.infer(valid_queue, model, criterion)
            logging.info('valid_acc %f', valid_acc)

            utils.save(model, os.path.join(args.save, 'weights.pt'))

    def train(self,train_queue, valid_queue, model, architect, criterion, optimizer, lr):
        objs = AvgrageMeter()
        top1 = AvgrageMeter()
        top5 = AvgrageMeter()

        for step, (input, target) in enumerate(train_queue):
            model.train()
            n = input.size(0)

            input = Variable(input, requires_grad=False).cuda()
            target = Variable(target, requires_grad=False).cuda()

            # get a random minibatch from the search queue with replacement
            input_search, target_search = next(iter(valid_queue))
            input_search = Variable(input_search, requires_grad=False).cuda()
            target_search = Variable(target_search, requires_grad=False).cuda()

            architect.step(input, target, input_search, target_search, lr, optimizer, unrolled=args.unrolled)

            optimizer.zero_grad()
            logits = model(input)
            loss = criterion(logits, target)

            loss.backward()
            nn.utils.clip_grad_norm(model.parameters(), args.grad_clip)
            optimizer.step()

            prec1 = utils.accuracy(logits, target, topk=(1,))[0]
            objs.update(loss.data, n)
            top1.update(prec1.data, n)
            # top5.update(prec5.data, n)

            if step % args.report_freq == 0:
                print('train %03d %e %f %f' % (step, objs.avg, top1.avg, top5.avg))
        #         logging.info('train %03d %e %f %f' % step, objs.avg, top1.avg, top5.avg)

        return top1.avg, objs.avg

    def infer(self,valid_queue, model, criterion):
        objs = AvgrageMeter()
        top1 = AvgrageMeter()
        top5 = AvgrageMeter()
        model.eval()
        with torch.no_grad():
            for step, (input, target) in enumerate(valid_queue):
                input = Variable(input, volatile=True).cuda()
                target = Variable(target, volatile=True).cuda()

                logits = model(input)
                loss = criterion(logits, target)

                prec1 = utils.accuracy(logits, target, topk=(1,))[0]
                n = input.size(0)
                objs.update(loss.data, n)
                top1.update(prec1.data, n)
                # top5.update(prec5.data, n)

                if step % args.report_freq == 0:
                    print('valid %03d %e %f %f' % (step, objs.avg, top1.avg, top5.avg))
        #           logging.info('valid %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

        return top1.avg, objs.avg