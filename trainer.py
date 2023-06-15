import glob
import logging
import sys
import time
from sklearn.metrics import accuracy_score

import torch
from torch import optim
import torch.nn as nn
import numpy as np
from torch.backends import cudnn

from Args import Args
from utils import *
import utils


class Trainer:
    """
    ====================================================================================================================
    INIT ===============================================================================================================
    ====================================================================================================================
    The Trainer class will receive the following inputs
        * model: The model returned by your NAS class
        * train_loader: The train loader created by your DataProcessor
        * valid_loader: The valid loader created by your DataProcessor
        * metadata: A dictionary with information about this dataset, with the following keys:
            'num_classes' : The number of output classes in the classification problem
            'codename' : A unique string that represents this dataset
            'input_shape': A tuple describing [n_total_datapoints, channel, height, width] of the input data
            'time_remaining': The amount of compute time left for your submission
            plus anything else you added in the DataProcessor or NAS classes
    """

    def __init__(self, model, device, train_dataloader, valid_dataloader, metadata):
        self.model = model
        self.device = device
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.metadata = metadata
        self.args = Args()
        # define  training parameters
        self.epochs = 2
        self.optimizer = optim.SGD(model.parameters(), lr=.01, momentum=.9, weight_decay=3e-4)
        self.criterion = nn.CrossEntropyLoss().to(self.device)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.epochs)

    """
    ====================================================================================================================
    TRAIN ==============================================================================================================
    ====================================================================================================================
    The train function will define how your model is trained on the train_dataloader.
    Output: Your *fully trained* model

    See the example submission for how this should look
    """
    def train(self):

        args = self.args
        args.save = 'eval-{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))
        utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))

        log_format = '%(asctime)s %(message)s'
        logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                            format=log_format, datefmt='%m/%d %I:%M:%S %p')
        fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
        fh.setFormatter(logging.Formatter(log_format))
        logging.getLogger().addHandler(fh)
        np.random.seed(args.seed)
        torch.cuda.set_device(args.gpu)
        cudnn.benchmark = True
        torch.manual_seed(args.seed)
        cudnn.enabled = True
        torch.cuda.manual_seed(args.seed)
        logging.info('gpu device = %d' % args.gpu)
        logging.info("args = %s", args)

        #genotype = eval(str(xx))
        #model = NetworkCIFAR(args.init_channels, CIFAR_CLASSES, args.layers, genotype=xx, auxiliary=args.auxiliary)
        self.model = self.model.to(self.device)

        logging.info("param size = %fMB", utils.count_parameters_in_MB(self.model))


        # train_transform, valid_transform = utils._data_transforms_cifar10(args)
        train_queue = self.train_dataloader
        valid_queue = self.valid_dataloader
        # split = int(np.floor(args.train_portion * num_train))
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, float(args.epochs))

        for epoch in range(self.epochs):
            scheduler.step()
            logging.info('epoch %d lr %e', epoch, scheduler.get_lr()[0])
            self.model.drop_path_prob = args.drop_path_prob * epoch / args.epochs

            train_acc, train_obj = self._train(train_queue, self.model, self.criterion, self.optimizer)
            print('train_acc %f' % train_acc)
            logging.info('train_acc %f', train_acc)
            with torch.no_grad():
                valid_acc, valid_obj = self.infer(valid_queue, self.model, self.criterion)
                print('valid_acc %f', valid_acc)
                logging.info('valid_acc %f', valid_acc)

            utils.save(self.model, os.path.join(args.save, 'weights.pt'))
        return self.model

    def _train(self,train_queue, model, criterion, optimizer):
        objs = AvgrageMeter()
        top1 = AvgrageMeter()
        top5 = AvgrageMeter()
        model.train()

        for step, (input, target) in enumerate(train_queue):
            input = input.to(self.device)
            target = target.to(self.device)

            optimizer.zero_grad()
            logits = model(input)
            loss = criterion(logits, target)
            loss.backward()
            nn.utils.clip_grad_norm(model.parameters(), self.args.grad_clip)
            optimizer.step()

            prec1 = utils.accuracy(logits, target, topk=(1,))
            n = input.size(0)
            objs.update(loss.data, n)
            top1.update(prec1[0], n)
            # top5.update(prec5.data, n)

            if step % self.args.report_freq == 0:
                print('train ', step, objs.avg, top1.avg, top5.avg)
                # logging.info('train ', step, objs.avg, top1.avg, top5.avg)

        return top1.avg, objs.avg


    def infer(self,valid_queue, model, criterion):
        objs = AvgrageMeter()
        top1 = AvgrageMeter()
        top5 = AvgrageMeter()
        model.eval()

        for step, (input, target) in enumerate(valid_queue):
            input = input.to(self.device)
            target = target.to(self.device)

            logits = model(input)
            loss = criterion(logits, target)

            prec1 = utils.accuracy(logits, target, topk=(1,))
            n = input.size(0)
            objs.update(loss.data, n)
            top1.update(prec1[0], n)
            # top5.update(prec5.data, n)

            if step % self.args.report_freq == 0:
                print('valid ', step, objs.avg, top1.avg, top5.avg)
                # logging.info('valid ', step, objs.avg, top1.avg, top5.avg)

        return top1.avg, objs.avg

    """
    ====================================================================================================================
    PREDICT ============================================================================================================
    ====================================================================================================================
    The prediction function will define how the test dataloader will be passed through your model. It will receive:
        * test_dataloader created by your DataProcessor

    And expects as output:
        A list/array of predicted class labels of length=n_test_datapoints, i.e, something like [0, 0, 1, 5, ..., 9] 

    See the example submission for how this should look.
    """

    def predict(self, test_loader):
        self.model.eval()
        predictions = []
        for data in test_loader:
            data = data.to(self.device)
            output = self.model.forward(data)
            predictions += torch.argmax(output, 1).detach().cpu().tolist()
        return predictions
