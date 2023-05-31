import logging
import os

import numpy as np
import torch
from torch.autograd import Variable
from torch.backends import cudnn
from torch import nn
from tqdm import tqdm

import utils
from architect import Architect
from model_search import *
from Args import Args
np.random.seed(42)
torch.cuda.set_device("cuda:0" if torch.cuda.is_available() else "cpu")
cudnn.benchmark = True
torch.manual_seed(42)
cudnn.enabled=True
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
        self.train_loader=train_loader
        self.valid_loader=valid_loader
        self.num_classes=metadata['num_classes']
        self.input_shape=metadata['input_shape']


    """
    ====================================================================================================================
    SEARCH =============================================================================================================
    ====================================================================================================================
    The search function is called with no arguments, and expects a PyTorch model as output. Use this to
    perform your architecture search. 
    """
    def search(self):
        criterion = nn.CrossEntropyLoss()
        criterion = criterion.cuda()
        model = Network(C=16,num_channel=self.input_shape[1],layers=8,criterion=criterion,num_classes = self.num_classes)
        model = model.cuda()
        arch_params = list (map(id,model.arch_parameters()))
        weight_params = filter(lambda p: id(p) not in arch_params, model.parameters())
        optimizer = torch.optim.AdamW(weight_params, lr=1e-4, weight_decay=3e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,50,eta_min=0)
        architect = Architect(model,args)
        ops = []
        for cell_type in ['normal', 'reduce']:
            for edge in range(model.num_edges):
                ops.append(['{}_{}_{}'.format(cell_type, edge, i) for i in
                            range(0, model.num_ops)])
        ops = np.concatenate(ops)
        pretrain_epochs = 25
        train_epochs = (25,50)
        epoch=0
        accum_shaps=[1e-3*torch.randn(model.num_edges,model.num_ops).cuda(),1e-3 * torch.randn(model.num_edges, model.num_ops).cuda()]
        for i, current_epochs in enumerate(train_epochs):
            for e in range(current_epochs):
                scheduler.step()
                lr = scheduler.get_lr()[0]
                genotype=model.genotype()
                genotype_full=model.genotype_full()
                if i == len(train_epochs) -1:
                    shap_normal, shap_reduce = self.shap_estimation(self.valid_loader, model, criterion, ops, num_samples=args.samples)
                    accum_shaps = self.change_alpha(model, [shap_normal,shap_reduce], accum_shaps, momentum = args.shapley_momentum, step_size = args.step_size)
                train_acc, train_obj = self.train(self.train_loader,model,criterion,optimizer)
                if epoch == args.epochs - 1 or epoch % 2 == 0:
                    valid_acc, valid_obj = self.infer(self.valid_loader, model, criterion)
                    print('valid_acc %f', valid_acc)
                if not args.resume and epoch == pretrain_epochs - 1:
                    utils.save(model, os.path.join(args.save, 'weights_pretrain.pt'))
                utils.save(model, os.path.join(args.save, 'weights.pt'))
                epoch += 1
                model.show_arch_parameters()
            genotype = model.genotype()
            print('genotype = %s', genotype)
        return model

    def remove_players(self,normal_weights, reduce_weights, op):

        selected_cell = str(op.split('_')[0])
        selected_eid = int(op.split('_')[1])
        opid = int(op.split('_')[-1])
        proj_mask = torch.ones_like(normal_weights[selected_eid])
        proj_mask[opid] = 0
        if selected_cell in ['normal']:
            normal_weights[selected_eid] = normal_weights[selected_eid] * proj_mask
        else:
            reduce_weights[selected_eid] = reduce_weights[selected_eid] * proj_mask

    def shap_estimation(self,valid_queue, model, criterion, players, num_samples, threshold=0.5):
        """
        Implementation of Monte-Carlo sampling of Shapley value for operation importance evaluation
        """

        permutations = None
        n = len(players)
        sv_acc = np.zeros((n, num_samples))

        with torch.no_grad():

            if permutations is None:
                # Keep the same permutations for all batches
                permutations = [np.random.permutation(n) for _ in range(num_samples)]

            for j in range(num_samples):
                x, y = next(iter(valid_queue))
                x, y = x.cuda(), y.cuda(non_blocking=True)
                logits = model(x, weights_dict=None)
                ori_prec1, = utils.accuracy(logits, y, topk=(1,))

                normal_weights = model.get_projected_weights('normal')
                reduce_weights = model.get_projected_weights('reduce')

                acc = ori_prec1.data.item()
                print('MC sampling %d times' % (j + 1))

                for idx, i in enumerate(permutations[j]):

                    self.remove_players(normal_weights, reduce_weights, players[i])

                    logits = model(x, weights_dict={'normal': normal_weights, 'reduce': reduce_weights})
                    prec1, = utils.accuracy(logits, y, topk=(1,))
                    new_acc = prec1.item()
                    delta_acc = acc - new_acc
                    sv_acc[i][j] = delta_acc
                    acc = new_acc
                    # print(players[i], delta_acc)

                    if acc < threshold * ori_prec1:
                        break

            result = np.mean(sv_acc, axis=-1) - np.std(sv_acc, axis=-1)
            shap_acc = np.reshape(result, (2, model.num_edges, model.num_ops))
            shap_normal, shap_reduce = shap_acc[0], shap_acc[1]

            return shap_normal, shap_reduce

    def change_alpha(self,model, shap_values, accu_shap_values, momentum=0.8, step_size=0.1):
        assert len(shap_values) == len(model.arch_parameters())

        shap = [torch.from_numpy(shap_values[i]).cuda() for i in range(len(model.arch_parameters()))]

        for i, params in enumerate(shap):
            mean = params.data.mean()
            std = params.data.std()
            params.data.add_(-mean).div_(std)

        updated_shap = [
            accu_shap_values[i] * momentum \
            + shap[i] * (1. - momentum)
            for i in range(len(model.arch_parameters()))]

        for i, p in enumerate(model.arch_parameters()):
            p.data.add_((step_size * updated_shap[i]).to(p.device))

        return updated_shap

    def train(self,train_queue, model, criterion, optimizer):
        objs = utils.AvgrageMeter()
        top1 = utils.AvgrageMeter()
        top5 = utils.AvgrageMeter()

        for step, (input, target) in tqdm(enumerate(train_queue), desc="training", total=len(train_queue)):
            model.train()
            n = input.size(0)
            input = Variable(input, requires_grad=False).cuda()
            target = Variable(target, requires_grad=False).cuda()
            optimizer.zero_grad()
            logits = model(input)
            loss = criterion(logits, target)

            loss.backward()
            nn.utils.clip_grad_norm(model.parameters(), args.grad_clip)
            optimizer.step()

            prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
            objs.update(loss.item(), n)
            top1.update(prec1.item(), n)
            top5.update(prec5.item(), n)

            if step % args.report_freq == 0:
                logging.info('train %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

        return top1.avg, objs.avg
    def infer(self,valid_queue, model, criterion):
        objs = utils.AvgrageMeter()
        top1 = utils.AvgrageMeter()
        top5 = utils.AvgrageMeter()
        model.eval()
        
        with torch.no_grad():
            for step, (input, target) in enumerate(valid_queue):

                input = Variable(input).cuda()
                target = Variable(target).cuda()
                logits = model(input)
                loss = criterion(logits, target)

                prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
                n = input.size(0)
                objs.update(loss.item(), n)
                top1.update(prec1.item(), n)
                top5.update(prec5.item(), n)

                if step % args.report_freq == 0:
                  logging.info('valid %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

          return top1.avg, objs.avg
