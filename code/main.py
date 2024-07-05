import numpy as np
from tqdm import tqdm
import sys
import os
import argparse
import warnings
import random
import time
import math
import builtins
import shutil
import pandas as pd
import scipy.io as io

# import wandb

from contrastive_dataset import fmri_dataset_connectome
from model import Gen_GNN, Gen_GAT, Gen_SAGE, Gen_GIN, TopK
from utils import cross_validation_with_val_set

import torch
import torch.nn as nn
import torch.nn.functional as F
import torcheval.metrics.functional as MF



parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--data', metavar='DIR',
                    help='path to dataset')

parser.add_argument('--run_name', default='contrastive_learning_mri', type=str,
                    help='run name of the model')
                    
parser.add_argument('--project_name', default='cam_proj', type=str,
                    help='project name for wandb to monitor the model')

parser.add_argument('-j', '--workers', default=32, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.03, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--schedule', default=[120, 160], nargs='*', type=int,
                    help='learning rate schedule (when to drop lr by 10x)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum of SGD solver')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=100, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')

parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')

parser.add_argument('--low-dim', default=128, type=int,
                    help='feature dimension (default: 128)')

parser.add_argument('--moco-m', default=0.999, type=float,
                    help='moco momentum of updating key encoder (default: 0.999)')
parser.add_argument('--temperature', default=0.2, type=float,
                    help='softmax temperature')

# parser.add_argument('--mlp', action='store_true',
#                     help='use mlp head')
# parser.add_argument('--aug-plus', action='store_true',
#                     help='use moco-v2/SimCLR data augmentation')
parser.add_argument('--cos', action='store_true',
                    help='use cosine lr schedule')
# parser.add_argument('--warmup-epoch', default=20, type=int,
#                     help='number of warm-up epochs to only train with InfoNCE loss')
parser.add_argument('--exp-dir', default='/content/drive/MyDrive/Cam_proj/BrainAge_data', type=str,
                    help='experiment directory')
parser.add_argument('--n_layers', default=2, type=int, help='num of transformer module layers')
parser.add_argument('--norm', default='post', type=str, help='where to put batch normalization')

parser.add_argument('--brain_atlas', default=400, type=int, help='num of brain regions from brain atlas')

parser.add_argument('--level1', default=100, type=int, help='num of level1 layer nodes max')
parser.add_argument('--level2', default=50, type=int, help='num of level2 layer nodes max')
parser.add_argument('--level3', default=25, type=int, help='num of level3 layer nodes max')
parser.add_argument('--level4', default=12, type=int, help='num of level4 layer nodes max')

parser.add_argument('--num-classes', default=4, type=int, help='num of level4 layer nodes max')
parser.add_argument('--num-sample', default=100, type=int, help='num of adjacency matrix sampling')
parser.add_argument('--tau', default=0.01, type=float, help='temperature of differentiable sampling')
parser.add_argument('--threshold', default=0.1, type=float, help='ideal density of m')
parser.add_argument('--dcrp', default=1, type=int, help='whether to use dcrp')
parser.add_argument('--multi', default=0, type=int, help='multi-layer application')

parser.add_argument('--dataset', default='1000C', type=str,
                    help='dataset_name')

parser.add_argument('--module', default='GCN', type=str,
                    help='gnn module type')    
parser.add_argument('--dcrp_module', default='GAT', type=str,
                    help='dcrp module type')   

parser.add_argument("--folds", type=int, default=10, help="Default is 10.")           

def main():
    args = parser.parse_args()

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    # args.distributed = args.world_size > 1 or args.multiprocessing_distributed
    
    if not os.path.exists(args.exp_dir):
        os.mkdir(args.exp_dir)
    
    ngpus_per_node = torch.cuda.device_count()

    # wandb.init(project=args.project_name, name = args.run_name, config = args)

    if args.dataset == '1000C':
        args.brain_atlas = 177   
        fmri = io.loadmat(os.path.join(args.data, '1000_connectome_fmri.mat'))['fmri']['matrix'][0]
        age = pd.read_csv(os.path.join(args.data, '1000_connectome_fmri.csv'))

    elif args.dataset == 'NKI':
        args.brain_atlas = 188
        fmri = io.loadmat(os.path.join(args.data, 'normal_fmri_matrix.mat'))['fmri']['matrix'][0]
        age = pd.read_csv(os.path.join(args.data, 'NKI_age.csv'))

    elif args.dataset == 'ADHD':
        args.brain_atlas = 190
        fmri_initial = io.loadmat(os.path.join(args.data, 'adhd_fmri.mat'))['matrix']
        age_initial = pd.read_csv(os.path.join(args.data, 'adhd_label.csv'))
        age = age_initial[age_initial['label'] == 'Typically Developing']
        age_health_index = age_initial[age_initial['label'] == 'Typically Developing'].index.tolist()
        fmri = fmri_initial[age_health_index]
        age = age.reset_index(drop=True)
        # fmri = fmri.reset_index(drop=True)

    age_array = age['age'].index.tolist()
    # print(age_array)

    age_selection_index = age_array.copy()
    for selection_index in age_selection_index:
        if fmri[selection_index].shape[0] != args.brain_atlas:
            age_array.remove(age_selection_index[selection_index])
    
    fmri = fmri[age_array]
    age_selection = age.loc[age_array]
    age_selection.reset_index(drop = True, inplace=True)
    age_array = age_selection['age'].index.tolist()

    dataset = fmri_dataset_connectome(args, age_array, fmri, age_selection)

    # Simply call main_worker function
    # main_worker(args.gpu, args)
    if args.module == 'GCN':
        model = Gen_GNN(args.brain_atlas, args.num_classes, args=args)
    elif args.module == 'GAT':
        model = Gen_GAT(args.brain_atlas, args.num_classes, args=args)
    elif args.module == 'SAGE':
        model = Gen_SAGE(args.brain_atlas, args.num_classes, args=args)
    elif args.module == 'GIN':
        model = Gen_GIN(args.brain_atlas, args.num_classes, args=args)
    elif args.module == 'TopK':
        model = TopK(args.brain_atlas, args.num_classes, args=args)
    cross_validation_with_val_set(dataset, model, args.folds, args.epochs, args.batch_size,
                                  args.lr, args,
                                  args.weight_decay, logger=None)


def main_worker(gpu, args):
    args.gpu = gpu
    
    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.module == 'GCN':
        model = Gen_GNN(args.brain_atlas, args.num_classes, args=args)
    elif args.module == 'GAT':
        model = Gen_GAT(args.brain_atlas, args.num_classes, args=args)
    elif args.module == 'SAGE':
        model = Gen_SAGE(args.brain_atlas, args.num_classes, args=args)
    elif args.module == 'GIN':
        model = Gen_GIN(args.brain_atlas, args.num_classes, args=args)
    elif args.module == 'TopK':
        model = TopK(args.brain_atlas, args.num_classes, args=args)

    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)

    wandb.watch(model, log='all')

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    # optimizer = torch.optim.SGD(model.parameters(), args.lr,
    #                             momentum=args.momentum,
    #                             weight_decay=args.weight_decay)

    optimizer = torch.optim.Adam(model.parameters(), args.lr,
                                weight_decay=args.weight_decay)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    # Data loading code
    traindir = os.path.join(args.data, 'train')

    if args.dataset == '1000C':
        args.brain_atlas = 177   
        fmri = io.loadmat(os.path.join(args.data, '1000_connectome_fmri.mat'))['fmri']['matrix'][0]
        age = pd.read_csv(os.path.join(args.data, '1000_connectome_fmri.csv'))

    if args.dataset == 'NKI':
        args.brain_atlas = 188
        fmri = io.loadmat(os.path.join(args.data, 'normal_fmri_matrix.mat'))['fmri']['matrix'][0]
        age = pd.read_csv(os.path.join(args.data, 'NKI_age.csv'))

    if args.dataset == 'ADHD':
        args.brain_atlas = 190
        fmri_initial = io.loadmat(os.path.join(args.data, 'adhd_fmri.mat'))['matrix']
        age_initial = pd.read_csv(os.path.join(args.data, 'adhd_label.csv'))
        age = age_initial[age_initial['label'] == 'Typically Developing']
        age_health_index = age_initial[age_initial['label'] == 'Typically Developing'].index.tolist()
        fmri = fmri_initial[age_health_index]
        age = age.reset_index(drop=True)
        # fmri = fmri.reset_index(drop=True)

    age_array = age['age'].index.tolist()
    # print(age_array)

    age_selection_index = age_array.copy()
    for selection_index in age_selection_index:
        if fmri[selection_index].shape[0] != args.brain_atlas:
            age_array.remove(age_selection_index[selection_index])
    
    fmri = fmri[age_array]
    age_selection = age.loc[age_array]
    age_selection.reset_index(drop = True, inplace=True)
    age_array = age_selection['age'].index.tolist()

    index_train_list = np.random.choice(age_array, size=int(len(age_selection)*0.6), replace=False)
    index_val_test = np.setdiff1d(age_array, index_train_list)
    index_val_list = np.random.choice(index_val_test.copy(), int(0.5*index_val_test.shape[0]), replace=False)
    index_test_list = np.setdiff1d(index_val_test, index_val_list)
    
    train_dataset = fmri_dataset_connectome(args, index_train_list, fmri, age_selection, mode='train')
    val_dataset = fmri_dataset_connectome(args, index_val_list, fmri, age_selection, mode='val')
    test_dataset = fmri_dataset_connectome(args, index_test_list, fmri, age_selection, mode='test')

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size)

    best_perform = 0
    test_flag = 1

    args.cos = True

    for epoch in range(args.start_epoch, args.epochs):

        adjust_learning_rate(optimizer, epoch, args)

        # train for one epoch
        train_loss = train(train_loader, model, optimizer, epoch, args)
        val_loss, val_mae, val_r2, val_rmse = val(val_loader, model)

        if test_flag:
            best_perform = val_loss
            test_loss = 0
            test_flag = 0
            test_mae = 0
            test_r2 = 0
            test_flag = 0
            test_rmse = 0

        if val_loss < best_perform:
            test_loss, test_mae, test_r2, test_rmse = val(test_loader, model)
            best_perform = val_loss

        wandb.log({'train_loss': train_loss,
            'learning_rate': optimizer.param_groups[0]["lr"],
            'val_loss': val_loss,
            'test_loss': test_loss,
            'test_mae': test_mae,
            'test_r2': test_r2,
            'test_rmse': test_rmse}) 

def train(train_loader, model, optimizer, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    # InfoNCE loss
    
    loss_batch = 0
    loss_mae = 0
    loss_rmse = 0
    loss_r2 = 0
    acc_batch = 0
    fp_fn = 0
    for i, (fmri, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        
        fmri = fmri.cuda()
        adj_fmri = fmri
        # for adj_index in np.arange(fmri.shape[0]):
        #     adj_fmri[adj_index] = torch.where(fmri[adj_index]>=0.85*torch.max(fmri[adj_index]), fmri[adj_index], 0)
        # compute output
        logits, info_loss = model(fmri, adj_fmri)
        # pred = logits.squeeze().max(dim=1)[1]

        # print(logits.shape)

        loss2 = nn.MSELoss()(logits.squeeze(), target.squeeze().cuda())
        mae_loss = F.l1_loss(logits.squeeze(), target.squeeze().cuda())
        rmse_loss = torch.sqrt(loss2)
        r2_loss = MF.r2_score(logits.squeeze(), target.squeeze().cuda())

        loss =  loss2 + info_loss

        # acc_batch += pred.eq(target.view(-1).cuda()).sum().item()
        # fp_fn += pred.ne(target.view(-1).cuda()).sum().item()

        losses.update(loss.item())
        loss_batch = loss_batch + loss.item()

        # acc = accuracy(output, target)[0] 
        # acc_inst.update(acc[0], images[0].size(0))

        # # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # loss_batch = loss_batch + loss.item()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            # losses.update(loss_batch)
            # loss_batch = 0
            progress.display(i)
            # # compute gradient and do SGD step
            # optimizer.step()
            # optimizer.zero_grad()

    return loss_batch

def val(val_loader, model):

    # switch to eval mode
    model.eval()
    
    loss_batch = 0
    loss_mae = 0
    loss_rmse = 0
    loss_r2 = 0
    acc_batch = 0
    fp_fn = 0
    for i, (fmri, target) in enumerate(val_loader):
        
        fmri = fmri.cuda()
        adj_fmri = fmri
        # for adj_index in np.arange(fmri.shape[0]):
        #     adj_fmri[adj_index] = torch.where(fmri[adj_index]>=0.85*torch.max(fmri[adj_index]), fmri[adj_index], 0)
        # # compute output
        logits, info_loss = model(fmri, adj_fmri)
        # pred = logits.squeeze().max(dim=1)[1]

        loss2 = nn.MSELoss()(logits.squeeze(), target.squeeze().cuda())
        mae_loss = F.l1_loss(logits.squeeze(), target.squeeze().cuda())
        rmse_loss = torch.sqrt(loss2)
        r2_loss = MF.r2_score(logits.squeeze(), target.squeeze().cuda())
     
        loss =  loss2
        loss_batch = loss_batch + loss.item()
        # acc = accuracy(output, target)[0] 
        # acc_inst.update(acc[0], images[0].size(0))
        loss_mae += mae_loss.item()
        loss_rmse += rmse_loss.item()
        loss_r2 += r2_loss.item()
    loss_mae /= len(val_loader)
    loss_rmse /= len(val_loader)
    loss_r2 /= len(val_loader)
    loss_batch /= len(val_loader)

    return loss_batch, loss_mae, loss_r2, loss_rmse

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')

def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate based on schedule"""
    lr = args.lr
    if args.cos:  # cosine lr schedule
        lr *= 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))
    else:  # stepwise lr schedule
        for milestone in args.schedule:
            lr *= 0.1 if epoch >= milestone else 1.
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

if __name__ == '__main__':
    main()