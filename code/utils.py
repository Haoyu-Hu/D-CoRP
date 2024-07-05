import time
import math
import torch
import torch.nn.functional as F
from torch import tensor
from torch.optim import Adam
from sklearn.model_selection import StratifiedKFold
from torch_geometric.data import Dataset, Batch, DataLoader
from torch_geometric.loader import LinkLoader
import torch.nn as nn
import os
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# torch.manual_seed(6789)
# np.random.seed(6789)
# torch.cuda.manual_seed_all(6789)
# os.environ['PYTHONHASHSEED'] = str(6789)

def cross_validation_with_val_set(dataset, model, folds, epochs, batch_size, lr,
                                  args, weight_decay, logger=None):
    val_losses, val_accs, test_accs, durations = [], [], [], []
    test_maes, test_rmses = [], []
    for fold, (train_idx, test_idx, val_idx) in enumerate(zip(*k_fold(dataset, folds))):
        # print(len(train_idx))
        # print(dataset[1])
        train_dataset = dataset[train_idx]
        test_dataset = dataset[test_idx]
        val_dataset = dataset[val_idx]

        fold_val_losses = []
        fold_val_accs = []
        fold_test_accs = []
        fold_test_maes = []
        fold_test_rmses = []

        infos = dict()

        train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size, shuffle=False)

        model.to(device).reset_parameters()
        optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        t_start = time.perf_counter()

        for epoch in range(1, epochs + 1):

            train_loss, train_mae, train_rmse = train_DR(model, optimizer, train_loader)
            val_loss, val_mae, val_rmse = eval_DR(model, val_loader)
            
            val_losses.append(val_loss)
            fold_val_losses.append(val_loss)

            test_loss, test_mae, test_rmse = eval_DR(model, test_loader)
            test_maes.append(test_mae)
            test_rmses.append(test_rmse)
            fold_test_maes.append(test_mae)
            fold_test_rmses.append(test_rmse)
            eval_info = {
                'fold': fold,
                'epoch': epoch,
                'train_loss': train_loss,
                'train_mae': train_mae,
                'val_loss': val_loss,
                'val_mae': val_mae,
                'test_mae': test_mae,
                'test_rmse': test_rmse
            }
            infos[epoch] = eval_info

            if logger is not None:
                logger(eval_info)

            # if epoch % lr_decay_step_size == 0:
            #     for param_group in optimizer.param_groups:
            #         param_group['lr'] = lr_decay_factor * param_group['lr']

            adjust_learning_rate(optimizer, epoch, args)
            if epoch % 10 == 0:
                print('Epoch: {:d}, train loss: {:.3f}, val loss: {:.5f}, val mae: {:.3f}, test mae: {:.3f}'
                      .format(epoch, eval_info["train_loss"], eval_info["val_loss"], eval_info["val_mae"], eval_info["test_mae"]))

        fold_val_loss, argmin = tensor(fold_val_losses).min(dim=0)
        fold_test_mae = fold_test_maes[argmin]
        fold_test_rmse = fold_test_rmses[argmin]

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        t_end = time.perf_counter()
        durations.append(t_end - t_start)
        print('Fold: {:d}, Val loss: {:.3f}, Test mae: {:.3f}'
              .format(eval_info["fold"], fold_val_loss, fold_test_mae))


    val_losses, duration = tensor(val_losses), tensor(durations)
    val_losses = val_losses.view(folds, epochs)
    test_maes, test_rmses = tensor(test_maes), tensor(test_rmses)
    test_maes, test_rmses = test_maes.view(folds, epochs), test_rmses.view(folds, epochs)


    min_val_loss, argmin = val_losses.min(dim=1)
    test_mae = test_maes[torch.arange(folds, dtype=torch.long), argmin]
    test_rmse = test_rmses[torch.arange(folds, dtype=torch.long), argmin]

    val_loss_mean = min_val_loss.mean().item()
    duration_mean = duration.mean().item()

    test_mae_mean = test_mae.mean().item()
    test_mae_std = test_mae.std().item()
    test_rmse_mean = test_rmse.mean().item()
    test_rmse_std = test_rmse.std().item()
    print('Val Loss: {:.4f}, Test MAE: {:.3f}+{:.3f}, Test RMSE: {:.3f}+{:.3f}, Duration: {:.3f}'
          .format(val_loss_mean, test_mae_mean, test_mae_std, test_rmse_mean, test_rmse_std, duration_mean))

    return 1

def k_fold(dataset, folds):
    skf = StratifiedKFold(folds, shuffle=True, random_state=6789)

    test_indices, train_indices = [], []

    for _, idx in skf.split(torch.zeros(len(dataset)), [age.long() for fmri, age in dataset]):
            test_indices.append(torch.from_numpy(idx).to(torch.long))
    # print(len(train_indices))
    # print(len(dataset))
    # print(dataset.y)

    val_indices = [test_indices[i - 1] for i in range(folds)]

    for i in range(folds):
        train_mask = torch.ones(len(dataset), dtype=torch.bool)
        train_mask[test_indices[i]] = 0
        train_mask[val_indices[i]] = 0
        train_indices.append(train_mask.nonzero(as_tuple=False).view(-1))

    return train_indices, test_indices, val_indices

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

def train_DR(model, optimizer, loader):
    model.train()

    total_loss = 0
    total_mae = 0
    total_rmse = 0
    correct = 0
    for (data, y) in loader:
        optimizer.zero_grad()
        data = data.to(device)
        logits, info_loss = model(data, data)

        loss2 = nn.MSELoss()(logits.squeeze(), y.squeeze().cuda())
        mae_loss = F.l1_loss(logits.squeeze(), y.squeeze().cuda())
        rmse_loss = torch.sqrt(loss2)

        # print(info_loss)
        # print(len(str(torch.div(loss2, info_loss).item())))
        scaler = torch.Tensor([len(str(torch.div(loss2, info_loss).item()))]).to(device)
        # print(torch.pow(10, scaler)*info_loss)
        loss =  loss2 + info_loss
        loss.backward()
        total_loss += loss.item() * data.size(0)
        total_mae += mae_loss.item() * data.size(0)
        total_rmse += rmse_loss.item() * data.size(0)
        optimizer.step()
    return total_loss / len(loader.dataset), total_mae / len(loader.dataset), total_rmse / len(loader.dataset)

def eval_DR(model, loader):
    model.eval()

    total_loss = 0
    total_mae = 0
    total_rmse = 0
    correct = 0
    for (data, y) in loader:
        data = data.to(device)
        with torch.no_grad():
            logits, info_loss = model(data, data)

        loss2 = nn.MSELoss()(logits.squeeze(), y.squeeze().cuda())
        mae_loss = F.l1_loss(logits.squeeze(), y.squeeze().cuda())
        rmse_loss = torch.sqrt(loss2)

        loss =  loss2 + info_loss
        total_loss += loss.item() * data.size(0)
        total_mae += mae_loss.item() * data.size(0)
        total_rmse += rmse_loss.item() * data.size(0)
    return total_loss / len(loader.dataset), total_mae / len(loader.dataset), total_rmse / len(loader.dataset)