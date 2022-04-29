import numpy as np
import torch
import os

np.random.seed(2022)

os.makedirs('data/train', exist_ok=True)
os.makedirs('data/test', exist_ok=True)
os.makedirs('data/val', exist_ok=True)

train = torch.load('data/MNIST/processed/training.pt')
test = torch.load('data/MNIST/processed/test.pt')

idx_train = np.random.permutation(train[0].shape[0])
idx_test = np.random.permutation(test[0].shape[0])

train = (train[0][idx_train, :, :], train[1][idx_train])
test = (test[0][idx_test, :, :], test[1][idx_test])

t_train = int(0.85 * train[0].shape[0])
t_test = int(0.85 * test[0].shape[0])

torch.save((train[0][:t_train, :, :], train[1][:t_train]), 'data/train/train.pt')
torch.save((test[0][:t_test, :, :], test[1][:t_test]), 'data/test/test.pt')
val = (
    torch.cat((train[0][t_train:, :, :], test[0][t_test:, :, :])),
    torch.cat((train[1][t_train:], test[1][t_test:]))
)

torch.save(val, 'data/val/val.pt')