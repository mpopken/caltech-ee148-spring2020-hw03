from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
from matplotlib import pyplot as plt
import sklearn
from sklearn.metrics import confusion_matrix
from sklearn.manifold import TSNE
import seaborn
import pandas as pd

import os

'''
This code is adapted from two sources:
(i) The official PyTorch MNIST example (https://github.com/pytorch/examples/blob/master/mnist/main.py)
(ii) Starter code from Yisong Yue's CS 155 Course (http://www.yisongyue.com/courses/cs155/2020_winter/)
'''

class fcNet(nn.Module):
    '''
    Design your model with fully connected layers (convolutional layers are not
    allowed here). Initial model is designed to have a poor performance. These
    are the sample units you can try:
        Linear, Dropout, activation layers (ReLU, softmax)
    '''
    def __init__(self):
        # Define the units that you will use in your model
        # Note that this has nothing to do with the order in which operations
        # are applied - that is defined in the forward function below.
        super(fcNet, self).__init__()
        self.fc1 = nn.Linear(in_features=784, out_features=20)
        self.fc2 = nn.Linear(20, 10)
        self.dropout1 = nn.Dropout(p=0.5)

    def forward(self, x):
        # Define the sequence of operations your model will apply to an input x
        x = torch.flatten(x, start_dim=1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        x = F.relu(x)

        output = F.log_softmax(x, dim=1)
        return output


class ConvNet(nn.Module):
    '''
    Design your model with convolutional layers.
    '''
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(3,3), stride=1)
        self.conv2 = nn.Conv2d(8, 8, 3, 1)
        self.dropout1 = nn.Dropout2d(0.5)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(200, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout2(x)

        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)

        output = F.log_softmax(x, dim=1)
        return output


class Net(nn.Module):
    '''
    Build the best MNIST classifier
    '''
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(3,3), stride=1)
        self.conv2 = nn.Conv2d(8, 8, 5, 1)
        # self.conv3 = nn.Conv2d(8, 8, 5, 1)

        self.drop1 = nn.Dropout2d(0.5)
        self.drop2 = nn.Dropout2d(0.5)
        # self.drop3 = nn.Dropout2d(0.5)

        self.batc1 = nn.BatchNorm2d(8)
        self.batc2 = nn.BatchNorm2d(8)
        # self.batc3 = nn.BatchNorm2d(8)
        
        self.line1 = nn.Linear(968, 256)
        self.line2 = nn.Linear(256, 64)
        self.line3 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.batc1(x)
        x = F.relu(x)
        # x = F.max_pool2d(x, 2)
        x = self.drop1(x)

        x = self.conv2(x)
        x = self.batc2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.drop2(x)

        # x = self.conv3(x)
        # x = self.batc3(x)
        # x = F.relu(x)
        # x = F.max_pool2d(x, 2)
        # x = self.drop3(x)

        x = torch.flatten(x, 1)
        x = self.line1(x)
        x = F.relu(x)
        x = self.line2(x)
        x = F.relu(x)
        x = self.line3(x)

        output = F.log_softmax(x, dim=1)
        return output

    def get_feature_vector(self, x):
        x = self.conv1(x)
        x = self.batc1(x)
        x = F.relu(x)
        # x = F.max_pool2d(x, 2)
        x = self.drop1(x)

        x = self.conv2(x)
        x = self.batc2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.drop2(x)

        # x = self.conv3(x)
        # x = self.batc3(x)
        # x = F.relu(x)
        # x = F.max_pool2d(x, 2)
        # x = self.drop3(x)

        x = torch.flatten(x, 1)
        x = self.line1(x)
        x = F.relu(x)
        x = self.line2(x)

        return x

def train(args, model, device, train_loader, optimizer, epoch):
    '''
    This is your training function. When you call this function, the model is
    trained for 1 epoch.
    '''
    model.train()   # Set the model to training mode
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()               # Clear the gradient
        output = model(data)                # Make predictions
        loss = F.nll_loss(output, target)   # Compute loss
        loss.backward()                     # Gradient computation
        optimizer.step()                    # Perform a single optimization step
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.sampler),
                100. * batch_idx / len(train_loader), loss.item()))

    


def test(model, device, test_loader, method='val', check_failures=False):
    assert(method in ['train', 'val'])
    model.eval()    # Set the model to inference mode
    test_loss = 0
    correct = 0
    test_num = 0
    results = []
    with torch.no_grad():   # For the inference step, gradient is not computed
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            flag = pred.eq(target.view_as(pred))
            correct += flag.sum().item()
            test_num += len(data)

            if check_failures:
                flag = np.arange(flag.shape[0])[~flag[:, 0]]
                results.extend([
                    (data[flag[i], :, :, :], pred[flag[i], 0])
                    for i in range(flag.shape[0])
                ])

    test_loss /= test_num

    print('\n{} set: Average loss: {:.4f}, Accuracy: {}/{} ({:.1f}%)\n'.format(
        'Validation' if method == 'val' else 'Training',
        test_loss, correct, test_num,
        100. * correct / test_num))

    if check_failures:
        return results
    return 100. * correct / test_num


def main():
    torch.manual_seed(2022)

    # Training settings
    # Use the command line to modify the default settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=14, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--step', type=int, default=1, metavar='N',
                        help='number of epochs between learning rate reductions (default: 1)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')

    parser.add_argument('--evaluate', action='store_true', default=False,
                        help='evaluate your model on the official test set')
    parser.add_argument('--load-model', type=str,
                        help='model file path')

    parser.add_argument('--save-model', action='store_true', default=True,
                        help='For Saving the current Model')

    parser.add_argument('--train-frac', type=float, default=1., metavar='f',
                        help='Choose the fraction of the training set to use')
    parser.add_argument('--plot-curves', action='store_true', default=False,
                        help='Plot the learning curves')
    parser.add_argument('--check-failures', action='store_true', default=False,
                        help='Visualize images on which the model failed')
    parser.add_argument('--kernels', action='store_true', default=False,
                        help='Visualize the kernels for the first convolutional layer')
    parser.add_argument('--confusion', action='store_true', default=False,
                        help='Visualize the confusion matrix')
    parser.add_argument('--feature-vectors', action='store_true', default=False,
                        help='Visualize the high-dimensional embedding')
    

    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    # Evaluate on the official test set
    if args.evaluate:
        assert os.path.exists(args.load_model)

        # Set the test model
        model = fcNet().to(device)
        model.load_state_dict(torch.load(args.load_model))

        test_dataset = datasets.MNIST('data', train=False,
                    transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,))
                    ]))

        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=args.test_batch_size, shuffle=True, **kwargs)

        test(model, device, test_loader)

        return

    # Pytorch has default MNIST dataloader which loads data at each iteration
    train_dataset = datasets.MNIST('../data', train=True, download=True,
                transform=transforms.Compose([       # Data preprocessing
                    transforms.ToTensor(),           # Add data augmentation here
                    # transforms.RandomPerspective(),
                    # transforms.GaussianBlur(3),
                    # transforms.RandomAffine(0, (0.1, 0.1)),
                    transforms.Normalize((0.1307,), (0.3081,))
                ]))

    # You can assign indices for training/validation or use a random subset for
    # training by using SubsetRandomSampler. Right now the train and validation
    # sets are built from the same indices - this is bad! Change it so that
    # the training and validation sets are disjoint and have the correct relative sizes.
    subset_indices_train = np.random.choice(len(train_dataset), size=int(0.85*len(train_dataset)), replace=False)
    subset_indices_valid = np.delete(np.arange(len(train_dataset)), subset_indices_train)

    # Use only a fraction of the total training set.
    subset_indices_train = np.random.choice(
        subset_indices_train,
        size=int(args.train_frac * subset_indices_train.shape[0]),
        replace=False
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size,
        sampler=SubsetRandomSampler(subset_indices_train)
    )
    val_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.test_batch_size,
        sampler=SubsetRandomSampler(subset_indices_valid)
    )

    torch.utils.data.DataLoader

    # Load your model [fcNet, ConvNet, Net]
    model = Net().to(device)

    # Try different optimizers here [Adam, SGD, RMSprop]
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    # Set your learning rate scheduler
    scheduler = StepLR(optimizer, step_size=args.step, gamma=args.gamma)


    # Training loop
    train_accuracy = []
    val_accuracy = []
    kernels = []
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        train_accuracy.append(test(model, device, train_loader, method='train'))
        val_accuracy.append(test(model, device, val_loader, method='val'))
        scheduler.step()    # learning rate scheduler

        # Visualize the kernels
        kernels.append(model.parameters().__next__()[np.random.randint(8)])

    if args.kernels:
        print(len(kernels))
        f, axarr = plt.subplots(3, 3)
        for i, k in enumerate(kernels[-9:]):
            axarr[i // 3, i % 3].imshow(transforms.ToPILImage()(k))
        plt.savefig('kernels.png', format='png')


    if args.save_model:
        torch.save(model.state_dict(), "mnist_model.pt")

    # Print learning curves.
    if args.plot_curves:
        plt.figure()
        plt.plot(range(1, args.epochs + 1), train_accuracy, label='Training')
        plt.plot(range(1, args.epochs + 1), val_accuracy, label='Validation')
        plt.legend()
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.title('Learning Curves')
        plt.savefig('learning_curves.png', format='png')

    # Visualize the failures.
    if args.check_failures:
        fails = test(model, device, val_loader, method='val', check_failures=True)[:9]
        f, axarr = plt.subplots(3, 3)
        for i, fail in enumerate(fails):
            axarr[i // 3, i % 3].imshow(transforms.ToPILImage()(fail[0] * 0.3081 + 0.1307))
            axarr[i // 3, i % 3].set_title(f'Prediction: {fail[1]}')
        plt.savefig('missed_preds.png', format='png')

    # Confusion
    if args.confusion:
        y_pred = []
        y_true = []

        for ims, target in val_loader:
                preds = model(ims) # Feed Network

                preds = (torch.max(torch.exp(preds), 1)[1]).data.cpu().numpy()
                y_pred.extend(preds) # Save Prediction
                
                target = target.data.cpu().numpy()
                y_true.extend(target) # Save Truth

        # Build confusion matrix
        cf_matrix = confusion_matrix(y_true, y_pred)
        cf_matrix = cf_matrix / np.sum(cf_matrix) * 10
        plt.figure()
        seaborn.heatmap(cf_matrix, annot=True)
        plt.savefig('confusion.png')

    if args.feature_vectors:
        vectors = np.empty((0, 64))
        ims = []
        for im, target in val_loader:
            for i in range(im.shape[0]):
                ims.append(im[i, :, :, :])
            vectors = np.concatenate([vectors, model.get_feature_vector(im).detach().numpy()])

        # X_embedded = TSNE(n_components=2, n_jobs=1, init='random').fit_transform(vectors)
        # print(X_embedded.shape)

        # Closest vectors
        idx = np.random.choice(vectors.shape[0], size=4, replace=False)
        xs = vectors[idx, :].reshape(1, 64, 4)
        dist = np.sqrt(((np.repeat(vectors.reshape(9000, 64, 1), 4, axis=2) - xs) ** 2).sum(axis=1))
        close_idx = np.broadcast_to(np.arange(9000), (4, 9000))[np.argsort(dist, axis=0).T < 8].reshape(4, 8)

        f, axarr = plt.subplots(4, 9)
        for k, i in enumerate(idx):
            axarr[k, 0].imshow(transforms.ToPILImage()(ims[i]))
            for j in range(8):
                axarr[k, j+1].imshow(transforms.ToPILImage()(ims[close_idx[k, j]]))
        plt.show()

if __name__ == '__main__':
    main()
