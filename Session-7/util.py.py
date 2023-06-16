import torch

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torchvision import datasets, transforms

import matplotlib.pyplot as plt
from tqdm import tqdm

from torch.optim.lr_scheduler import StepLR

SEED = 1

# CUDA?
cuda = torch.cuda.is_available()
print("CUDA Available?", cuda)

# For reproducibility
torch.manual_seed(SEED)

if cuda:
    torch.cuda.manual_seed(SEED)

def loadTrainData():
    print("Loading Train Data.....")
    train_transforms = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train = datasets.MNIST('./data', train=True, download=True, transform=train_transforms)
    return train

'''
This is BAD CODE, #FIXME
'''
def loadTrainData_for_Model_11():
    print("Loading Train Data for Model_11.....")
    train_transforms = transforms.Compose([transforms.RandomRotation((-7.0, 7.0), fill=(1,)), transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train = datasets.MNIST('./data', train=True, download=True, transform=train_transforms)
    return train


def loadTestData():
    print("Loading Testing Data.....")
    test_transforms = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))])
    test = datasets.MNIST('./data', train=False, download=True, transform=test_transforms)
    return test

'''
This is BAD CODE, #FIXME
'''
def loadTestData_Model_11():
    print("Loading Testing Data for Model_11.....")
    test_transforms = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))])
    test = datasets.MNIST('./data', train=False, download=True, transform=test_transforms)
    return test

def loadTrainLoader(train):
    dataloader_args = dict(shuffle=True, batch_size=128, num_workers=4, pin_memory=True) if cuda else dict(shuffle=True, batch_size=64)
    train_loader = torch.utils.data.DataLoader(train, **dataloader_args)
    return train_loader

def loadTestLoader(test):
    dataloader_args = dict(shuffle=True, batch_size=128, num_workers=4, pin_memory=True) if cuda else dict(shuffle=True, batch_size=64)
    test_loader = torch.utils.data.DataLoader(test, **dataloader_args)
    return test_loader

def explainTrainData(train, train_loader):
    train_data = train.train_data
    train_data = train.transform(train_data.numpy())
    
    print('[Train]')
    print(' - Numpy Shape:', train.train_data.cpu().numpy().shape)
    print(' - Tensor Shape:', train.train_data.size())
    print(' - min:', torch.min(train_data))
    print(' - max:', torch.max(train_data))
    print(' - mean:', torch.mean(train_data))
    print(' - std:', torch.std(train_data))
    print(' - var:', torch.var(train_data))
    
def displayImageByIndex(train_loader, index):
    dataiter = iter(train_loader)
    images, labels = next(dataiter)
    
    plt.imshow(images[index].numpy().squeeze(), cmap='gray_r')
   
def displayRandomImages(train_loader):
    dataiter = iter(train_loader)
    images, _ = next(dataiter)
    
    figure = plt.figure()
    num_of_images = 60
    for index in range(1, num_of_images + 1):
        plt.subplot(6, 10, index)
        plt.axis('off')
        plt.imshow(images[index].numpy().squeeze(), cmap='gray_r')

def train(model, device, train_loader, optimizer, epoch):
    model.train()
    pbar = tqdm(train_loader)
    correct = 0
    processed = 0
    train_losses_for_each_epoch = []
    train_acc_for_each_epoch = []
    
    for batch_idx, (data, target) in enumerate(pbar):
        # get samples
        data, target = data.to(device), target.to(device)
    
        # Init
        optimizer.zero_grad()
        # In PyTorch, we need to set the gradients to zero before starting to do backpropragation because PyTorch accumulates the gradients on subsequent backward passes. 
        # Because of this, when you start your training loop, ideally you should zero out the gradients so that you do the parameter update correctly.
    
        # Predict
        y_pred = model(data)
    
        # Calculate loss
        loss = F.nll_loss(y_pred, target)
        train_losses_for_each_epoch.append(loss)
    
        # Backpropagation
        loss.backward()
        optimizer.step()
    
        # Update pbar-tqdm
        
        pred = y_pred.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item()
        processed += len(data)
    
        pbar.set_description(desc= f'Loss={loss.item()} Batch_id={batch_idx} Accuracy={100*correct/processed:0.2f}')
        train_acc_for_each_epoch.append(100*correct/processed)
    return train_acc_for_each_epoch, train_losses_for_each_epoch

def test(model, device, test_loader):
    test_losses_for_each_epoch = []
    test_acc_for_each_epoch = []
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    test_losses_for_each_epoch.append(test_loss)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(test_loss, correct, len(test_loader.dataset),100. * correct / len(test_loader.dataset)))
    
    test_acc_for_each_epoch.append(100. * correct / len(test_loader.dataset))
    
    return test_acc_for_each_epoch, test_losses_for_each_epoch
    
def plot_results(train_acc, train_losses, test_acc, test_losses):
    fig, axs = plt.subplots(2,2,figsize=(10,5))
    t = [t_items.item() for t_items in train_losses]
    axs[0, 0].plot(t)
    axs[0, 0].set_title("Training Loss")
    axs[1, 0].plot(train_acc)
    axs[1, 0].set_title("Training Accuracy")
    axs[0, 1].plot(test_losses)
    axs[0, 1].set_title("Test Loss")
    axs[1, 1].plot(test_acc)
    axs[1, 1].set_title("Test Accuracy")
    
def run_epochs(model, device, train_loader, test_loader, numberOfEpochs):
    train_acc = []
    train_losses = []
    test_acc = []
    test_losses = []
    
    train_acc_for_each_epoch = [] 
    train_losses_for_each_epoch = []
    test_acc_for_each_epoch = [] 
    test_losses_for_each_epoch = []
    
    EPOCHS = numberOfEpochs
    
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    
    for epoch in range(EPOCHS):
        print("EPOCH:", epoch)
        train_acc_for_each_epoch, train_losses_for_each_epoch = train(model=model, device=device, 
                                                                      train_loader=train_loader, optimizer=optimizer, 
                                                                      epoch=epoch)
        test_acc_for_each_epoch, test_losses_for_each_epoch = test(model=model, device=device, 
                                                                   test_loader=test_loader)
    
        train_acc.extend(train_acc_for_each_epoch)
        train_losses.extend(train_losses_for_each_epoch)
    
        test_acc.extend(test_acc_for_each_epoch)
        test_losses.extend(test_losses_for_each_epoch)
    
    return train_acc, train_losses, test_acc, test_losses

'''
This is BAD Code, #FIX_ME
'''
def run_epochs_for_Model_11(model, device, train_loader, test_loader, numberOfEpochs):
    train_acc = []
    train_losses = []
    test_acc = []
    test_losses = []
    
    train_acc_for_each_epoch = [] 
    train_losses_for_each_epoch = []
    test_acc_for_each_epoch = [] 
    test_losses_for_each_epoch = []
    
    EPOCHS = numberOfEpochs
    
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    scheduler = StepLR(optimizer, step_size=6, gamma=0.1)
    
    for epoch in range(EPOCHS):
        print("EPOCH:", epoch)
        train_acc_for_each_epoch, train_losses_for_each_epoch = train(model=model, device=device, 
                                                                      train_loader=train_loader, optimizer=optimizer, 
                                                                      epoch=epoch)
        test_acc_for_each_epoch, test_losses_for_each_epoch = test(model=model, device=device, 
                                                                   test_loader=test_loader)
        
        scheduler.step()
        
        train_acc.extend(train_acc_for_each_epoch)
        train_losses.extend(train_losses_for_each_epoch)
    
        test_acc.extend(test_acc_for_each_epoch)
        test_losses.extend(test_losses_for_each_epoch)
    
    return train_acc, train_losses, test_acc, test_losses

def load_optimizer(model):
    return optim.SGD(model.parameters(), lr=0.01, momentum=0.9)