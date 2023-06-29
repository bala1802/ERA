import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim

import torch.nn as nn
import torch.nn.functional as F

from torchvision import datasets, transforms
import random
import matplotlib.pyplot as plt

from tqdm import tqdm
from torch.optim.lr_scheduler import StepLR

import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchvision.transforms.functional import to_tensor

import numpy as np

SEED = 1

# CUDA?
cuda = torch.cuda.is_available()
print("CUDA Available?", cuda)

# For reproducibility
torch.manual_seed(SEED)

if cuda:
    torch.cuda.manual_seed(SEED)

class Transforms:
    def __init__(self, transforms: A.Compose):
        self.transforms = transforms

    def __call__(self, img, *args, **kwargs):
        return self.transforms(image=np.array(img))

'''
Load Train Data
'''
def load_train_data():
    transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=45, p=0.5),
        A.CoarseDropout(max_holes=1, max_height=16, max_width=16, min_holes = 1, min_height=16, min_width=16, fill_value=[0.4914, 0.4822, 0.4465], mask_fill_value = None, p=0.5),
        A.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ToTensorV2()
    ], additional_targets={"image": "image"})
    
    train = datasets.CIFAR10(root='./data', train=True, download=True)
    train.transform = lambda img: transform(image=np.array(img))["image"]
    return train

'''
Load Test Data
'''
def load_test_data():
    transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=45, p=0.5),
        A.CoarseDropout(max_holes=1, max_height=16, max_width=16, min_holes = 1, min_height=16, min_width=16, fill_value=[0.4914, 0.4822, 0.4465], mask_fill_value = None, p=0.5),
        A.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ToTensorV2()
    ], additional_targets={"image": "image"})
    
    test = datasets.CIFAR10(root='./data', train=False, download=True)
    test.transform = lambda img: transform(image=np.array(img))["image"]
    return test

'''
Load Train Loader
'''
def loadTrainLoader(train):
    dataloader_args = dict(shuffle=True, batch_size=128, num_workers=4, pin_memory=True) if cuda else dict(shuffle=True, batch_size=64)
    train_loader = torch.utils.data.DataLoader(train, **dataloader_args)
    return train_loader

'''
Load Test Loader
'''
def loadTestLoader(test):
    dataloader_args = dict(shuffle=True, batch_size=128, num_workers=4, pin_memory=True) if cuda else dict(shuffle=True, batch_size=64)
    test_loader = torch.utils.data.DataLoader(test, **dataloader_args)
    return test_loader

'''
Load Class Names
'''
def load_class_names():
    return ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

'''
Explain The training data
'''
def explainData(train_data, test_data, train_loader, test_loader):
    print(f'Train Data Length : ', len(train_data))
    print(f'Shape of Each image present in the train_data: {train_data[0][0].shape}')
    print(f'Test Data Length : ', len(test_data))
    print(f'Shape of Each image present in the test_data: {test_data[0][0].shape}')
    print(f'Train Loader Length : ', len(train_loader))
    print(f'Test Loader Length : ', len(test_loader))

'''
Visualize the random images from the Training data
'''
def visualize_random_images(train_loader):
    class_names = load_class_names()
    random_batch = random.choice(list(train_loader))
    images, labels = random_batch
    num_images_to_print = 10
    random_indices = random.sample(range(128), num_images_to_print)
    fig, axes = plt.subplots(num_images_to_print // 5, 5, figsize=(8, 5))
    
    for i, index in enumerate(random_indices):
        image = images[index]
        label = labels[index]
    
        # Reshape the image tensor to (C, H, W) format
        image = image.permute(1, 2, 0)
        image = image.clamp(0, 1)
    
        # Plot the image
        ax = axes[i // 5, i % 5]
        ax.imshow(image)
        ax.set_title(f"Label: {class_names[label]}")  # Use the class name
        ax.axis('off')
    
    plt.tight_layout()
    plt.show()

'''
Training the model
'''
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

'''
Testing the model
'''
def test(model, device, test_loader):
    test_losses_for_each_epoch = []
    test_acc_for_each_epoch = []
    misclassified_images = []
    misclassified_labels = []
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            
            # Track misclassified samples
            incorrect_mask = ~pred.eq(target.view_as(pred)).squeeze()
            misclassified_images.extend(data[incorrect_mask])
            misclassified_labels.extend(target[incorrect_mask].squeeze())
            
            correct += pred.eq(target.view_as(pred)).sum().item()
            
    test_loss /= len(test_loader.dataset)
    test_losses_for_each_epoch.append(test_loss)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(test_loss, correct, len(test_loader.dataset), 100. * correct / len(test_loader.dataset)))

    test_acc_for_each_epoch.append(100. * correct / len(test_loader.dataset))
    
    return test_acc_for_each_epoch, test_losses_for_each_epoch, misclassified_images, misclassified_labels

'''
Run the training and testing for the given number of epochs
'''
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
    
    misclassified_images_for_all_epochs = []
    misclassified_labels_for_all_epochs = []
    
    for epoch in range(EPOCHS):
        print("EPOCH:", epoch)
        train_acc_for_each_epoch, train_losses_for_each_epoch = train(model=model, device=device, 
                                                                      train_loader=train_loader, optimizer=optimizer, 
                                                                      epoch=epoch)
        test_acc_for_each_epoch, test_losses_for_each_epoch, local_misclassified_images, local_correct_labels = test(model=model, device=device, 
                                                                   test_loader=test_loader)
    
        train_acc.extend(train_acc_for_each_epoch)
        train_losses.extend(train_losses_for_each_epoch)
    
        test_acc.extend(test_acc_for_each_epoch)
        test_losses.extend(test_losses_for_each_epoch)
        
        misclassified_images_for_all_epochs.extend(local_misclassified_images)
        misclassified_labels_for_all_epochs.extend(local_correct_labels)
        
    return train_acc, train_losses, test_acc, test_losses, misclassified_images_for_all_epochs, misclassified_labels_for_all_epochs

'''
Plot the train and test results
'''
def plot_results(train_acc, train_losses, test_acc, test_losses):
    fig, axs = plt.subplots(2,2,figsize=(20,10))
    t = [t_items.item() for t_items in train_losses]
    axs[0, 0].plot(t)
    axs[0, 0].set_title("Training Loss")
    axs[1, 0].plot(train_acc)
    axs[1, 0].set_title("Training Accuracy")
    axs[0, 1].plot(test_losses)
    axs[0, 1].set_title("Test Loss")
    axs[1, 1].plot(test_acc)
    axs[1, 1].set_title("Test Accuracy")

'''
Display the misclassified images
'''
def display_misclassified_images(images, labels):
    num_images = len(images)
    num_rows = num_images // 5 + 1

    fig, axes = plt.subplots(num_rows, 5, figsize=(12, 2 * num_rows))
    class_names = load_class_names()
    
    counter = 0
    for i, (image, label) in enumerate(zip(images, labels)):
        ax = axes[i // 5, i % 5]
        image = image.cpu().numpy().transpose(1, 2, 0)
        ax.imshow(image)
        ax.set_title("Correct Label: {}".format(class_names[label.item()]))
        ax.axis('off')
        
        counter += 1
        if counter == 10:
            break

    plt.tight_layout()
    plt.show()

'''
Load the Optimizer model
'''
def load_optimizer(model):
    return optim.SGD(model.parameters(), lr=0.01, momentum=0.9)