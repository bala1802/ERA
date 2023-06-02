import torch
import matplotlib.pyplot as plt
from torchvision import datasets, transforms

def constructTrainTransforms():
  train_transforms = transforms.Compose([
    transforms.RandomApply([transforms.CenterCrop(22), ], p=0.1),
    transforms.Resize((28, 28)),
    transforms.RandomRotation((-15., 15.), fill=0),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
    ])
  return train_transforms

def constructTestTransforms():
  test_transforms = transforms.Compose([
      transforms.ToTensor(),
      transforms.Normalize((0.1307,), (0.3081,)),
      ])
  return test_transforms

def constructTrainTestLoader(train_data, test_data):
  batch_size = 512
  kwargs = {'batch_size': batch_size, 'shuffle': True, 'num_workers': 2, 'pin_memory': True}

  test_loader = torch.utils.data.DataLoader(test_data, **kwargs)
  train_loader = torch.utils.data.DataLoader(train_data, **kwargs)

  return train_loader, test_loader

def load_and_transform_TrainData(train_transforms):
  train_data = datasets.MNIST('../data', train=True, download=True, transform=train_transforms)
  return train_data

def load_and_transform_TestData(test_transforms):
  test_data = datasets.MNIST('../data', train=False, download=True, transform=test_transforms)
  return test_data

def viewTrain_data(count, train_loader):
  batch_data, batch_label = next(iter(train_loader))
  fig = plt.figure()

  for i in range(count):
    plt.subplot(3,4,i+1)
    plt.tight_layout()
    plt.imshow(batch_data[i].squeeze(0), cmap='gray')
    plt.title(batch_label[i].item())
    plt.xticks([])
    plt.yticks([])

  return fig

def getDevice():
  return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def GetCorrectPredCount(pPrediction, pLabels):
  return pPrediction.argmax(dim=1).eq(pLabels).sum().item()

def train_test_metrics(train_losses, train_acc, test_losses, test_acc):
  fig, axs = plt.subplots(2,2,figsize=(15,10))
  axs[0, 0].plot(train_losses)
  axs[0, 0].set_title("Training Loss")
  axs[1, 0].plot(train_acc)
  axs[1, 0].set_title("Training Accuracy")
  axs[0, 1].plot(test_losses)
  axs[0, 1].set_title("Test Loss")
  axs[1, 1].plot(test_acc)
  axs[1, 1].set_title("Test Accuracy")
  
  return fig