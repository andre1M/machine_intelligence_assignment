from global_vars import DEVICE, EPOCHS, LR, BATCH_SIZE, THRESHOLD, SEED, MOMENTUM, WD
from utils import compute_statistics, train
from model import resnet20

from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from torch import nn, optim
import torch


# for reproducibility
torch.manual_seed(SEED)

# define transforms
trainset = datasets.CIFAR10(
    root='../data',
    train=True,
    download=False,
    transform=transforms.ToTensor()
)
mean, std = compute_statistics(trainset)
del trainset

train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
])
test_transform = transforms.Compose([
    transforms.Resize(32),
    transforms.ToTensor(),
    # transforms.Normalize(mean=mean, std=std)
])

# define dataset
trainset = datasets.CIFAR10(
    root='../data',
    train=True,
    download=False,
    transform=train_transform
)
testset = datasets.CIFAR10(
    root='../data',
    train=False,
    download=False,
    transform=test_transform
)

trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)
testloader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=True)

dataloaders = dict(train=trainloader, val=testloader)

# initialize model
model = resnet20(10)

print('Device:', DEVICE)
model = model.to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(
    params=model.parameters(),
    lr=LR,
    momentum=MOMENTUM,
    weight_decay=WD
)

# train
model, hist = train(
    model=model,
    dataloaders=dataloaders,
    criterion=criterion,
    optimizer=optimizer,
    num_epochs=EPOCHS,
    threshold=THRESHOLD
)
