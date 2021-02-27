import torchvision


trainset = torchvision.datasets.CIFAR10(root='../data', train=True,
                                        download=False, transform=None)

testset = torchvision.datasets.CIFAR10(root='../data', train=False,
                                       download=False, transform=None)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
