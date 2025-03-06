import torch
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets

import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim

from sklearn.model_selection import train_test_split

from models import modelB, create_data_loaders, train, test
from utils import smart_device, safe_torch_save

import argparse

class Cifar10NetSimple(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = torch.log_softmax(x, dim=1)
        return x

class Cifar10Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = torch.log_softmax(x, dim=1)
        return x

def train_val(model_class, train_loader, val_loader, loss_fn, optimizer, args, frac_train=0.8, num_epochs=100, batch_size=32, device='cuda', path_out_model=None):
    # train_idx, val_idx = train_test_split(list(range(len(dataset))), train_size=frac_train)
    # train_loader, val_loader = create_data_loaders(dataset, (train_idx, val_idx), batch_size=batch_size)

    model = model_class().to(device)
    
    for epoch in range(1, num_epochs + 1):
        train_loss, train_accuracy = train(model, device, train_loader, loss_fn, optimizer, epoch, args)
        # train_loss = train_one_epoch(model, train_loader, loss_fn, optimizer, device)
        # val_loss, val_accuracy = evaluate(model, val_loader, loss_fn, device)
        val_loss, val_accuracy = test(model, device, val_loader, loss_fn)
        print(f'Epoch {epoch}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}')

    if path_out_model is not None:
        safe_torch_save(model,path_out_model)

    return model, (val_loss, val_accuracy)

#---
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

batch_size = 4
num_epochs = 2
trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

model = Cifar10Net()
loss_fn = nn.NLLLoss() # criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
path_out_model = "models/cifar10net.pt"
    
"""
# Getting MNIST data
transform=transforms.Compose([transforms.ToTensor(), 
                                  transforms.Normalize((0.1307,), (0.3081,))])
trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
testset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
batch_size=32
num_epochs=50
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)
classes = (0,1,2,3,4,5,6,7,8,9)
    
model = modelB()
loss_fn = nn.NLLLoss() # criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
path_out_model = "models/mnist_modelB_from_cifar_code.pt"
"""
#---

def main():
    parser = argparse.ArgumentParser(description='compute cifar10 model')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
    
    args = parser.parse_args()

    dir_out = 'models'
    device = smart_device()
    
    # model, results = train_val(Cifar10Net, trainloader, testloader, loss_fn, optimizer, args, frac_train=0.8, num_epochs=50, batch_size=batch_size, device=device, path_out_model="models/cifar10net.pt")
    for epoch in range(num_epochs):  # loop over the dataset multiple times
        print(f"Epoch {epoch} of {num_epochs}")
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                running_loss = 0.0
    print('Finished Training')

    if path_out_model is not None:
        safe_torch_save(model,path_out_model)
    
    model.load_state_dict(torch.load(path_out_model))

    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            # calculate outputs by running images through the network
            outputs = model(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')
    
    # prepare to count predictions for each class
    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}
    # again no gradients needed
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = model(images)
            _, predictions = torch.max(outputs, 1)
            # collect the correct predictions for each class
            for label, prediction in zip(labels, predictions):
                if label == prediction:
                    correct_pred[classes[label]] += 1
                total_pred[classes[label]] += 1
    # print accuracy for each class
    for classname, correct_count in correct_pred.items():
        accuracy = 100 * float(correct_count) / total_pred[classname]
        if isinstance(classname,str):
            print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')
        else:
            print(f'Accuracy for class: {classname} is {accuracy:.1f} %')

if __name__ == "__main__":
    main()