import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torchvision import datasets, transforms
import matplotlib.pyplot as plt
# from tqdm import tqdm

# from torchmetrics import Accuracy

from sklearn.cluster import KMeans
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import numpy as np


from utils import safe_torch_save

class modelB(nn.Module): #For MNIST images
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 8),
            nn.ReLU(),
            nn.Linear(8, 8),
            nn.ReLU(),
            nn.Linear(8, 10),
        )

    def forward(self, x, train = False):
        # if train:
        x = self.flatten(x)
        x = self.linear_relu_stack(x) # logits
        x = torch.log_softmax(x, dim=1) #log-probabilites
        return x

class modelC(nn.Module): #For MNIST images
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 8),
            nn.ReLU(),
            nn.Linear(8, 8),
            nn.ReLU(),
            nn.Linear(8, 8),
            nn.ReLU(),
            nn.Linear(8, 10),
        )

    def forward(self, x, train = False):
        # if train:
        x = self.flatten(x)
        x = self.linear_relu_stack(x) # logits
        x = torch.log_softmax(x, dim=1) #log-probabilites
        return x

def predict_probs_batch(model, batch, device):
    model.eval()
    with torch.no_grad():
        data, targets = batch
        data, targets = data.to(device), targets.to(device)
        # logits = model(data)
        # pred_probs = logits.softmax(dim=1)
        log_probs = model(data)
        pred_probs = log_probs.exp()
    return pred_probs

def predict_trueprob_batch(model, batch, device):
    model.eval()
    with torch.no_grad():
        data, targets = batch
        data, targets = data.to(device), targets.to(device)
        log_probs = model(data)
        pred_probs = log_probs.exp()
        # pred_trueprob = torch.zeros_like(targets,device=device)
        # for i, t in enumerate(targets):
        #     pred_trueprob[i] = pred_probs[i, t] 
        pred_trueprob = torch.gather(pred_probs,1,targets.unsqueeze(1))
    return pred_trueprob

def predict_lbls_batch(model, batch, device):
    probs = predict_probs_batch(model, batch, device)
    return probs.argmax(dim=1, keepdim=True)

def eval_acc_batch(model, batch, device):
    data, targets = batch
    data, targets = data.to(device), targets.to(device)    
    preds = predict_lbls_batch(model, batch, device)
    correct = preds.eq(targets.view_as(preds)).sum().item()
    total = targets.size(0)
    return correct, total

def training_step(model, device, batch, loss_fn, optimizer, batch_idx, epoch, args):
    data, target = batch
    data, target = data.to(device), target.to(device)
    optimizer.zero_grad()
    output = model(data)
    loss = loss_fn(output, target)
    loss.backward()
    optimizer.step()
    return loss

def test_step(model, device, batch, loss_fn):
    model.eval()
    with torch.no_grad():
        data, target = batch
        data, target = data.to(device), target.to(device)
        output = model(data)
        loss = loss_fn(output, target).sum().item()  # sum up batch loss
        # loss = loss_fn(output, target, reduction='sum').item()  # sum up batch loss
        correct, total = eval_acc_batch(model, batch, device)
    return loss, correct, total

def predict_probs_loader(model, data_loader, device):
    probs = []
    for batch_idx, batch in enumerate(data_loader):
        probs_batch = predict_probs_batch(model, batch, device)
        probs.append(probs_batch)
    return torch.cat(probs, dim=0)

def predict_trueprob_loader(model, data_loader, device):
    probs = []
    for batch_idx, batch in enumerate(data_loader):
        probs_batch = predict_trueprob_batch(model, batch, device)
        probs.append(probs_batch)
    return torch.cat(probs, dim=0)

def predict_lbls_loader(model, data_loader, device):
    preds = []
    for batch_idx, batch in enumerate(data_loader):
        preds_batch = predict_lbls_batch(model, batch, device)
        preds.append(preds_batch)
    return torch.cat(preds, dim=0)

def eval_acc_loader(model, data_loader, device):
    correct = 0
    samples = 0
    for batch_idx, batch in enumerate(data_loader):
        correct_batch, total_batch = eval_acc_batch(model, batch, device)
        correct += correct_batch
        samples += total_batch
    acc = correct / samples
    return acc

def train(model, device, train_loader, loss_fn, optimizer, epoch, args):
    loss = 0
    correct = 0
    total = 0
    for batch_idx, batch in enumerate(train_loader):
        model.train()
        loss_batch = training_step(model, device, batch, loss_fn, optimizer, batch_idx, epoch, args) 
        loss += loss_batch.item()
        correct_batch, total_batch = eval_acc_batch(model, batch, device)       
        correct += correct_batch
        total += total_batch
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            epoch, batch_idx * len(batch[0]), len(train_loader.dataset),
            100. * batch_idx / len(train_loader), loss))
        if args.dry_run:
            break
    avg_loss = loss / total
    accuracy = correct / total
    return avg_loss, accuracy

def test(model, device, test_loader, loss_fn):
    loss = 0
    correct = 0
    total = 0
    for batch in test_loader:
        loss_batch, correct_batch, total_batch = test_step(model, device, batch, loss_fn)
        loss += loss_batch
        correct += correct_batch
        total += total_batch
    avg_loss = loss / len(test_loader.dataset)
    accuracy = correct / total

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        avg_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    
    return avg_loss, accuracy

def get_loss_and_optimizer(model):
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    return loss_fn, optimizer


def train_val(model_class, dataset, args, frac_train=0.8, num_epochs=100, batch_size=32, device='cuda', path_out_model=None):
    train_idx, val_idx = train_test_split(list(range(len(dataset))), train_size=frac_train)
    train_loader, val_loader = create_data_loaders(dataset, (train_idx, val_idx), batch_size=batch_size)

    model = model_class().to(device)
    loss_fn, optimizer = get_loss_and_optimizer(model)

    for epoch in range(1, num_epochs + 1):
        train_loss, train_accuracy = train(model, device, train_loader, loss_fn, optimizer, epoch, args)
        # train_loss = train_one_epoch(model, train_loader, loss_fn, optimizer, device)
        # val_loss, val_accuracy = evaluate(model, val_loader, loss_fn, device)
        val_loss, val_accuracy = test(model, device, val_loader, loss_fn)
        print(f'Epoch {epoch}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}')

    if path_out_model is not None:
        safe_torch_save(model,path_out_model)

    return model, (val_loss, val_accuracy)


"""
def cross_validate(model_class, dataset, k_folds=5, num_epochs=100, batch_size=32, device='cuda', dir_out_models=None):
    kfold = KFold(n_splits=k_folds, shuffle=True)
    models = []
    results = []
    for fold, (train_idx, val_idx) in enumerate(kfold.split(dataset)):
        print(f'Fold {fold+1}/{k_folds}')
        train_loader, val_loader = create_data_loaders(dataset, (train_idx, val_idx), batch_size=batch_size)

        model = model_class().to(device)
        loss_fn, optimizer = get_loss_and_optimizer(model)

        for epoch in range(1, num_epochs + 1):
            train_loss = train_one_epoch(model, train_loader, loss_fn, optimizer, device)
            val_loss, val_accuracy = evaluate(model, val_loader, loss_fn, device)
            print(f'Epoch {epoch}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}')

        if dir_out_models is not None:
            path_out_model = os.path.join(dir_out_models,f"fold_{fold}.pt")
            safe_torch_save(model,path_out_model)

        models.append(model)
        results.append((val_loss, val_accuracy))

    return models, results 
"""

from torch.utils.data import DataLoader, Subset

def create_data_loaders(dataset, indices, batch_size=32):
    # Create training and validation subsets
    train_indices, val_indices = indices
    train_loader = DataLoader(Subset(dataset, train_indices), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(Subset(dataset, val_indices), batch_size=batch_size, shuffle=False)
    return train_loader, val_loader




# Getting MNIST data
transform=transforms.Compose([transforms.ToTensor(), 
                                  transforms.Normalize((0.1307,), (0.3081,))])
mnist_train = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
mnist_test = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# loss_fn = F.nll_loss # if outs = log-probabilities
loss_fn = nn.NLLLoss() # for reduction="sum"
# loss_fn = nn.CrossEntropyLoss() # outs = logits

