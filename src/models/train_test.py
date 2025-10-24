import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

device = 'cuda' if torch.cuda.is_available() else 'cpu'


'''
Initial implementation of training and testing functions for models.
This module assumes a classification task using CrossEntropyLoss.
'''

def train(model, criterion, optimizer, train_loader, n_epochs=20, mask=None, verbose=True, binary=True, use_scheduler=False):
    loss_list = []
    if use_scheduler:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
    for epc in range(n_epochs):
        for X, y in train_loader:
            optimizer.zero_grad()
            X = X.to(device)
            y = y.to(device)
            out = model.forward(X)
            loss = criterion(out, y)
            loss_list.append(loss.item())
            loss.backward()

            if mask is not None:
                with torch.no_grad():
                    for name, p in model.named_parameters():
                        if name in mask.keys():
                            p.grad *= mask[name]
            
            optimizer.step()
        if (epc%5) == 0 and verbose:
            print(f"Epoch {epc}/{n_epochs}- Loss: {loss.item()}")
        if use_scheduler:
            scheduler.step()
            
    return model, loss_list

def test(model, dataloader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for X, labels in dataloader:
            X, labels = X.to(device), labels.to(device)
            output = model(X)
            predicted = output.argmax(dim=1)
            total += labels.shape[0]
            correct += (predicted == labels).sum().item()

    return correct / total