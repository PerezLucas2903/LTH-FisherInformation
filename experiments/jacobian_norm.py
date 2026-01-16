import random
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader

repo_root = Path(__file__).resolve().parents[1]
src_path = repo_root / "src"
import sys
sys.path.insert(0, str(src_path))


from fisher_information.fim import FisherInformationMatrix
from models.image_classification_models import ConvModelMNIST
from models.train_test import *
from prunning_methods.LTH import *

device = "cuda" if torch.cuda.is_available() else "cpu"


mnist_train = torchvision.datasets.MNIST(root='../data', train=True, download=True, transform=torchvision.transforms.ToTensor())
mnist_test = torchvision.datasets.MNIST(root='../data', train=False, download=True, transform=torchvision.transforms.ToTensor())

mnist_train_loader = DataLoader(mnist_train, batch_size = 256, shuffle=True)
mnist_train_fim_loader = DataLoader(mnist_train, batch_size = 1, shuffle=True) 
mnist_test_loader = DataLoader(mnist_test, batch_size = 20, shuffle=True)


fim_args = {"complete_fim": True, 
            "layers":  None, 
            "mask":  None, 
            "sampling_type":  'complete', 
            "sampling_frequency":  None
            }


LTH_args = {"model": ConvModelMNIST().to(device), 
            "criterion": nn.CrossEntropyLoss(), 
            "train_loader": mnist_train_loader, 
            "test_loader": mnist_test_loader, 
            "fim_loader": mnist_train_fim_loader, 
            "fim_args": fim_args, 
            "lr" : 1e-3,
            "n_iterations":10, 
            "n_epochs":30, 
            "prunning_percentage":0.1, 
            "no_prunning_layers":None, 
            "verbose":True,
            "print_freq":10, 
            "use_scheduler":False, 
            "save_path":None,
            "calculate_fim": True,
            "calculate_jacobian": True,
            'save_model': True
            }
           

output_dict = train_LTH(**LTH_args)

torch.save(output_dict, 'jacobian_norm_convmodelmnist.pt')


parameters = []
for key, value in output_dict["model_list"][-2].items():
    parameters.append(value.to('cpu').ravel())
param_ravel = torch.cat(parameters)


jacob = []
for key, value in output_dict["jacobian_list"][-2][0].items():
    jacob.append(value.to('cpu').ravel())

jacob_ravel = torch.cat(jacob)


mask_ravel = []
for key, value in output_dict["mask_list"][-1].items():
    print(key, value.numel())
    mask_ravel.append(value.to('cpu').ravel())
    if key == "net.0.weight":
        mask_ravel.append(torch.ones(4))
    elif key == "net.3.weight":
        mask_ravel.append(torch.ones(16))
    elif key == "net.7.weight":
        mask_ravel.append(torch.ones(10))

mask_ravel = torch.cat(mask_ravel)

data = {'Parameters': np.abs(param_ravel.numpy()), 'Jacobian': jacob_ravel.numpy(), 'Mask': mask_ravel.numpy()}
df = pd.DataFrame(data)


scatter_1 =sns.scatterplot(data=data, x='Parameters', y='Jacobian', hue='Mask', s=10)
scatter_2 = sns.scatterplot(data=df[df['Mask']==0], x='Parameters', y='Jacobian', hue='Mask', s=10)
scatter_3 = sns.scatterplot(data=df[df['Parameters']!=0], x='Parameters', y='Jacobian', hue='Mask', s=10)

scatter_1.savefig('jacobian_vs_parameters_convmodelmnist.pdf')
scatter_2.savefig('jacobian_vs_parameters_prunned_convmodelmnist.pdf')
scatter_3.savefig('jacobian_vs_parameters_nonzero_convmodelmnist.pdf')