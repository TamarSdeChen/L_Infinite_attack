import pickle
import program_
import argparse
import torch
import torch.nn as nn
import pandas as pd
import torch.multiprocessing as tmp
from torch.utils.data import DataLoader
from metropolis_hastings import run_MH
from utils import *
import matplotlib.pyplot as plt

img_dim = 32
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0])])
train_data = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
data_loader = torch.utils.data.DataLoader(train_data, shuffle=False, batch_size=1)
img = data_loader.dataset[0]
plt.figure()
plt.imshow(img[0].permute(1, 2, 0))
# plt.show()
plt.savefig("picture1.png")

model = load_model('resnet18')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
perturbation = [(0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0)]
# try_perturb_img(model, img[0], img[1], perturbation, device)
pert_img = torch.clone(img[0])
img_shape = img[0].shape[-1]

# perturbation is tuple perturbation for 4 squares ((,,),(,,),(,,),(,,))
for num_square in range(4):
    # pert = perturbation()
    for row in range(2):
        for col in range(2):
            output = get_intarvel(row, col, img_shape)
            for c, pert in enumerate(perturbation[2 * row + 1 * col]):  # (0,1,0)
                if pert == 0:
                    pert_img[c, output[0]:output[1], output[2]:output[3]] = \
                        pert_img[c, output[0]:output[1], output[2]:output[3]] - EPSILON
                else:
                    pert_img[c, output[0]:output[1], output[2]:output[3]] = \
                        pert_img[c, output[0]:output[1], output[2]:output[3]] + EPSILON
pert_img[pert_img < 0] = 0
pert_img[pert_img > 1] = 1

plt.figure()
plt.imshow(pert_img.permute(1, 2, 0))
# plt.show()
plt.savefig("picture2.png")