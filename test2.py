import os
import os.path as osp
import argparse
import glob
import torch
import torch.nn as nn

import numpy as np
from PIL import Image
from utils import *
from networks.network import get_model

from torchvision import datasets, models, transforms
import torch.optim as optim
from torch.utils.data import DataLoader

from tqdm import tqdm

class DeepPreset(object):
    def __init__(self, ckpt, ckpt_2=""):
        ckpt = torch.load(ckpt)

        # Load model
        self.G = get_model(ckpt['opts'].g_net)(ckpt['opts']).cuda()
        
        self.G.load_state_dict(ckpt['G'])
            
        num_features = self.G.llayer_2[0].in_features
        # print(len(self.G.parameters()))
        for param in self.G.parameters():
          param.requires_grad = False

        self.G.llayer_2 = nn.Sequential(
          nn.Linear(num_features, 69),
          nn.Tanh(),
          nn.Linear(69, 4),
          nn.LogSoftmax(dim=1)
        )

        if (ckpt_2 != ""):
          ckpt_2 = torch.load(ckpt_2)
          self.G.load_state_dict(ckpt_2)

        self.loss_func = nn.NLLLoss()



def train_model():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default="/content/dp_woPPL.pth.tar", help='Checkpoint path')
    parser.add_argument("--ckpt_2", type=str, default="", help='Checkpoint path')
    parser.add_argument("--dataset", type=str, default="")
    parser.add_argument("--num_epochs", type=int, default="10")
    parser.add_argument("--batch_size", type=int, default="8")
    args = parser.parse_args()

    train_transform = transforms.Compose([
        transforms.RandomRotation(degrees=15),
        transforms.RandomHorizontalFlip(),
        transforms.Resize((352, 352)),
        transforms.CenterCrop((352, 352)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])

    train_data = datasets.ImageFolder(args.dataset, transform=train_transform)
    print(train_data.class_to_idx)
    train_data_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    train_data_size = len(train_data)

    deep_preset = DeepPreset(args.ckpt, args.ckpt_2)
    
    model = deep_preset.G
    loss_criterion = deep_preset.loss_func

    optimizer = optim.Adam(model.parameters(), lr=0.005)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10,20], gamma=0.1)
    epochs = args.num_epochs
    model.to("cuda")
    for epoch in range(epochs):
        print("Epoch: {}/{}".format(epoch+1, epochs))
        model.train()
        train_loss = 0.0
        train_acc = 0.0

        for i, (inputs, labels) in tqdm(enumerate(train_data_loader)):
            inputs = inputs.to("cuda")
            labels = labels.to("cuda")
            
            # Clean existing gradients
            optimizer.zero_grad()
            
            # Forward pass - compute outputs on input data using the model
            _, outputs, _ = model.stylize(inputs, inputs, None, preset_only=True)
            
            # Compute loss
            loss = loss_criterion(outputs, labels)
            
            # Backpropagate the gradients
            loss.backward()
            
            # Update the parameters
            optimizer.step()
            scheduler.step()
            
            # Compute the total loss for the batch and add it to train_loss
            train_loss += loss.item() * inputs.size(0)
            
            # Compute the accuracy
            ret, predictions = torch.max(outputs.data, 1)
            correct_counts = predictions.eq(labels.data.view_as(predictions))
            
            # Convert correct_counts to float and then compute the mean
            acc = torch.mean(correct_counts.type(torch.FloatTensor))
            
            # Compute total accuracy in the whole batch and add to train_acc
            train_acc += acc.item() * inputs.size(0)

        avg_train_loss = train_loss/train_data_size 
        avg_train_acc = train_acc/train_data_size

        print("predictions", predictions)
        print("labels", labels)
        print("Epoch : {:03d}, Training: Loss - {:.4f}, Accuracy - {:.4f}%".format(epoch+1, avg_train_loss, avg_train_acc*100))
        
        # Save if the model has best accuracy till now
        torch.save(model.state_dict(), 'model_'+str(epoch)+'.pt')
                
        


def main():
    # For parsing commandline arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default="/content/dp_woPPL.pth.tar", help='Checkpoint path')
    parser.add_argument("--ckpt_2", type=str, default="", help='Checkpoint path')
    parser.add_argument("--dataset", type=str, default="")
    parser.add_argument("--num_epochs", type=int, default="10")
    parser.add_argument("--batch_size", type=int, default="8")
    args = parser.parse_args()

    deep_preset = DeepPreset(args.ckpt, args.ckpt_2)
    
    model = deep_preset.G

    # print(model)
    for param in model.parameters():
      if (param.requires_grad):
        print(param.shape)

# main()
train_model()
