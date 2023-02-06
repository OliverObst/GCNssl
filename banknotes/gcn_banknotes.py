#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 30 14:53:38 2023

@author: Oliver Obst
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import random
import time

class GCN(nn.Module):
    def __init__(self, num_node_features, num_classes):
        super().__init__()
        self.conv1 = GCNConv(num_node_features, 16)
        self.conv2 = GCNConv(16, num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)
    
    
class DataSet:
    def __init_(self):
        self.x = torch.zeros((0,4), dtype=torch.float32)
        self.y = torch.zeros((0,1), dtype=torch.int64)
        self.train_mask = torch.zeros(0, dtype=torch.bool)
        self.val_mask = torch.zeros(0, dtype=torch.bool)
        self.test_mask = torch.zeros(0, dtype=torch.bool)
        self.edge_index = torch.zeros((2,0), dtype=torch.int64)
        self.edge_weight = torch.zeros(0, dtype=torch.float32)
        
    def preprocess(self, data, n_train, n_val = 0):
        order = np.random.permutation(data.shape[0])
        self.x = torch.from_numpy(data[order, :-1].astype(np.float32))
        self.y = torch.from_numpy(data[order, -1].astype(np.int64))
        
        self.train_mask = torch.tensor([i < n_train for i in range(data.shape[0])])
        self.val_mask = torch.tensor([i >= n_train and i < n_train + n_val for i in range(data.shape[0])])
        self.test_mask = torch.tensor([i >= n_train + n_val for i in range(data.shape[0])])
        
    def adj_to_edge(self, adj):
        N = adj.shape[0]
        edge_index = []
        edge_weight = []
        for i in range(N):
            for j in range(N):
                if adj[i, j] != 0:
                    edge_index.append([i, j])
                    edge_weight.append(adj[i, j])
        self.edge_index = torch.tensor(edge_index).T
        self.edge_weight = torch.tensor(edge_weight)


def select_few(data, n):
    class_labels = np.unique(data.y[data.train_mask].numpy())
    class_indices = {label: [] for label in class_labels}

    for i, label in enumerate(data.y[data.train_mask].numpy()):
        class_indices[label].append(i)

    for label, indices in class_indices.items():
        class_indices[label] = random.sample(indices, n)

    selected_indices = [index for indices in class_indices.values() for index in indices]

    new_train_mask = torch.zeros(data.train_mask.shape, dtype=torch.bool) 
    new_train_mask[selected_indices] = 1
    
    return new_train_mask



def compute_adjacency_matrix(data, beta = 1.0, binary = None):
    adj = torch.exp(-beta * torch.cdist(data, data))
    
    if binary:
        if type(binary) != float and type(binary) != int:
            binary = 0.5
        adj[adj < binary] = 0
        adj[adj > 0] = 1
        
    return adj
    
# Load the dataset
d = np.loadtxt('https://archive.ics.uci.edu/ml/machine-learning-databases/00267/data_banknote_authentication.txt', delimiter=',')

RUNS = 100
results = np.zeros((RUNS,1))

start = time.time()
for run in range(RUNS):

    data = DataSet()
    data.preprocess(d, n_train = 400)

    # compute graph matrices
    adj = compute_adjacency_matrix(data.x[data.train_mask], beta=1.0, binary = 0.1)
    data.adj_to_edge(adj)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # create a new model, 4 features / node, 2 classes
    model = GCN(4,2).to(device)
    # we keep 10 examples per class labelled
    new_train_mask = select_few(data,10)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    model.train()
    for epoch in range(500):
        optimizer.zero_grad()
        out = model(data)
#        loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])    
        loss = F.nll_loss(out[new_train_mask], data.y[new_train_mask])
        loss.backward()
        optimizer.step()

    model.eval()
    pred = model(data).argmax(dim=1)
    correct = (pred[data.test_mask] == data.y[data.test_mask]).sum()
    acc = int(correct) / int(data.test_mask.sum())
    print(f'Run {run+1}/{RUNS} accuracy: {acc:.4f}')
    
    results[run,0] = acc

end = time.time()
print(f'Time for {RUNS} runs: {end - start}')    
print(f'Best result: {np.max(results):.4f}')
print(f'Mean result: {np.mean(results):.4f}')
