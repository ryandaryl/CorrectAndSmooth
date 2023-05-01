import math
import time
import os
import shutil
import datetime

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import dgl.nn.pytorch as dglnn
from matplotlib import pyplot as plt
from matplotlib.ticker import AutoMinorLocator, MultipleLocator
from ogb.nodeproppred import DglNodePropPredDataset, Evaluator

device = None
in_feats, n_classes, subgraph_size = None, None, None
epsilon = 1 - math.log(2)


class GCN(nn.Module):
    def __init__(self, in_feats, n_hidden, n_classes, n_layers, activation, dropout, use_linear):
        super().__init__()
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.use_linear = use_linear

        self.convs = nn.ModuleList()
        if use_linear:
            self.linear = nn.ModuleList()
        self.bns = nn.ModuleList()

        for i in range(n_layers):
            in_hidden = n_hidden if i > 0 else in_feats
            out_hidden = n_hidden if i < n_layers - 1 else n_classes
            bias = i == n_layers - 1

            self.convs.append(dglnn.GraphConv(in_hidden, out_hidden, "both", bias=bias))
            if use_linear:
                self.linear.append(nn.Linear(in_hidden, out_hidden, bias=False))
            if i < n_layers - 1:
                self.bns.append(nn.BatchNorm1d(out_hidden))

        self.dropout0 = nn.Dropout(min(0.1, dropout))
        self.dropout = nn.Dropout(dropout)
        self.activation = activation

    def forward(self, graph, feat):
        h = feat
        h = self.dropout0(h)

        for i in range(self.n_layers):
            conv = self.convs[i](graph, h)

            if self.use_linear:
                linear = self.linear[i](h)
                h = conv + linear
            else:
                h = conv

            if i < self.n_layers - 1:
                h = self.bns[i](h)
                h = self.activation(h)
                h = self.dropout(h)

        return h


def gen_model(use_norm, n_hidden, n_layers, dropout):
    norm = "both" if use_norm else "none"
    model = GCN(
        in_feats=in_feats,
        n_classes=n_classes,
        n_hidden=n_hidden,
        n_layers=n_layers,
        activation=F.relu,
        dropout=dropout,
        use_linear=False,
    )
    return model


def cross_entropy(x, labels):
    y = F.cross_entropy(x, labels[:, 0], reduction="none")
    y = th.log(epsilon + y) - math.log(epsilon)
    return th.mean(y)


def compute_acc(pred, labels, evaluator):
    return evaluator.eval({"y_pred": pred.argmax(dim=-1, keepdim=True), "y_true": labels})["acc"]


def add_labels(feat, labels, idx):
    onehot = th.zeros([feat.shape[0], n_classes]).to(device)
    onehot[idx, labels[idx, 0]] = 1
    return th.cat([feat, onehot], dim=-1)


def adjust_learning_rate(optimizer, lr, epoch):
    if epoch <= 50:
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr * epoch / 50


def train(model, graph, labels, train_idx, optimizer, use_labels):
    model.train()

    feat = graph.ndata["feat"]

    if use_labels:
        mask_rate = 0.5
        mask = th.rand(train_idx.shape) < mask_rate

        train_labels_idx = train_idx[mask]
        train_pred_idx = train_idx[~mask]

        feat = add_labels(feat, labels, train_labels_idx)
    else:
        mask_rate = 0.5
        mask = th.rand(train_idx.shape) < mask_rate

        train_pred_idx = train_idx[mask]
    optimizer.zero_grad()
    pred = model(graph.subgraph(range(subgraph_size)) if subgraph_size else graph, feat[:subgraph_size, :in_feats])
    loss = cross_entropy(
        pred[train_pred_idx.clip(0, subgraph_size - 1) if subgraph_size else train_pred_idx][:subgraph_size],
        labels[train_pred_idx][:subgraph_size])
    loss.backward()
    optimizer.step()

    return loss, pred


@th.no_grad()
def evaluate(model, graph, labels, train_idx, val_idx, test_idx, use_labels, evaluator):
    model.eval()

    feat = graph.ndata["feat"]

    if use_labels:
        feat = add_labels(feat, labels, train_idx)
    pred = model(graph, feat[:, :in_feats])
    train_loss = cross_entropy(pred[train_idx], labels[train_idx])
    val_loss = cross_entropy(pred[val_idx], labels[val_idx])
    test_loss = cross_entropy(pred[test_idx], labels[test_idx])

    return (
        compute_acc(pred[train_idx], labels[train_idx], evaluator),
        compute_acc(pred[val_idx], labels[val_idx], evaluator),
        compute_acc(pred[test_idx], labels[test_idx], evaluator),
        train_loss,
        val_loss,
        test_loss,
        pred
    )


def count_parameters(use_norm, n_hidden, n_layers, dropout):
    model = gen_model(use_norm, n_hidden, n_layers, dropout)
    print([np.prod(p.size()) for p in model.parameters() if p.requires_grad])
    return sum([np.prod(p.size()) for p in model.parameters() if p.requires_grad])


def prepare_folder(name, model):
    model_dir = f'models/{name}'
   
    if os.path.exists(model_dir):
        shutil.rmtree(model_dir)
    os.makedirs(model_dir)
    with open(f'{model_dir}/metadata', 'w') as f:
        f.write(f'# of params: {sum(p.numel() for p in model.parameters())}\n')
    return model_dir


def main():
    global device, in_feats, subgraph_size, n_classes, epsilon

    device = th.device("cuda:0" if th.cuda.is_available() else th.device("cpu"))
    n_epochs = 100
    lr = 0.002
    n_layers = 3
    use_labels = False
    use_norm = True
    weight_decay = 0
    n_hidden = 256
    dropout = 0.75

    # load data
    data = DglNodePropPredDataset(name="ogbn-arxiv")
    evaluator = Evaluator(name="ogbn-arxiv")

    splitted_idx = data.get_idx_split()
    train_idx, val_idx, test_idx = splitted_idx["train"], splitted_idx["valid"], splitted_idx["test"]
    graph, labels = data[0]

    srcs, dsts = graph.all_edges()
    graph.add_edges(dsts, srcs)
    graph = graph.remove_self_loop().add_self_loop()
    print(f"Total edges after adding self-loop {graph.number_of_edges()}")

    in_feats = graph.ndata["feat"].shape[1]
    n_classes = (labels.max() + 1).item()
    subgraph_size = None
    # graph.create_format_()

    train_idx = train_idx.to(device)
    val_idx = val_idx.to(device)
    test_idx = test_idx.to(device)
    labels = labels.to(device)
    graph = graph.to(device)

    # run
    val_accs = []
    test_accs = []
    model_dir = f'../models/arxiv_gat'
       
    #if os.path.exists(model_dir):
    #    shutil.rmtree(model_dir)
    #os.makedirs(model_dir)
    with open(f'{model_dir}/metadata', 'w') as f:
        f.write(f'# of params: {sum(p.numel() for p in gen_model(use_norm, n_hidden, n_layers, dropout).parameters())}\n')

    model = gen_model(use_norm, n_hidden, n_layers, dropout)
    print(count_parameters(use_norm, n_hidden, n_layers, dropout))
    model = model.to(device)

    optimizer = optim.RMSprop(model.parameters(), lr=lr, weight_decay=weight_decay)

    # training loop
    total_time = 0
    best_val_acc, best_test_acc, best_val_loss = 0, 0, float("inf")
    best_out = None

    accs, train_accs, val_accs, test_accs = [], [], [], []
    losses, train_losses, val_losses, test_losses = [], [], [], []

    for epoch in range(1, n_epochs + 1):
        start_time = datetime.datetime.now()
        tic = time.time()

        adjust_learning_rate(optimizer, lr, epoch)

        loss, pred = train(model, graph, labels, train_idx, optimizer, use_labels)
        acc = compute_acc(pred[train_idx.clip(0, subgraph_size - 1) if subgraph_size else train_idx], labels[train_idx], evaluator)

        train_acc, val_acc, test_acc, train_loss, val_loss, test_loss, out = evaluate(
            model, graph, labels, train_idx, val_idx, test_idx, use_labels, evaluator
        )

        toc = time.time()
        total_time += toc - tic

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            best_test_acc = test_acc
            best_out = out

        print(
            f"{epoch=}\n"
            f"Loss: {loss.item():.4f}, Acc: {acc:.4f}\n"
            f"Train/Val/Test loss: {train_loss:.4f}/{val_loss:.4f}/{test_loss:.4f}\n"
            f"Train/Val/Test/Best val/Best test acc: {train_acc:.4f}/{val_acc:.4f}/{test_acc:.4f}/{best_val_acc:.4f}/{best_test_acc:.4f}"
        )

        for l, e in zip(
            [accs, train_accs, val_accs, test_accs, losses, train_losses, val_losses, test_losses],
            [acc, train_acc, val_acc, test_acc, loss.item(), train_loss, val_loss, test_loss],
        ):
            l.append(e)
        print(datetime.datetime.now() - start_time)

    print("*" * 50)
    print(f"Average epoch time: {total_time / n_epochs}, Test acc: {best_test_acc}")
    val_acc, test_acc, out = best_val_acc, best_test_acc, best_out

    val_accs.append(val_acc)
    test_accs.append(test_acc)
    th.save(F.softmax(out, dim=1), f'{model_dir}/{i-1}.pt')

    print(f"Runned {args.n_runs} times")
    print("Val Accs:", val_accs)
    print("Test Accs:", test_accs)
    print(f"Average val accuracy: {np.mean(val_accs)} ± {np.std(val_accs)}")
    print(f"Average test accuracy: {np.mean(test_accs)} ± {np.std(test_accs)}")
    print(f"Number of params: {count_parameters(use_norm, n_hidden, n_layers, dropout)}")


if __name__ == "__main__":
    main()
