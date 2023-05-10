import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import dgl.nn.pytorch as dglnn
from ogb.nodeproppred import DglNodePropPredDataset, Evaluator
import pytorch_lightning as pl

from models import GAT


class GCN(pl.LightningModule):
    def __init__(
        self,
        in_feats,
        n_hidden,
        n_classes,
        n_layers,
        activation,
        dropout,
        use_linear,
        val_metric=None,
    ):
        super().__init__()
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.use_linear = use_linear
        self.val_metric = val_metric

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

    def configure_optimizers(self):
        return optim.RMSprop(self.parameters(), lr=0.002, weight_decay=0)

    def training_step(self, batch, batch_idx):
        graph, feat, labels, idx = batch[0]
        pred = self(graph, feat)
        loss = cross_entropy(pred[idx], labels[idx])
        return loss

    def validation_step(self, batch, batch_idx):
        graph, feat, labels, idx = batch[0]
        pred = self(graph, feat)
        if self.val_metric:
            self.log_dict(self.val_metric(pred, labels, idx), prog_bar=True)
        loss = cross_entropy(pred[idx], labels[idx])
        return loss


class DataModule(pl.LightningDataModule):
    def __init__(self, ogbn_name, mask_rate=0.5, use_labels=False):
        super().__init__()
        self.mask_rate = mask_rate
        self.ogbn_name = ogbn_name
        self.use_labels = use_labels
        self.graph = None
        self.labels = None
        data = DglNodePropPredDataset(name=self.ogbn_name)
        split_idx = data.get_idx_split()
        self.train_idx = split_idx["train"]
        self.val_idx = split_idx["valid"]
        self.test_idx = split_idx["test"]
        self.n_classes = int(data.num_classes)

    def setup(self, stage):
        data = DglNodePropPredDataset(name=self.ogbn_name)
        graph, labels = data[0]
        srcs, dsts = graph.all_edges()
        graph.add_edges(dsts, srcs)
        graph = graph.remove_self_loop().add_self_loop()
        print(f"Total edges after adding self-loop {graph.number_of_edges()}")
        self.graph = graph
        self.labels = labels

    def train_dataloader(self):
        feat = self.graph.ndata["feat"]
        mask = torch.rand(self.train_idx.shape) < self.mask_rate
        train_labels_idx = self.train_idx[~mask]
        train_pred_idx = self.train_idx[mask]
        if self.use_labels:
            feat = add_labels(feat, self.labels, train_labels_idx, self.n_classes)
        return torch.utils.data.DataLoader(
            [(self.graph, feat, self.labels, train_pred_idx)],
            collate_fn=lambda data: data,
        )

    def val_dataloader(self):
        feat = self.graph.ndata["feat"]
        if self.use_labels:
            feat = add_labels(feat, self.labels, self.train_idx, self.n_classes)
        return torch.utils.data.DataLoader(
            [(self.graph, feat, self.labels, self.val_idx)],
            collate_fn=lambda data: data,
        )


def add_labels(feat, labels, idx, n_classes):
    onehot = torch.zeros([feat.shape[0], n_classes])
    onehot[idx, labels[idx, 0]] = 1
    return torch.cat([feat, onehot], dim=-1)


def cross_entropy(x, labels):
    epsilon = 1 - math.log(2)
    y = F.cross_entropy(x, labels[:, 0], reduction="none")
    y = torch.log(epsilon + y) - math.log(epsilon)
    return torch.mean(y)


def compute_acc(pred, labels, evaluator):
    return evaluator.eval(
        {"y_pred": pred.argmax(dim=-1, keepdim=True), "y_true": labels}
    )["acc"]


def main():
    n_epochs = 100
    n_layers = 3
    n_hidden = 256
    n_heads = 3
    dropout = 0.75
    attn_drop = 0.05
    norm = "none" # "both"

    datamodule = DataModule("ogbn-arxiv", use_labels=True)
    data = DglNodePropPredDataset("ogbn-arxiv")
    graph, labels = data[0]
    in_feats = graph.ndata["feat"].shape[1]
    n_classes = (labels.max() + 1).item()

    evaluator = Evaluator(name="ogbn-arxiv")
    accuracy = lambda pred, labels, idx: {
        f"{stage}_acc": compute_acc(
            pred[getattr(datamodule, f"{stage}_idx")],
            labels[getattr(datamodule, f"{stage}_idx")],
            evaluator,
        )
        for stage in ["train", "val", "test"]
    }

    """model = GCN(
        in_feats=in_feats + n_classes,
        n_classes=n_classes,
        n_hidden=n_hidden,
        n_layers=n_layers,
        activation=F.relu,
        dropout=dropout,
        use_linear=False,
        val_metric=accuracy,
    )"""
    model = GAT(
        in_feats + n_classes,
        n_classes,
        n_hidden=n_hidden,
        n_layers=n_layers,
        n_heads=n_heads,
        activation=F.relu,
        dropout=dropout,
        attn_drop=attn_drop,
        norm=norm,
        val_metric=accuracy,
    )
    print([np.prod(p.size()) for p in model.parameters() if p.requires_grad])
    print(sum([np.prod(p.size()) for p in model.parameters() if p.requires_grad]))

    trainer = pl.Trainer(
        default_root_dir="../data",
        accelerator="auto",
        max_epochs=n_epochs,
        log_every_n_steps=1,
    )
    trainer.fit(model, datamodule=datamodule)


if __name__ == "__main__":
    main()
