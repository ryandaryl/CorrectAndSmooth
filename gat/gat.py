import math
import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import dgl.nn.pytorch as dglnn
from ogb.nodeproppred import DglNodePropPredDataset, Evaluator
import pytorch_lightning as pl

n_classes = None
epsilon = 1 - math.log(2)


class GCN(pl.LightningModule):
    def __init__(self, in_feats, n_hidden, n_classes, n_layers, activation,
            dropout, use_linear, use_labels, train_idx, val_idx, test_idx, evaluator):
        super().__init__()
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.use_linear = use_linear
        self.use_labels = use_labels
        self.train_idx = train_idx
        self.val_idx = val_idx
        self.test_idx = test_idx
        self.evaluator = evaluator

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
        graph, labels = batch[0]
        feat = graph.ndata["feat"]
        if self.use_labels:
            mask_rate = 0.5
            mask = torch.rand(self.train_idx.shape) < mask_rate
            train_labels_idx = self.train_idx[mask]
            train_pred_idx = self.train_idx[~mask]
            feat = add_labels(feat, labels, train_labels_idx)
        else:
            mask_rate = 0.5
            mask = torch.rand(self.train_idx.shape) < mask_rate
            train_pred_idx = self.train_idx[mask]
        pred = self(graph, feat)
        loss = cross_entropy(
            pred[train_pred_idx],
            labels[train_pred_idx])
        return loss

    def validation_step(self, batch, batch_idx):
        graph, labels = batch[0]
        feat = graph.ndata["feat"]
        if self.use_labels:
            feat = add_labels(feat, labels, self.train_idx)
        pred = self(graph, feat)
        self.log_dict({f'{stage}_acc': compute_acc(
            pred[getattr(self, f'{stage}_idx')],
            labels[getattr(self, f'{stage}_idx')],
            self.evaluator
        ) for stage in ['train', 'val', 'test']})
        loss = cross_entropy(
            pred[self.val_idx],
            labels[self.val_idx])
        return loss


def add_labels(feat, labels, idx):
    onehot = torch.zeros([feat.shape[0], n_classes])
    onehot[idx, labels[idx, 0]] = 1
    return torch.cat([feat, onehot], dim=-1)


def cross_entropy(x, labels):
    y = F.cross_entropy(x, labels[:, 0], reduction="none")
    y = torch.log(epsilon + y) - math.log(epsilon)
    return torch.mean(y)


def compute_acc(pred, labels, evaluator):
    return evaluator.eval({"y_pred": pred.argmax(dim=-1, keepdim=True), "y_true": labels})["acc"]


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else torch.device("cpu"))
    n_epochs = 100
    n_layers = 3
    use_labels = False
    n_hidden = 256
    dropout = 0.75

    data = DglNodePropPredDataset(name="ogbn-arxiv")
    evaluator = Evaluator(name="ogbn-arxiv")

    split_idx = data.get_idx_split()
    train_idx, val_idx, test_idx = [split_idx[k].to(device) for k in ["train", "valid", "test"]]
    graph, labels = [tensor_data.to(device) for tensor_data in data[0]]
    dataloader = torch.utils.data.DataLoader([(graph, labels)], collate_fn=lambda data: data)

    srcs, dsts = graph.all_edges()
    graph.add_edges(dsts, srcs)
    graph = graph.remove_self_loop().add_self_loop()
    print(f"Total edges after adding self-loop {graph.number_of_edges()}")

    in_feats = graph.ndata["feat"].shape[1]
    n_classes = (labels.max() + 1).item()

    model = GCN(
        in_feats=in_feats,
        n_classes=n_classes,
        n_hidden=n_hidden,
        n_layers=n_layers,
        activation=F.relu,
        dropout=dropout,
        use_linear=False,
        use_labels=use_labels,
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        evaluator=evaluator,
    )
    print([np.prod(p.size()) for p in model.parameters() if p.requires_grad])
    print(sum([np.prod(p.size()) for p in model.parameters() if p.requires_grad]))

    class PrintMetrics(pl.callbacks.Callback):

        def on_epoch_end(self, trainer, *args):
            print(trainer.logged_metrics)

    trainer = pl.Trainer(
        default_root_dir='../data',
        callbacks=[PrintMetrics()],
        accelerator="auto",
        max_epochs=50,
        #enable_progress_bar=False,
        log_every_n_steps=1,
    )
    trainer.fit(model, dataloader, dataloader)

    initial_learning_rate = model.optimizer.param_group["lr"]

    best_val_acc, best_test_acc, best_val_loss = 0, 0, float("inf")

    """for epoch in range(1, n_epochs + 1):
        start_time = datetime.datetime.now()

        if epoch <= 50:
            for param_group in model.optimizer.param_groups:
                param_group["lr"] = initial_learning_rate * epoch / 50

        loss, pred = train(model, graph, labels, train_idx, optimizer, use_labels)
        acc = compute_acc(pred[train_idx], labels[train_idx], evaluator)

        train_acc, val_acc, test_acc, train_loss, val_loss, test_loss = evaluate(
            model, graph, labels, train_idx, val_idx, test_idx, use_labels, evaluator, device)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            best_test_acc = test_acc

        print(
            f"{epoch=}\n"
            f"Loss: {loss.item():.4f}, Acc: {acc:.4f}\n"
            f"Train/Val/Test loss: {train_loss:.4f}/{val_loss:.4f}/{test_loss:.4f}\n"
            f"Train/Val/Test/Best val/Best test acc: {train_acc:.4f}/{val_acc:.4f}/{test_acc:.4f}/{best_val_acc:.4f}/{best_test_acc:.4f}"
        )
        print(datetime.datetime.now() - start_time)"""


if __name__ == "__main__":
    main()
