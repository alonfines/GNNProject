import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from torch_geometric.nn import GCNConv, global_mean_pool
import torchmetrics
import torch.nn as nn
from sklearn.metrics import average_precision_score


class GCN(pl.LightningModule):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout,lr):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels, normalize=False))
        
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels, normalize=False))
        
        self.train_acc = torchmetrics.classification.BinaryAccuracy()
        self.val_acc = torchmetrics.classification.BinaryAccuracy()
        self.test_acc = torchmetrics.classification.BinaryAccuracy()   
        
        
        self.convs.append(GCNConv(hidden_channels, out_channels, normalize=False))
        self.dropout = dropout
        self.criterion = torch.nn.BCEWithLogitsLoss()
        self.lr = lr  
        
    def forward(self, x, edge_index, batch=None):
        # Ensure edge_index is of type Long and x is of type float
        edge_index = edge_index.long()
        x = x.float()
        
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        
        # Global pooling to get graph-level predictions
        if batch is not None:
            x = global_mean_pool(x, batch)
        return x

    def training_step(self, batch, batch_idx):
        y_hat = self(batch.x, batch.edge_index, batch.batch)
        loss = self.criterion(y_hat, batch.y)
        probs = torch.sigmoid(y_hat)
        preds = probs > 0.5
        ap = average_precision_score(batch.y.cpu().numpy(), probs.cpu().detach().numpy(), average='macro')
        acc = self.train_acc(preds, batch.y.int())
        self.log('train_loss', loss, batch_size=batch.y.size(0))
        self.log('train_acc', acc, batch_size=batch.y.size(0))
        self.log('train_ap', ap, batch_size=batch.y.size(0))
        return loss

    def validation_step(self, batch, batch_idx):
        y_hat = self(batch.x, batch.edge_index, batch.batch)
        loss = self.criterion(y_hat, batch.y)
        probs = torch.sigmoid(y_hat)
        preds = probs > 0.5
        ap = average_precision_score(batch.y.cpu().numpy(), probs.cpu().detach().numpy(), average='macro')
        acc = self.val_acc(preds, batch.y.int())
        self.log('val_loss', loss, batch_size=batch.y.size(0))
        self.log('val_acc', acc, batch_size=batch.y.size(0))
        self.log('val_ap', ap, batch_size=batch.y.size(0))
        return loss

    def test_step(self, batch, batch_idx):
        y_hat = self(batch.x, batch.edge_index, batch.batch)
        loss = self.criterion(y_hat, batch.y)
        probs = torch.sigmoid(y_hat)
        preds = probs > 0.5
        ap = average_precision_score(batch.y.cpu().numpy(), probs.cpu().detach().numpy(), average='macro')
        acc = self.test_acc(preds, batch.y.int())
        self.log('test_loss', loss, batch_size=batch.y.size(0))
        self.log('test_acc', acc, batch_size=batch.y.size(0))
        self.log('test_ap', ap, batch_size=batch.y.size(0))
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer