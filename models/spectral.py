import torch
import torch.nn.functional as F
import torch.nn as nn
import pytorch_lightning as pl
import numpy as np
from sklearn.metrics import average_precision_score
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.utils import get_laplacian, to_dense_adj
import torchmetrics
import warnings

class SpectralFilterMLP(nn.Module):
    def __init__(self, k, hidden_dim=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        self.k = k

    def forward(self, eigenvalues):
        lambdas = eigenvalues[:self.k].unsqueeze(1)  # (k, 1)
        weights = self.net(lambdas).squeeze(1)       # (k,)
        return torch.diag(weights)                   # (k, k)

class FusionBlock(nn.Module):
    def __init__(self, hidden_channels, num_heads):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_channels)
        self.attn = nn.MultiheadAttention(hidden_channels, num_heads=num_heads, batch_first=True)
        self.weights = nn.Parameter(torch.ones(2) / 2)

    def forward(self, x_spatial, x_spectral):
        x_spatial = self.norm(x_spatial)
        x_spectral = self.norm(x_spectral)
        x_spatial_attn, _ = self.attn(x_spatial.unsqueeze(0), x_spatial.unsqueeze(0), x_spatial.unsqueeze(0))
        x_spectral_attn, _ = self.attn(x_spectral.unsqueeze(0), x_spectral.unsqueeze(0), x_spectral.unsqueeze(0))
        x_fused = self.weights[0] * x_spatial_attn + self.weights[1] * x_spectral_attn
        return x_fused.squeeze(0)

class SpectralGCN(pl.LightningModule):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout, k, lr,num_heads):
        super().__init__()
        self.save_hyperparameters()
        self.k = k
        self.lr = lr
        self.dropout = dropout
        self.hidden_channels = hidden_channels

        # GCN layers
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
        self.convs.append(GCNConv(hidden_channels, hidden_channels))

        # Spectral modules
        self.spectral_filter = SpectralFilterMLP(k=k)
        self.spectral_proj = nn.Linear(k, hidden_channels)

        # Fusion and output
        self.fusion = FusionBlock(hidden_channels,num_heads)
        self.output_proj = nn.Linear(hidden_channels, out_channels)

        # Loss and metrics
        self.criterion = nn.BCEWithLogitsLoss()
        self.metrics = nn.ModuleDict({
            "train_acc": torchmetrics.classification.MultilabelAccuracy(num_labels=out_channels, average='macro'),
            "val_acc": torchmetrics.classification.MultilabelAccuracy(num_labels=out_channels, average='macro'),
            "test_acc": torchmetrics.classification.MultilabelAccuracy(num_labels=out_channels, average='macro'),
        })

    def compute_laplacian_basis(self, edge_index, num_nodes):
        try:
            edge_index, edge_weight = get_laplacian(edge_index, normalization='sym')
            L = to_dense_adj(edge_index, edge_attr=edge_weight, max_num_nodes=num_nodes)[0]
            L = L + torch.eye(L.size(0), device=L.device) * 1e-5
            eigenvalues, eigenvectors = torch.linalg.eigh(L)
            eigenvalues = torch.abs(eigenvalues)
            eigenvalues = eigenvalues / eigenvalues.max()
            return eigenvalues[:self.k], eigenvectors[:, :self.k]
        except RuntimeError as e:
            print(f"Laplacian EVD failed: {e}")
            return None, None

    def apply_spectral_filter(self, x, eigenvalues, eigenvectors):
        if eigenvalues is None or eigenvectors is None:
            return torch.zeros((x.size(0), self.hidden_channels), device=x.device)
        x = x[:, :self.k] if x.size(1) > self.k else F.pad(x, (0, self.k - x.size(1)))
        x_hat = eigenvectors.T @ x
        x_hat_filtered = self.spectral_filter(eigenvalues) @ x_hat
        x_filtered = eigenvectors @ x_hat_filtered
        return self.spectral_proj(x_filtered)

    def forward(self, X, A, batch=None, return_embeddings=False):
        X = X.float()
        A = A.long()
        num_nodes = X.size(0)
        lambdas, V = self.compute_laplacian_basis(A, num_nodes)
        for conv in self.convs[:-1]:
            x_spatial = conv(X, A)
            x_spectral = self.apply_spectral_filter(X, lambdas, V)
            X = self.fusion(x_spatial, x_spectral)
            X = F.relu(X)
            X = F.dropout(X, p=self.dropout, training=self.training)
        x_spatial = self.convs[-1](X, A)
        x_spectral = self.apply_spectral_filter(X, lambdas, V)
        X = self.fusion(x_spatial, x_spectral)
        if return_embeddings:
            return X
        X = self.output_proj(X)
        if batch is not None:
            X = global_mean_pool(X, batch)
        return X

    def on_train_start(self):
        for metric in self.metrics.values():
            metric.to(self.device)

    def step(self, batch, stage):
        y_hat = self(batch.x, batch.edge_index, batch.batch)
        loss = self.criterion(y_hat, batch.y)
        preds = (torch.sigmoid(y_hat) > 0.5).int()
        acc = self.metrics[f"{stage}_acc"](preds, batch.y.int())

        # Compute AP using sklearn
        y_true = batch.y.detach().cpu().numpy()
        y_prob = torch.sigmoid(y_hat).detach().cpu().numpy()
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="No positive class found in y_true")
            try:
                ap = average_precision_score(y_true, y_prob, average="macro")
            except ValueError:
                ap = 0.0

        self.log(f"{stage}_loss", loss, batch_size=batch.y.size(0))
        self.log(f"{stage}_acc", acc, batch_size=batch.y.size(0))
        self.log(f"{stage}_ap", ap, batch_size=batch.y.size(0))

        return loss

    def training_step(self, batch, batch_idx):
        return self.step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self.step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self.step(batch, "test")

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
