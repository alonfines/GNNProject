import torch
import torch.nn as nn
from pytorch_lightning import LightningModule
from torch_geometric.data import Data
from sklearn.metrics import average_precision_score
from models.spectral import SpectralGCN
from models.vn import ProbabilisticVirtualNode


class CombinedModel(LightningModule):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        num_layers: int,
        dropout: float,
        k: int,
        num_virtual_nodes: int,
        out_channels: int,
        lr: float,
        num_heads: int
    ):
        super().__init__()
        self.save_hyperparameters()
        self.spectral = SpectralGCN(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            num_layers=num_layers,
            dropout=dropout,
            k=k,
            lr=lr,
            num_heads=num_heads)
        self.vn = ProbabilisticVirtualNode(
            in_channels=hidden_channels,
            hidden_channels=hidden_channels,
            num_virtual_nodes=num_virtual_nodes,
            k=k,
            out_channels=out_channels,
            lr=lr,
            num_layers=num_layers,
            dropout=dropout
        )

        self.criterion = nn.BCEWithLogitsLoss()

    def forward(self, data):
        spec_embeddings = self.spectral(
            data.x, data.edge_index, getattr(data, "batch", None), return_embeddings=True
        )  # [num_nodes, hidden_channels]

        data_for_vn = Data(
            x=spec_embeddings,
            edge_index=data.edge_index,
            batch=getattr(data, "batch", None),
            y=data.y
        )

        vn_logits = self.vn(data_for_vn)  # [batch_size, num_classes]

        spec_logits = self.spectral(
            data.x, data.edge_index, getattr(data, "batch", None), return_embeddings=False
        )  # [batch_size, num_classes]

        return spec_logits, vn_logits

    def training_step(self, batch, batch_idx):
        spec_logits, vn_logits = self(batch)
        y = batch.y.float()

        loss_spec = self.criterion(spec_logits, y)
        loss_vn = self.criterion(vn_logits, y)
        loss = loss_spec + loss_vn

        vn_probs = torch.sigmoid(vn_logits).detach().cpu().numpy()
        y_true = y.detach().cpu().numpy()
        ap = average_precision_score(y_true, vn_probs, average="macro")

        self.log("train_loss_spec", loss_spec, on_step=True, prog_bar=False, batch_size=y.size(0))
        self.log("train_loss_vn", loss_vn, on_step=True, prog_bar=False, batch_size=y.size(0))
        self.log("train_loss", loss, on_step=True, prog_bar=True, batch_size=y.size(0))
        self.log("train_ap", ap, on_step=True, prog_bar=True, batch_size=y.size(0))
        return loss

    def validation_step(self, batch, batch_idx):
        _, vn_logits = self(batch)
        y = batch.y.float()
        val_loss = self.criterion(vn_logits, y)

        vn_probs = torch.sigmoid(vn_logits).detach().cpu().numpy()
        y_true = y.detach().cpu().numpy()
        avg_prec = average_precision_score(y_true, vn_probs, average="macro")

        self.log("val_loss", val_loss, prog_bar=True, batch_size=y.size(0))
        self.log("val_ap", avg_prec, prog_bar=True, batch_size=y.size(0))
        return val_loss
    
    def test_step(self, batch, batch_idx):
        _, vn_logits = self(batch)
        y = batch.y.float()
        test_loss = self.criterion(vn_logits, y)

        vn_probs = torch.sigmoid(vn_logits).detach().cpu().numpy()
        y_true = y.detach().cpu().numpy()
        avg_prec = average_precision_score(y_true, vn_probs, average="macro")

        self.log("test_loss", test_loss, prog_bar=True, batch_size=y.size(0))
        self.log("test_ap", avg_prec, prog_bar=True, batch_size=y.size(0))
        return test_loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

