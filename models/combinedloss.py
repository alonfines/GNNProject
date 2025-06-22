import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from sklearn.metrics import average_precision_score
import numpy as np
from models.spectral import SpectralGCN
from models.vn import ProbabilisticVirtualNode

class CombinedLossModel(LightningModule):
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
            num_heads=num_heads
        )
        
        self.vn = ProbabilisticVirtualNode(
            in_channels=in_channels,
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
        spec_logits = self.spectral(data.x, data.edge_index, getattr(data, "batch", None))
        vn_logits = self.vn(data)
        return spec_logits, vn_logits

    def training_step(self, batch, batch_idx):
        spec_logits, vn_logits = self(batch)
        y = batch.y.float()
        y_true = y.detach().cpu().numpy()
        loss = 0

        loss_spec = self.criterion(spec_logits, y)
        loss += loss_spec
        spec_probs = torch.sigmoid(spec_logits).detach().cpu().numpy()
        spec_avg_prec = average_precision_score(y_true, spec_probs, average='macro') if np.any(y_true) else 0.0
        self.log("train_loss_spec", loss_spec, on_step=True, batch_size=y.size(0))
        self.log("train_spectral_avg_precision", spec_avg_prec, on_epoch=True, prog_bar=True, batch_size=y.size(0))

        
        loss_vn = self.criterion(vn_logits, y)
        loss += loss_vn
        vn_probs = torch.sigmoid(vn_logits).detach().cpu().numpy()
        vn_avg_prec = average_precision_score(y_true, vn_probs, average='macro') if np.any(y_true) else 0.0
        self.log("train_loss_vn", loss_vn, on_step=True, batch_size=y.size(0))
        self.log("train_vp", vn_avg_prec, on_epoch=True, prog_bar=True, batch_size=y.size(0))

        self.log("train_loss", loss, on_step=True, batch_size=y.size(0))
        return loss

    def validation_step(self, batch, batch_idx):
        spec_logits, vn_logits = self(batch)
        y = batch.y.float()
        y_true = y.detach().cpu().numpy()

        loss_spec = self.criterion(spec_logits, y)
        loss_vn = self.criterion(vn_logits, y)
        val_loss = loss_spec + loss_vn

        spec_probs = torch.sigmoid(spec_logits).detach().cpu().numpy()
        vn_probs = torch.sigmoid(vn_logits).detach().cpu().numpy()
        spec_ap = average_precision_score(y_true, spec_probs, average='macro') if np.any(y_true) else 0.0
        vn_ap = average_precision_score(y_true, vn_probs, average='macro') if np.any(y_true) else 0.0
        val_ap = (spec_ap + vn_ap) / 2.0
        
        self.log("val_loss_spec", loss_spec, batch_size=y.size(0))
        self.log("val_loss_vn", loss_vn, batch_size=y.size(0))
        self.log("val_spectral_ap", spec_ap, on_epoch=True, prog_bar=False, batch_size=y.size(0))
        self.log("val_vn_ap", vn_ap, on_epoch=True, prog_bar=True, batch_size=y.size(0))
        self.log("val_loss", val_loss, batch_size=y.size(0))
        self.log("val_ap", val_ap, on_epoch=True, prog_bar=True, batch_size=y.size(0))  
        return val_loss
    
    def test_step(self, batch, batch_idx):
        spec_logits, vn_logits = self(batch)
        y = batch.y.float()
        y_true = y.detach().cpu().numpy()

        loss_spec = self.criterion(spec_logits, y)
        loss_vn = self.criterion(vn_logits, y)
        test_loss = loss_spec + loss_vn

        spec_probs = torch.sigmoid(spec_logits).detach().cpu().numpy()
        vn_probs = torch.sigmoid(vn_logits).detach().cpu().numpy()
        spec_ap = average_precision_score(y_true, spec_probs, average='macro') if np.any(y_true) else 0.0
        vn_ap = average_precision_score(y_true, vn_probs, average='macro') if np.any(y_true) else 0.0
        test_ap = (spec_ap + vn_ap) / 2.0
        
        self.log("test_loss_spec", loss_spec, batch_size=y.size(0))
        self.log("test_loss_vn", loss_vn, batch_size=y.size(0))
        self.log("test_spectral_ap", spec_ap, on_epoch=True, prog_bar=False, batch_size=y.size(0))
        self.log("test_vn_ap", vn_ap, on_epoch=True, prog_bar=True, batch_size=y.size(0))
        self.log("test_loss", test_loss, batch_size=y.size(0))
        self.log("test_ap", test_ap, on_epoch=True, prog_bar=True, batch_size=y.size(0))

        

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

