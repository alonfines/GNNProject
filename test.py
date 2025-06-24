from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger
import pytorch_lightning as pl
import wandb
import os
import torch
from models.gcn import GCN
from models.spectral import SpectralGCN
from peptides_func_data import load_data
from models.vn import ProbabilisticVirtualNode 
from models.combined import CombinedModel
from models.combinedloss import CombinedLossModel

def test(model_type, lr, batch_size, hidden_channels,dropout,monitor_metric,k,num_heads,num_layers, num_vn, checkpoint_path):
    # Initialize WandB
    wandb.init(project="gnn-project", name=f"{model_type}-training", reinit=True)
    wandb_logger = WandbLogger(project="gnn-project", log_model=True)

    # Load data
    train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset = load_data(batch_size=batch_size)

    in_channels = test_dataset[0].x.size(1)
    out_channels = test_dataset[0].y.size(1)

    # Model and checkpoint callback
    if model_type == "GCN":
        model = GCN(in_channels=in_channels, hidden_channels=hidden_channels, out_channels=out_channels,dropout=dropout, lr=lr, num_layers=num_layers)
        monitor_metric = monitor_metric
        filename = "gcn_model_best_{epoch:02d}_{val_ap:.4f}"

    elif model_type == "SpectralGCN":
        model = SpectralGCN(in_channels=in_channels, hidden_channels=hidden_channels, out_channels=out_channels,dropout=dropout, lr=lr, num_layers=num_layers,k=k, num_heads=num_heads)
        monitor_metric = monitor_metric
        filename = "spectral_gcn_model_best_{epoch:02d}_{val_loss:.4f}"

    elif model_type == "Vn":
        model = ProbabilisticVirtualNode(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            num_virtual_nodes=num_vn,
            lr=lr,
            num_layers=num_layers,
            dropout=dropout,
            k=k
        )
        monitor_metric = "val_ap"
        filename = "vn_model_best_{epoch:02d}_{val_ap:.4f}"
    
    elif model_type == "Combined":
        model = CombinedModel(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            num_layers=num_layers,
            dropout=dropout,
            k=k,
            num_virtual_nodes=num_vn,
            lr=lr,
            num_heads=num_heads
        )
        monitor_metric = "val_ap"
        filename = "combined_model_best_{epoch:02d}_{val_ap:.4f}"
     
    elif model_type == "CombinedLoss":
        model = CombinedLossModel(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            num_layers=num_layers,
            dropout=dropout,
            k=k,
            num_virtual_nodes=num_vn,
            lr=lr,
            num_heads=num_heads
        )
        monitor_metric = "val_ap"
        filename = "combined_loss_model_best_{epoch:02d}_{val_ap:.4f}"   
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    trainer = pl.Trainer(
        max_epochs=1,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        logger=wandb_logger,
        callbacks=[
            ModelCheckpoint(
                monitor=monitor_metric,
                dirpath="checkpoints",
                filename=filename,
                save_top_k=1,
                mode="max" if "ap" in monitor_metric else "min"
            ),
            EarlyStopping(
                monitor=monitor_metric,
                patience=1,
                mode="max" if "ap" in monitor_metric else "min"
            )
        ]
    )    
    
    print(f"Testing best model from: {checkpoint_path}")
    trainer.test(model=model, dataloaders=test_loader, ckpt_path=checkpoint_path)
    

if __name__ == "__main__":
    import os
    import yaml
    import warnings
    warnings.filterwarnings("ignore", message="No positive class found in y_true")

    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    model_type = config["model_type"]
    
    model_config = config[model_type]
    lr = float(model_config["lr"])
    batch_size = model_config["batch_size"]
    hidden_channels = model_config["hidden_channels"]
    dropout = model_config["dropout"]
    k = model_config.get("k", None)  
    num_heads = model_config.get("num_heads", None)
    monitor_metric = model_config["monitor_metric"]
    num_layers = model_config["num_layers"]
    num_vn = model_config.get("num_vn", None)
    checkpoint_path = model_config.get("checkpoint_path", None)
    print(f"Training {model_type} model with the following configuration:\n")
    
    test(
        model_type=model_type,
        lr=lr,
        batch_size=batch_size,
        hidden_channels=hidden_channels,
        k=k,
        num_heads=num_heads,
        dropout=dropout,
        monitor_metric=monitor_metric,
        num_layers=num_layers,
        num_vn=num_vn,
        checkpoint_path=checkpoint_path
    )
