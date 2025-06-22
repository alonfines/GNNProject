from train import train
import os
import yaml
import warnings
warnings.filterwarnings("ignore", message="No positive class found in y_true")

config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml")
with open(config_path, "r") as f:
    config = yaml.safe_load(f)

model_type = "SpectralGCN"  # Change this to the desired model type, e.g., "SpectralK", "SpectralKCombinedLoss", etc.

model_config = config[model_type]
lr = float(model_config["lr"])
batch_size = model_config["batch_size"]
max_epochs = model_config["max_epochs"]
hidden_channels = model_config["hidden_channels"]
dropout = model_config["dropout"]
num_heads = model_config.get("num_heads", None)
monitor_metric = model_config["monitor_metric"]
patience = model_config["patience"]
num_layers = model_config["num_layers"]
num_vn = model_config.get("num_vn", None)

for k in range(4,10):
    print(f"\n\n\nTraining {model_type} model with k={k}:\n\n\n")
    train(
            model_type=model_type,
            lr=lr,
            batch_size=batch_size,
            max_epochs= max_epochs,
            hidden_channels=hidden_channels,
            k=k,
            num_heads=num_heads,
            patience=patience,
            dropout=dropout,
            monitor_metric=monitor_metric,
            num_layers=num_layers,
            num_vn=num_vn
        )
