import pytorch_lightning as pl
from model import GCN, SpectralGCN
from peptides_func_data import load_data
import torch

def test_model(model_path=None):
    # Load data
    _, _, test_loader, _, _, test_dataset = load_data()
    
    # Initialize model with same architecture
    in_channels = test_dataset[0].x.size(1)
    hidden_channels = 32
    out_channels = test_dataset[0].y.size(1)
    model = GCN(in_channels=in_channels, 
                hidden_channels=hidden_channels,
                out_channels=out_channels)
    
    # Load trained weights if provided
    if model_path:
        model = SpectralGCN.load_from_checkpoint(model_path,
                                    in_channels=in_channels,
                                    hidden_channels=hidden_channels,
                                    out_channels=out_channels)
    
    # Initialize trainer
    trainer = pl.Trainer()
    
    # Test model
    print("\nTesting model...")
    test_results = trainer.test(model, test_loader)
    print(f"Test loss: {test_results[0]['test_loss']:.4f}")
    print(f"Test accuracy: {test_results[0]['test_acc']:.4f}")
    
    return test_results

if __name__ == "__main__":
    # You can specify the path to your saved model here
    # test_model("path/to/your/model.pt")
    test_model(model_path="/gpfs0/tamyr/users/alonfi/GNN Project/spectral_gcn_model_best_epoch=32_val_loss=0.3562.ckpt")
