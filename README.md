# Combining Spectral Filtering and Probabilistic Virtual Nodes to Handle Over-Squashing

by Alon Fainshtein and Noam Cohen

## Relevant Papers
Sato, Y., Zügner, D., & Günnemann, S. (2024). Spatio-Spectral Graph Neural Networks. NeurIPS 2024. https://arxiv.org/abs/2405.19121
Sato, Y., Zügner, D., & Günnemann, S. (2023). Implicit Probabilistic Rewiring of Message Passing Neural Networks. ICLR 2023. https://arxiv.org/abs/2405.17311


## 🚀 Quick Start

### Prerequisites

Install the required dependencies:

```bash
pip install torch torch-geometric pytorch-lightning wandb pyyaml torchmetrics scikit-learn
```

### Training and Testing

1. **Configure the model** in `config.yaml`:
   ```yaml
   model_type: "GCN"  # Options: GCN, Vn, SpectralGCN, Combined, CombinedLoss
   ```

2. **Run training**:
   ```bash
   python train.py
   ```

3. **Test the model**:
   ```bash
   python test.py
   ```

## ⚙️ Configuration

The `config.yaml` file contains hyperparameters for each model type:

### Common Parameters
- `in_channels`: Input feature dimensions (32)
- `hidden_channels`: Hidden layer dimensions (64)
- `out_channels`: Output dimensions (10)
- `num_layers`: Number of GNN layers (3)
- `dropout`: Dropout rate (0.3)
- `lr`: Learning rate (0.0001)
- `batch_size`: Batch size (32)
- `max_epochs`: Maximum training epochs (100)
- `patience`: Early stopping patience (10)

### Model-Specific Parameters
- **SpectralGCN**: `k` (spectral components), `num_heads` (attention heads)
- **Vn**: `k` (spectral components), `num_vn` (virtual nodes)
- **Combined/CombinedLoss**: `k`, `num_heads`, `num_vn`

## 📊 Models

### 1. Graph Convolutional Network (GCN)
Standard graph convolutional network for node classification with global mean pooling for graph-level predictions.

**Key Features:**
- Uses `GCNConv` layers with ReLU activation
- Global mean pooling for graph-level representation
- Binary cross-entropy loss with logits
- Average precision and accuracy metrics

### 2. Spectral Graph Convolutional Network (SpectralGCN)
Enhanced GCN with spectral decomposition and multi-head attention mechanisms.

### 3. Virtual Node Model (Vn)
Implements probabilistic virtual nodes to improve graph representation learning.

### 4. Combined Model
Hybrid architecture combining multiple GNN approaches for enhanced performance.

### 5. Combined Loss Model
Advanced combined model with specialized loss functions for better optimization.

## 🔧 Usage Examples

### Training Different Models

```bash
# Train GCN
python train.py  # Set model_type: "GCN" in config.yaml

# Train SpectralGCN with specific k
python train_spectral_k.py

# Train Virtual Node model
# Set model_type: "Vn" in config.yaml and run train.py
```

### Testing Models

```bash
# Test the best checkpoint
python test.py
```

### Configuration Examples

```yaml
# GCN Configuration
GCN:
  in_channels: 32
  hidden_channels: 64
  out_channels: 10
  num_layers: 3
  dropout: 0.3
  lr: 0.0001
  batch_size: 32
  monitor_metric: "val_ap"
  patience: 10
  max_epochs: 100

# SpectralGCN Configuration
SpectralGCN:
  in_channels: 32
  hidden_channels: 64
  out_channels: 10
  num_layers: 3
  dropout: 0.3
  lr: 0.0001
  batch_size: 32
  k: 4
  num_heads: 4
  monitor_metric: "val_ap"
  patience: 10
  max_epochs: 100
```

## 📝 Data

The project uses the **Peptides-Func** dataset from the LRGB benchmark:
- **Task**: Multi-label classification of peptide functions
- **Graphs**: 15,535 peptide graphs
- **Features**: 32-dimensional node features
- **Labels**: 10 binary classification tasks
- **Splits**: Train/Validation/Test splits provided by the dataset

## 🛠️ Technologies

- **PyTorch**: Deep learning framework
- **PyTorch Geometric**: Graph neural network library
- **PyTorch Lightning**: Training framework with automatic logging
- **Weights & Biases**: Experiment tracking and visualization
- **TorchMetrics**: Evaluation metrics
- **Scikit-learn**: Additional metrics (average precision)
- **YAML**: Configuration management

## 📊 Metrics

The models are evaluated using:
- **Average Precision (AP)**: Primary metric for multi-label classification

## 🔑 Key Features

- **Multi-model comparison**: Compare different GNN architectures
- **Automatic checkpointing**: Best models saved automatically
- **Early stopping**: Prevents overfitting
- **Experiment tracking**: Integration with Weights & Biases
- **GPU support**: Automatic GPU detection and utilization
- **Mixed precision**: 16-bit precision for faster training on compatible GPUs

## 📦 Outputs

- **Checkpoints**: Best model weights saved in `checkpoints/`
- **Logs**: Training logs in `lightning_logs/`
- **WandB**: Experiment tracking and visualization
- **Metrics**: Validation AP scores and training curves

### Training Process
1. **Data Loading**: Peptides-Func dataset with train/val/test splits
2. **Model Initialization**: Based on configuration
3. **Training Loop**: PyTorch Lightning handles training, validation, and logging
4. **Checkpointing**: Best model saved based on validation AP
5. **Early Stopping**: Training stops if no improvement for specified patience

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test with different model configurations
5. Submit a pull request

---

For questions or issues, please open an issue in the repository.
