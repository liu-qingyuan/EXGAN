# EXGAN
This is a fork of the official implementation of "Generating Counterfactual Instances for Explainable Class-Imbalance Learning" from [smallcube/EXGAN](https://github.com/smallcube/EXGAN). We only upload the code for training model on image data (i.e., imbalanced MNIST).
The code for training on conventional data will be uploaded once the paper has been accepted.

## Requirements
```
- Python 3.8+
- PyTorch 1.8+
- CUDA (optional, but recommended)
- scikit-learn
- numpy
- pandas
- tqdm
```

## Installation
```bash
# Clone the repository
git clone https://github.com/liu-qingyuan/EXGAN.git
cd EXGAN

# Install dependencies
pip install -r requirements.txt
```

## Usage

### For MNIST Dataset
Train the model on MNIST dataset with imbalance ratio 100:
```bash
python train_MINIST.py --ir 100 --max_epochs 10 --ensemble_num 1 --batch_size 128
```

### For Tabular Dataset
Train the model on healthcare dataset:
```bash
python train_tabular.py --ir 1.78 --max_epochs 50 --ensemble_num 3 --batch_size 32 --lr_g 0.0001 --lr_d 0.0001
```

Key parameters:
```
# For MNIST
--ir: Imbalance ratio (default: 100)
--max_epochs: Number of training epochs (default: 10)
--ensemble_num: Number of discriminators in ensemble (default: 1)
--batch_size: Batch size for training (default: 128)
--lr_g: Learning rate for generator (default: 0.0002)
--lr_d: Learning rate for discriminator (default: 0.0002)
--init_type: Weight initialization type (default: 'ortho')
--SN_used: Whether to use Spectral Normalization (default: True)

# For Tabular Data
--ir: Imbalance ratio (default: 1.78)
--max_epochs: Maximum training epochs (default: 50)
--ensemble_num: Number of discriminators in ensemble (default: 3)
--batch_size: Batch size (default: 32)
--lr_g: Generator learning rate (default: 0.0001)
--lr_d: Discriminator learning rate (default: 0.0001)
```

## Model Architecture
The EXGAN model consists of:
```
1. Two Generators:
   - G_A_to_B: Transforms majority class samples to minority class style
   - G_B_to_A: Transforms minority class samples to majority class style

2. Ensemble of Discriminators:
   - Each discriminator performs both real/fake and class discrimination
   - Uses spectral normalization for stable training

3. Loss Functions:
   - Adversarial loss for real/fake discrimination
   - Classification loss for class discrimination
   - Cycle consistency loss for preserving sample identity
   - Consistency loss for maintaining class characteristics
```

### Tabular Model Architecture
For tabular data, the model uses:
```python
Generator(
    input_dim=63,          # Feature dimension
    hidden_dim=256,        # Hidden layer dimension
    layers=[
        Linear + InstanceNorm + ReLU + Dropout,
        Linear + InstanceNorm + ReLU + Dropout,
        Linear + Tanh
    ]
)
```

## Directory Structure
```
EXGAN/
├── models/
│   ├── EX_GAN_MINIST.py    # MNIST model implementation
│   ├── EX_GAN_Tabular.py   # Tabular model implementation
│   ├── layers.py           # Custom layer implementations
│   ├── losses_original.py  # Loss function definitions
│   └── utils.py           # Utility functions
├── data/
│   └── Health/            # Healthcare dataset directory
│       └── AI4healthcare.xlsx
├── log/                    # Training logs and checkpoints
├── results/                # Evaluation results
├── train_MINIST.py        # MNIST training script
├── train_tabular.py       # Tabular data training script
└── requirements.txt        # Dependencies
```

## Results
Results will be saved in:
```
# For MNIST
./results/EX-GAN.csv        # Performance metrics

# For Tabular Data
./results/EX-GAN-Health.csv        # Per-fold results
./results/EX-GAN-Health-Final.csv  # Final averaged results
```

## Notes
- For MNIST:
  - The code automatically downloads the MNIST dataset
  - GPU is recommended for training, but not required

- For Tabular Data:
  - Place your healthcare dataset in data/Health/AI4healthcare.xlsx
  - Results include AUC, F1, ACC, ACC_0, ACC_1, and G-mean metrics
  - Uses 10-fold cross validation
  - Model checkpoints are saved in logs/Health/checkpoint_best.pth

