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
Train the model on MNIST dataset with imbalance ratio 100:
```bash
python train_MINIST.py --ir 100 --max_epochs 10 --ensemble_num 1 --batch_size 128
```

Key parameters:
```
--ir: Imbalance ratio (default: 100)
--max_epochs: Number of training epochs (default: 10)
--ensemble_num: Number of discriminators in ensemble (default: 1)
--batch_size: Batch size for training (default: 128)
--lr_g: Learning rate for generator (default: 0.0002)
--lr_d: Learning rate for discriminator (default: 0.0002)
--init_type: Weight initialization type (default: 'ortho')
--SN_used: Whether to use Spectral Normalization (default: True)
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

## Directory Structure
```
EXGAN/
├── models/
│   ├── EX_GAN_MINIST.py    # Main model implementation
│   ├── layers.py           # Custom layer implementations
│   ├── losses_original.py  # Loss function definitions
│   └── utils.py           # Utility functions
├── data/                   # Data directory (created automatically)
├── log/                    # Training logs and checkpoints
├── results/                # Evaluation results
├── train_MINIST.py        # Training script
└── requirements.txt        # Dependencies
```

## Results
Results will be saved in:
```
./results/EX-GAN.csv        # Performance metrics (AUC, F1-score, etc.)
./log/                      # Model checkpoints and training logs
```

## Notes
- The code automatically downloads the MNIST dataset
- Training logs and model checkpoints are saved in the 'log' directory
- Final results are saved in CSV format in the 'results' directory
- GPU is recommended for training, but not required
