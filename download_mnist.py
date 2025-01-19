# download_mnist.py
import torchvision
import os

# 确保目录存在
os.makedirs('data/MNIST/raw', exist_ok=True)
os.makedirs('data/MNIST/processed', exist_ok=True)

# 下载MNIST数据集
print("Downloading MNIST dataset...")
train_dataset = torchvision.datasets.MNIST(
    root='./data',
    train=True,
    download=True
)

test_dataset = torchvision.datasets.MNIST(
    root='./data',
    train=False,
    download=True
)

print("Download complete!")