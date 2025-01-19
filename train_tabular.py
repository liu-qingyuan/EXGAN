# -*- coding: utf-8 -*-
import os
import sys
from time import time

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname("__file__"), '..')))

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import KFold

from models.pyod_utils import get_measure
from models.utils import parse_args
from models.EX_GAN_Tabular import EX_GAN
from models.TabularDataset import TabularDataset

n_folds = 10
result_dir = "./results/"

df_columns = ['IR', 'AUC', 'F1', 'time', 'time_std']
roc_df = pd.DataFrame(columns=df_columns)

# initialize the container for saving the results
result_df = pd.DataFrame(columns=df_columns)
args = parse_args()

# 设置数据集名称
args.data_name = 'Health'
    
ir = args.ir

# 首先加载完整数据集并创建标准化器
raw_dataset = TabularDataset('./data', train=True, ir=args.ir)
scaler = raw_dataset.get_scaler()
X = raw_dataset.data.numpy()
y = raw_dataset.targets.numpy()

# 创建K折交叉验证
kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

time_mat = np.zeros([n_folds, 1])
roc_mat = np.zeros([n_folds, 1])
fscore_mat = np.zeros([n_folds, 1])

# 在开始训练前添加打印信息
print("Starting training with settings:")
print(f"IR: {args.ir}")
print(f"Max epochs: {args.max_epochs}")
print(f"Ensemble num: {args.ensemble_num}")
print("\nStarting 10-fold cross validation...")

for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
    print(f"\nFold {fold + 1}/{n_folds}")
    t0 = time()
    
    # 创建当前fold的数据集，共享相同的标准化器
    train_dataset = TabularDataset(
        './data',
        train=True,
        ir=args.ir,
        train_indices=train_idx,
        test_indices=test_idx,
        scaler=scaler  # 使用相同的标准化器
    )
    test_dataset = TabularDataset(
        './data',
        train=False,
        ir=args.ir,
        train_indices=train_idx,
        test_indices=test_idx,
        scaler=scaler  # 使用相同的标准化器
    )
    
    print("Initializing EX_GAN...")
    cb_gan = EX_GAN(args)
    
    print("Starting model training...")
    auc_train, f_train, auc_test, f_test = cb_gan.fit(
        train_data=train_dataset,
        test_data=test_dataset
    )
    
    t1 = time()
    duration = round(t1 - t0, ndigits=4)
    
    # 在每个fold结束时输出详细结果
    print(f"\nFold {fold + 1} Detailed Results:")
    print(f"Train Set Size: {len(train_dataset)}")
    print(f"Test Set Size: {len(test_dataset)}")
    print(f"Train Class Distribution: {np.bincount(train_dataset.targets)}")
    print(f"Test Class Distribution: {np.bincount(test_dataset.targets)}")
    print(f"Training - AUC: {auc_train:.4f}, F-score: {f_train:.4f}")
    print(f"Testing  - AUC: {auc_test:.4f}, F-score: {f_test:.4f}")
    print(f"Execution time: {duration:.4f} seconds")
    print("-" * 50)

    time_mat[fold, 0] = duration
    roc_mat[fold, 0] = auc_test
    fscore_mat[fold, 0] = f_test

# 计算平均结果
print("\nOverall Results:")
print(f"Average Test AUC: {np.mean(roc_mat):.4f} ± {np.std(roc_mat):.4f}")
print(f"Average Test F1: {np.mean(fscore_mat):.4f} ± {np.std(fscore_mat):.4f}")
print(f"Average Time: {np.mean(time_mat):.4f} ± {np.std(time_mat):.4f}")

roc_list = [ir]
roc_list.append(np.mean(roc_mat, axis=0).item())
roc_list.append(np.mean(fscore_mat, axis=0).item())
roc_list.append(np.mean(time_mat, axis=0).item())
roc_list.append(np.std(time_mat, axis=0).item())

temp_df = pd.DataFrame(roc_list).transpose()
temp_df.columns = df_columns
roc_df = pd.concat([roc_df, temp_df], axis=0)

# Save the results for each run
save_path1 = os.path.join(result_dir, 'EX-GAN-Health.csv')
roc_df.to_csv(save_path1, index=False, float_format='%.3f') 