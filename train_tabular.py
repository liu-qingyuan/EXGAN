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

df_columns = ['IR', 'AUC', 'F1', 'ACC', 'ACC_0', 'ACC_1', 'G-mean', 'time', 'time_std']
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
    # 训练模型并获取预测结果
    auc_train, f_train, gmean_train, acc_train, acc_0_train, acc_1_train, \
    auc_test, f_test, gmean_test, acc_test, acc_0_test, acc_1_test = cb_gan.fit(
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
    print("\nTraining Metrics:")
    print(f"AUC: {auc_train:.4f}, F1: {f_train:.4f}, G-mean: {gmean_train:.4f}")
    print(f"ACC: {acc_train:.4f} (Class 0: {acc_0_train:.4f}, Class 1: {acc_1_train:.4f})")
    print("\nTesting Metrics:")
    print(f"AUC: {auc_test:.4f}, F1: {f_test:.4f}, G-mean: {gmean_test:.4f}")
    print(f"ACC: {acc_test:.4f} (Class 0: {acc_0_test:.4f}, Class 1: {acc_1_test:.4f})")
    print(f"\nExecution time: {duration:.4f} seconds")
    print("-" * 80)

    time_mat[fold, 0] = duration
    roc_mat[fold, 0] = auc_test
    fscore_mat[fold, 0] = f_test

    # 保存每个fold的结果
    fold_results = {
        'IR': ir,
        'AUC': auc_test,
        'F1': f_test,
        'ACC': acc_test,
        'ACC_0': acc_0_test,
        'ACC_1': acc_1_test,
        'G-mean': gmean_test,
        'time': duration,
        'time_std': 0
    }
    temp_df = pd.DataFrame([fold_results])
    roc_df = pd.concat([roc_df, temp_df], ignore_index=True)
    
    # 保存当前进度
    save_path = os.path.join(result_dir, 'EX-GAN-Health.csv')
    roc_df.to_csv(save_path, index=False, float_format='%.3f')

# 计算并更新最终结果
final_results = {
    'IR': ir,
    'AUC': np.mean(roc_df['AUC']),
    'F1': np.mean(roc_df['F1']),
    'ACC': np.mean(roc_df['ACC']),
    'ACC_0': np.mean(roc_df['ACC_0']),
    'ACC_1': np.mean(roc_df['ACC_1']),
    'G-mean': np.mean(roc_df['G-mean']),
    'time': np.mean(roc_df['time']),
    'time_std': np.std(roc_df['time'])
}

print("\nFinal Results:")
print(f"Average Test AUC: {final_results['AUC']:.4f} ± {np.std(roc_df['AUC']):.4f}")
print(f"Average Test F1: {final_results['F1']:.4f} ± {np.std(roc_df['F1']):.4f}")
print(f"Average Test ACC: {final_results['ACC']:.4f} ± {np.std(roc_df['ACC']):.4f}")
print(f"Average Test ACC_0: {final_results['ACC_0']:.4f} ± {np.std(roc_df['ACC_0']):.4f}")
print(f"Average Test ACC_1: {final_results['ACC_1']:.4f} ± {np.std(roc_df['ACC_1']):.4f}")
print(f"Average Time: {final_results['time']:.4f} ± {final_results['time_std']:.4f}")

# 保存最终结果
final_df = pd.DataFrame([final_results])
save_path = os.path.join(result_dir, 'EX-GAN-Health-Final.csv')
final_df.to_csv(save_path, index=False, float_format='%.3f') 