# -*- coding: utf-8 -*-
import os
import sys
from time import time

from numpy.core.numeric import count_nonzero
from sklearn.utils import shuffle

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname("__file__"), '..')))

import numpy as np
import pandas as pd

import torch

from models.pyod_utils import min_max_normalization, AUC_and_Gmean, get_measure
from models.pyod_utils import precision_n_scores, gmean_scores
from models.utils import parse_args
from sklearn.metrics import roc_auc_score
from models.EX_GAN_MINIST import EX_GAN

from models.Imbalanced_MINIST import MNIST
from torchvision import transforms

n_folds = 10
result_dir = "./results/"

df_columns = ['IR', 'AUC', 'F1', 'time', 'time_std']
roc_df = pd.DataFrame(columns=df_columns)

# initialize the container for saving the results
result_df = pd.DataFrame(columns=df_columns)
args = parse_args()


#define the dataloader
    
ir = args.ir

# construct containers for saving results
roc_list = [ir]

time_mat = np.zeros([n_folds, 1])
roc_mat = np.zeros([n_folds, 1])
fscore_mat = np.zeros([n_folds, 1])

# 在开始训练前添加打印信息
print("Starting training with settings:")
print(f"IR: {args.ir}")
print(f"Max epochs: {args.max_epochs}")
print(f"Ensemble num: {args.ensemble_num}")
print("\nInitializing model...")

#repeat the k-fold cross validation n_iterations times
count = 0
for _ in range(n_folds):
    print(f"\nStarting fold {_ + 1}/{n_folds}")
    t0 = time()
    
    print("Initializing EX_GAN...")
    cb_gan = EX_GAN(args)
    
    print("Starting model training...")
    auc_train, f_train, auc_test, f_test = cb_gan.fit()
    
    t1 = time()
    duration = round(t1 - t0, ndigits=4)
    
    print(f'Fold {_ + 1} Results:')
    print(f'Training - AUC: {auc_train:.4f}, F-score: {f_train:.4f}')
    print(f'Testing  - AUC: {auc_test:.4f}, F-score: {f_test:.4f}')
    print(f'Execution time: {duration} seconds')

    time_mat[count, 0] = duration
    roc_mat[count, 0] = auc_test
    fscore_mat[count, 0] = f_test
    count += 1

roc_list = [ir]
roc_list.append(np.mean(roc_mat, axis=0).item())
roc_list.append(np.mean(fscore_mat, axis=0).item())
roc_list.append(np.mean(time_mat, axis=0).item())
roc_list.append(np.std(time_mat, axis=0).item())

temp_df = pd.DataFrame(roc_list).transpose()
temp_df.columns = df_columns
roc_df = pd.concat([roc_df, temp_df], axis=0)

# Save the results for each run
save_path1 = os.path.join(result_dir, 'EX-GAN.csv')
roc_df.to_csv(save_path1, index=False, float_format='%.3f')