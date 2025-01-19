import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'

import numpy as np
import math
import functools
import random
import os
from collections import defaultdict

import torch
import torch.nn as nn
from torch.nn import init
import torch.optim as optim
import torch.nn.functional as F
from torch.nn import Parameter as P
import gc

from .layers import SNLinear
from .losses_original import loss_dis_real, loss_dis_fake
from .pyod_utils import get_measure
from .TabularDataset import TabularDataset
from tqdm import tqdm

class Generator(nn.Module):
    def __init__(self, input_dim=63, hidden_dim=128, init='ortho', SN_used=True):
        super(Generator, self).__init__()
        self.init = init
        
        # 使用全连接层替代卷积层
        if SN_used:
            self.which_linear = functools.partial(SNLinear, num_svs=1, num_itrs=1)
        else:
            self.which_linear = nn.Linear
            
        self.model = nn.Sequential(
            self.which_linear(input_dim, hidden_dim),
            nn.ReLU(),
            self.which_linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            self.which_linear(hidden_dim, input_dim),
            nn.Tanh()  # 使用Tanh确保输出在合理范围
        )
        
        self.init_weights()
        
    def init_weights(self):
        self.param_count = 0
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Embedding)):
                if self.init == 'ortho':
                    init.orthogonal_(module.weight)
                elif self.init == 'N02':
                    init.normal_(module.weight, 0, 0.02)
                elif self.init in ['glorot', 'xavier']:
                    init.xavier_uniform_(module.weight)
                else:
                    print('Init style not recognized...')
                self.param_count += sum([p.data.nelement() for p in module.parameters()])
        print('Param count for G''s initialized parameters: %d' % self.param_count)

    def forward(self, x):
        return self.model(x)

class Discriminator(nn.Module):
    def __init__(self, input_dim=63, hidden_dim=128, num_class=2, init='ortho', SN_used=True):
        super(Discriminator, self).__init__()
        self.n_classes = num_class
        self.init = init

        if SN_used:
            self.which_linear = functools.partial(SNLinear, num_svs=1, num_itrs=1)
        else:
            self.which_linear = nn.Linear

        self.which_embedding = nn.Embedding
        
        # 特征提取器
        self.feature_extractor = nn.Sequential(
            self.which_linear(input_dim, hidden_dim),
            nn.ReLU(),
            self.which_linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # 真假判别器
        self.output_fc = self.which_linear(hidden_dim, 1)
        
        # 类别判别器
        self.output_category = nn.Sequential(
            self.which_linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
        # 嵌入层
        self.embed = self.which_embedding(self.n_classes, hidden_dim)
        
        self.init_weights()
    
    def init_weights(self):
        self.param_count = 0
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Embedding)):
                if self.init == 'ortho':
                    init.orthogonal_(module.weight)
                elif self.init == 'N02':
                    init.normal_(module.weight, 0, 0.02)
                elif self.init in ['glorot', 'xavier']:
                    init.xavier_uniform_(module.weight)
                else:
                    print('Init style not recognized...')
                self.param_count += sum([p.data.nelement() for p in module.parameters()])
    
    def forward(self, x, y=None, mode=0):
        if y is not None:
            y = y.to(x.device)
            
        # 提取特征
        h = self.feature_extractor(x)
        
        if mode == 0:  # 训练整个判别器网络
            out = self.output_fc(h)
            self.embed = self.embed.to(x.device)
            out_real_fake = out + torch.sum(self.embed(y) * h, 1, keepdim=True)
            out_category = self.output_category(h)
            return out_real_fake, out_category
        elif mode == 1:  # 只训练真假判别
            return self.output_fc(h)
        else:  # 只训练类别判别
            return self.output_category(h)

class EX_GAN(nn.Module):
    def __init__(self, args, lamd=1.0):
        gc.collect()
        torch.cuda.empty_cache()
        print("Initial GPU memory cleared")
        
        super(EX_GAN, self).__init__()
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        self.Cycle_Loss = nn.L1Loss()
        self.l1_loss = nn.L1Loss()
        self.lamd = lamd

        manualSeed = random.randint(1, 10000)
        random.seed(manualSeed)
        torch.manual_seed(manualSeed)
    
        # 准备生成器
        self.netG_A_to_B = Generator(
            input_dim=63,  # 特征维度
            hidden_dim=128,  # 隐藏层维度
            init=args.init_type, 
            SN_used=args.SN_used
        )
        self.netG_B_to_A = Generator(
            input_dim=63,
            hidden_dim=128,
            init=args.init_type, 
            SN_used=args.SN_used
        )
        
        self.netG_A_to_B = self.netG_A_to_B.to(self.device)
        self.netG_B_to_A = self.netG_B_to_A.to(self.device)

        self.optimizerG = optim.Adam(
            list(self.netG_A_to_B.parameters()) + list(self.netG_B_to_A.parameters()),
            lr=args.lr_g, 
            betas=(0.00, 0.99)
        )
        
        # 创建判别器集成
        self.NetD_Ensemble = []
        self.opti_Ensemble = []
        lr_ds = np.random.rand(args.ensemble_num) * (args.lr_d*5-args.lr_d) + args.lr_d
        
        for index in range(args.ensemble_num):
            netD = Discriminator(
                input_dim=63,
                hidden_dim=128,
                num_class=2,
                init=args.init_type, 
                SN_used=args.SN_used
            )
            netD = netD.to(self.device)
            optimizerD = optim.Adam(netD.parameters(), lr=lr_ds[index], betas=(0.00, 0.99))
            self.NetD_Ensemble += [netD]
            self.opti_Ensemble += [optimizerD]

    def fit(self, train_data=None, test_data=None):
        print("\nPreparing datasets...")
        
        if train_data is None or test_data is None:
            try:
                print("Using default dataset split...")
                train_dataset = TabularDataset('./data', train=True, ir=self.args.ir)
                test_dataset = TabularDataset('./data', train=False)
            except Exception as e:
                print(f"Error loading dataset: {str(e)}")
                raise
        else:
            print("Using provided dataset split...")
            train_dataset = train_data
            test_dataset = test_data

        print("Creating data loaders...")
        self.train_loader = torch.utils.data.DataLoader(
            train_dataset, 
            batch_size=self.args.batch_size, 
            shuffle=True
        )
        self.test_loader = torch.utils.data.DataLoader(
            test_dataset, 
            batch_size=self.args.batch_size
        )

        print("Data loading complete!")
        print(f"Train set size: {len(self.train_loader.dataset)}")
        print(f"Test set size: {len(self.test_loader.dataset)}")
        
        log_dir = os.path.join(self.args.log_path, self.args.data_name)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        # Start iteration
        Best_Measure_Recorded = -1
        Best_AUC_test = -1
        Best_F_test = -1
        Best_AUC_train = -1
        Best_F_train = -1
        self.train_history = defaultdict(list)
        
        for epoch in range(self.args.max_epochs):
            train_AUC, train_score, train_Gmean, test_auc, test_score, test_gmean = self.train_one_epoch(epoch, self.train_loader, self.test_loader)
            if train_score*train_AUC > Best_Measure_Recorded:
                Best_Measure_Recorded = train_score*train_AUC
                Best_AUC_test = test_auc
                Best_F_test = test_score
                Best_AUC_train = train_AUC
                Best_F_train = train_score

                states = {
                    'epoch':epoch,
                    'gen_A_to_B':self.netG_A_to_B.state_dict(),
                    'gen_B_to_A':self.netG_B_to_A.state_dict(),
                    'max_auc':train_AUC
                }
                for i in range(self.args.ensemble_num):
                    netD = self.NetD_Ensemble[i]
                    optimi_D = self.opti_Ensemble[i]
                    states['dis_dict'+str(i)] = netD.state_dict()
                
                torch.save(states, os.path.join(log_dir, 'checkpoint_best.pth'))
            #print(train_AUC, test_AUC, epoch)
            if self.args.print:
                print('Epoch %d: Train_AUC=%.4f train_fscore=%.4f train_Gmean=%.4f Test_AUC=%.4f test_fscore=%.4f Test_Gmean=%.4f' % (epoch + 1, train_AUC, train_score, train_Gmean, test_auc, test_score, test_gmean))
        
       
        #step 1: load the best models
        self.Best_Ensemble = []
        try:
            # 添加 map_location 参数来确保模型加载到正确的设备
            states = torch.load(
                os.path.join(log_dir, 'checkpoint_best.pth'),
                map_location=self.device
            )
            self.netG_A_to_B.load_state_dict(states['gen_A_to_B'])
            self.netG_B_to_A.load_state_dict(states['gen_B_to_A'])
            for i in range(self.args.ensemble_num):
                netD = self.NetD_Ensemble[i]
                netD.load_state_dict(states['dis_dict'+str(i)])
                # 确保判别器在正确的设备上
                netD = netD.to(self.device)
                self.Best_Ensemble += [netD]
        except Exception as e:
            print(f"Warning: Could not load best model: {str(e)}")
            # 如果加载失败，使用当前模型
            self.Best_Ensemble = self.NetD_Ensemble
        
        return Best_AUC_train, Best_F_train, Best_AUC_test, Best_F_test
    
    def predict(self, data_loader, dis_Ensemble=None, need_explain=False):
        y_pred = []
        y_true = []
        data_ex = []
        data = []
        for i, (digits, labels) in enumerate(data_loader):
            # 将数据移到正确的设备上
            digits = digits.to(self.device)
            labels = labels.to(self.device)
            
            for i in range(self.args.ensemble_num):
                pt = self.Best_Ensemble[i](digits, mode=2) if dis_Ensemble is None else dis_Ensemble[i](digits, mode=2)
                if i==0:
                    final_pt = pt.detach()
                else:
                    final_pt += pt
            final_pt /= self.args.ensemble_num
            final_pt = final_pt.view(final_pt.shape[0],)
            y_pred += [final_pt]
            y_true += [labels.view(labels.shape[0],)]

            digits_ex = torch.zeros_like(digits)
            data += [digits.detach()]

            if need_explain:
                index0 = (final_pt<0.5)
                index1 = (final_pt>=0.5)
                data0 = digits[index0]
                data1 = digits[index1]
                if data0.shape[0]>0:
                    data0_ex = self.netG_A_to_B(data0)
                    digits_ex[index0] = data0_ex
                if data1.shape[0]>0:
                    data1_ex = self.netG_B_to_A(data1)
                    digits_ex[index1] = data1_ex
                
                data_ex += [digits_ex]
        y_pred = torch.cat(y_pred, 0)
        y_true = torch.cat(y_true, 0)
        y_pred = y_pred.detach().cpu().numpy()
        y_true = y_true.detach().cpu().numpy()
        if need_explain:
            data_ex = torch.cat(data_ex, dim=0)
            
        return y_pred, y_true, data_ex


    def train_one_epoch(self, epoch=1, train_loader=None, test_loader=None):
        print(f"\nEpoch {epoch}/{self.args.max_epochs}")
        
        # 使用tqdm创建进度条
        pbar = tqdm(total=len(train_loader), desc=f'Training Epoch {epoch}')
        
        for batch_idx, (digits, labels) in enumerate(train_loader):
            pbar.update(1)
            
            # 将数据移到GPU
            digits = digits.to(self.device)
            labels = labels.to(self.device)
            
            # 获取训练数据
            index0 = (labels==0)
            index1 = (labels==1)
            real_x0 = digits[index0]
            real_x1 = digits[index1]

            # 生成器前向传播
            fake_B = self.netG_A_to_B(real_x0)
            fake_A = self.netG_B_to_A(real_x1)
            reconstrcuted_A = self.netG_B_to_A(fake_B)
            reconstrcuted_B = self.netG_A_to_B(fake_A)

            fake_x = torch.cat([fake_A, fake_B], 0)
            fake_y = torch.cat([torch.zeros(fake_A.shape[0],), torch.ones(fake_B.shape[0],)], 0).long().to(self.device)
            real_x = torch.cat([real_x0, real_x1], 0)
            real_y = torch.cat([torch.zeros(real_x0.shape[0],), torch.ones(real_x1.shape[0],)], 0).long().to(self.device)
            
            # 训练判别器
            dis_loss = 0
            adv_weight_real, cat_weight_real, adv_weight_fake, cat_weight_fake = None, None, None, None
            for i in range(self.args.ensemble_num):
                optimizer = self.opti_Ensemble[i]
                netD = self.NetD_Ensemble[i]

                out_adv_real, out_cat_real = netD(real_x, real_y)
                loss_adv_real, loss_cat_real, adv_weight_real, cat_weight_real = loss_dis_real(
                    out_adv_real, out_cat_real, real_y, adv_weight_real, cat_weight_real)
                real_loss = loss_adv_real + loss_cat_real
                
                out_adv_fake, out_cat_fake = netD(fake_x.detach(), fake_y.detach())
                loss_adv_fake, loss_cat_fake, adv_weight_fake, cat_weight_fake = loss_dis_fake(
                    out_adv_fake, out_cat_fake, fake_y.detach(), adv_weight_fake, cat_weight_fake)
                fake_loss = loss_adv_fake + loss_cat_fake
                
                sum_loss = real_loss + fake_loss
                dis_loss += sum_loss

                optimizer.zero_grad()
                sum_loss.backward(retain_graph=True)
                optimizer.step()

            # 训练生成器
            gen_loss = 0
            adv_weight_gen, cat_weight_gen = None, None
            for i in range(self.args.ensemble_num):
                netD = self.NetD_Ensemble[i]
                out_adv, out_cat = netD(fake_x, fake_y)
                loss_adv, loss_cat, adv_weight_gen, cat_weight_gen = loss_dis_real(
                    out_adv, out_cat, fake_y, adv_weight_gen, cat_weight_gen)
                gen_loss += (loss_adv + loss_cat)
                
            cycle_loss = self.l1_loss(reconstrcuted_A, real_x0) + self.l1_loss(reconstrcuted_B, real_x1)
            consistency_loss = self.l1_loss(real_x0, fake_B) + self.l1_loss(real_x1, fake_A)
            gen_loss += cycle_loss + 1.0*epoch/self.args.max_epochs * consistency_loss

            self.optimizerG.zero_grad()
            gen_loss.backward()
            self.optimizerG.step()

            # 每5个batch清理一次内存
            if batch_idx % 5 == 0:
                torch.cuda.empty_cache()
            
            # 显示当前GPU内存使用情况
            if batch_idx % 20 == 0:
                allocated = torch.cuda.memory_allocated() / 1024**3
                reserved = torch.cuda.memory_reserved() / 1024**3
                pbar.set_postfix({'GPU Memory (GB)': f'Used: {allocated:.1f}, Reserved: {reserved:.1f}'})

        # 评估
        print("\nEvaluating...")
        with torch.no_grad():
            y_pred_train, y_true_train, _ = self.predict(train_loader, self.NetD_Ensemble)
            y_pred_test, y_true_test, _ = self.predict(test_loader, self.NetD_Ensemble)
            
            auc_train, fscore_train, gmean_train = get_measure(y_true_train, y_pred_train)
            auc_test, fscore_test, gmean_test = get_measure(y_true_test, y_pred_test)

        pbar.close()
        return auc_train, fscore_train, gmean_train, auc_test, fscore_test, gmean_test
