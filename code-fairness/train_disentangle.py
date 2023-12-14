import os
import math
import time
import random
import torch
import argparse
import numpy as np
import torch
import torch.nn as nn
import utils as ut
import torch.optim as optim
from torch import autograd
from torch.utils import data
# import matplotlib.pyplot as plt
import torch.utils.data as Data
from torch.autograd import Variable
import dataloader as dataload
from codebase.disentangle.config import get_config
from codebase.disentangle.learner import Learner
import sklearn.metrics as skm
import warnings
from sklearn.metrics import accuracy_score, log_loss
import torch.nn.functional as F

warnings.filterwarnings('ignore')
device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
print("torch.cuda.is_available():%s" % (torch.cuda.is_available()))

args, _ = get_config()
workstation_path = './'
# train_dataset_dir = os.path.join(workstation_path, 'dataset', args.dataset, 'dev', 'data_nonuniform.csv')
# test_dataset_dir = os.path.join(workstation_path, 'dataset', args.dataset, 'train', 'data_nonuniform.csv')

if args.dataset == 'huawei':
    train_dataset_dir = os.path.join(workstation_path, 'dataset/', 'huawei', 'dev.csv')
    test_dataset_dir = os.path.join(workstation_path, 'dataset/', 'huawei', 'dev.csv')
    pretrain_dataloader = dataload.dataload_huawei(train_dataset_dir, args.batch_size, mode='dev')
    if args.impression_or_click == 'impression':
        train_dataloader = dataload.dataload_huawei(train_dataset_dir, args.batch_size, mode='dev')
    test_dataloader = dataload.dataload_huawei(test_dataset_dir, args.batch_size, mode='test')
else:
    if args.dataset[:9] == 'logitdata' or args.dataset[:7] == 'invdata' or args.dataset[:3] == 'non':
        syn = True
        train_dataset_dir = os.path.join(workstation_path, 'dataset', args.dataset, 'train', 'data_fair.csv')
        test_dataset_dir = os.path.join(workstation_path, 'dataset', args.dataset, 'dev', 'data_fair.csv')
    else:
        syn = False
        train_dataset_dir = os.path.join(workstation_path, 'dataset', args.dataset, 'train', 'train.csv')
        test_dataset_dir = os.path.join(workstation_path, 'dataset', args.dataset, 'dev', 'dev.csv')
    if args.debias_mode == 'Fairness':
        fair = True
    else:
        fair = False
#     print(fair, fair, fair)


    nagetive_path = os.path.join(workstation_path, 'dataset', args.dataset, 'dev', 'negative_samples.npy')

    if args.debias_mode == 'IPM_Embedding_Sample':
        # ipm_pair_path = os.path.join(workstation_path, 'dataset', args.dataset, 'dev', 'pair_dict.npy')
        # ipm_sample_path = os.path.join(workstation_path, 'dataset', args.dataset, 'dev', 'sample_dict.npy')
        ipm_pair_path = None
        ipm_sample_path = os.path.join(workstation_path, 'dataset', args.dataset, 'dev', 'ipm_data',
                                       'ipm_data5_1.npy')

    else:
        ipm_pair_path = None
        ipm_sample_path = None

    if args.feature_data:
        data_feature_path = os.path.join(workstation_path, 'dataset', args.dataset)
    else:
        data_feature_path = None
    pretrain_dataloader = dataload.dataload_ori(train_dataset_dir, args.batch_size, data_feature_path=data_feature_path,
                                                syn=syn)
    if args.debias_mode == 'IPM_Embedding_K':
        ipm_dataset_dir = os.path.join(workstation_path, 'dataset', args.dataset, 'dev', 'ipm_data', 'data_ipm5.npy')
        ipm_dataloader = dataload.dataload_ipm(ipm_dataset_dir, args.batch_size, data_feature_path=data_feature_path,
                                               syn=syn)

    if args.impression_or_click == 'impression':
        train_dataloader = dataload.dataload_ori(train_dataset_dir, args.batch_size,
                                                 data_feature_path=data_feature_path, syn=syn,
                                                 ipm_sample_path=ipm_sample_path, fairness=fair)
#     test_dataloader = dataload.dataload_ori(test_dataset_dir, 1, data_feature_path=data_feature_path,
#                                             syn=syn, mode='test', nagetive_path=nagetive_path)
    test_dataloader_auc_acc = dataload.dataload_ori(test_dataset_dir, 1000, data_feature_path=data_feature_path,
                                            syn=syn, mode='test', shuffle=False, fairness=fair)
#     val_dataloader = dataload.dataload_ori(test_dataset_dir, args.batch_size, data_feature_path=data_feature_path,
#                                            syn=syn, mode='val', fairness=fair)

# 构建模型
model = Learner(args)
if args.debias_mode in ['Fairness']:
    # training model
    user_group_feature = np.load(os.path.join(workstation_path, 'dataset', args.dataset, 'user_features_group.npy'), allow_pickle=True).item()
    item_group_feature = np.load(os.path.join(workstation_path, 'dataset', args.dataset, 'item_features_group.npy'), allow_pickle=True).item()
    for i in model.parameters():
        i.requires_grad = False
    for i in model.debias_model.discriminitor.parameters():
        i.requires_grad = True
    # define optimizer for max
    max_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-5, betas=(0.9, 0.999))

    for i in model.parameters():
        i.requires_grad = True
    for i in model.debias_model.discriminitor.parameters():
        i.requires_grad = False
    # define optimizer for min
    min_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3, betas=(0.9, 0.999))
#     min_optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999))

    for epoch in range(args.epoch_max):
        model.train()
        emb_loss_max = 0
        emb_loss_min = 0
        ctr_loss = 0
        total_emb_loss = 0
        total_ctr_loss = 0
        auc = 0
        total_auc = 0
        total_acc = 0
        total_logloss = 0
        total_confounder_loss = 0
        total_debias_loss = 0

        if args.impression_or_click == 'impression':
            for x, a, y, r, gx, ga in train_dataloader:
                x, a, y, r, gx, ga = x.to(device), a.to(device), y.to(device), r.to(device), gx.to(device), ga.to(device)
                if not args.use_sensitive:
                    x = torch.concat((x[:, :16], x[:, 32:48]), axis=1)
                    a = a[:, :16]

#                 if args.is_debias:
#                     max_optimizer.zero_grad()
#                     emb_loss_max = -model.learn(x, a, r, y, gx, ga, savestep=epoch, minmax_turn='max')
#                     emb_loss_max.backward()
#                     max_optimizer.step()
#                     total_emb_loss += emb_loss_max.item()
#                     test_dataloadertrain_dataloader_len_len = len(train_dataloader)


                min_optimizer.zero_grad()

                if args.is_debias:
                    emb_loss_min, ctr_loss, kl_loss = model.learn(x, a, r, y, gx, ga, epoch, minmax_turn='min')
                    # print(ctr_loss)
                else:
                    ctr_loss = model.learn(x, a, r, y, gx, ga, epoch, minmax_turn='min')
                if args.downstream == 'MLP':
                    auc = skm.roc_auc_score(y.cpu().numpy(),
                                            torch.argmax(model.predict(x, a), dim=1).cpu().detach().numpy())
                    acc = accuracy_score(y.cpu().numpy(),
                                         torch.argmax(model.predict(x, a), dim=1).cpu().detach().numpy())
                    logloss = log_loss(y.cpu().numpy(),
                                       F.sigmoid(torch.max(model.predict(x, a), dim=1)[0]).cpu().detach().numpy())
                else:
                    y_pred = model.predict(x, a)
                    auc = skm.roc_auc_score(y.cpu().numpy(), y_pred.cpu().detach().numpy())
                    acc = skm.accuracy_score(y.cpu().numpy(), torch.where(y_pred > 0.5, torch.ones_like(y_pred),
                                                                          torch.zeros_like(
                                                                              y_pred)).cpu().detach().numpy())
                    logloss = skm.log_loss(y.cpu().numpy(), y_pred.cpu().detach().numpy())

                L1 = kl_loss if args.is_debias else ctr_loss
                L1.backward()
                min_optimizer.step()

                total_ctr_loss += ctr_loss.item()
                total_auc += auc
                total_acc += acc
                total_logloss += logloss
                total_debias_loss += emb_loss_min

            train_dataloader_len = len(train_dataloader)
            print(
                "Epoch:{}\n train_auc:{}, train_acc:{}, vallogloss:{}, train_debias_loss:{}, train_ctr_loss:{}".format(
                    epoch, float(
                        total_auc / train_dataloader_len), float(total_acc / train_dataloader_len), float(
                        total_logloss / train_dataloader_len), float(
                        total_debias_loss / train_dataloader_len), float(
                        total_ctr_loss / train_dataloader_len)))
        if epoch % 1 == 0:
            total_test_auc = 0
            total_test_acc = 0
            total_test_logloss = 0
            total_test_ndcg = 0
            total_test_recall = 0
            total_test_precision = 0
            test_x_dis = 0
            test_a_dis = 0
            test_non_dis = 0
            test_dataloader_auc_acc_len = len(test_dataloader_auc_acc)
            test_dataloader_len = len(test_dataloader_auc_acc)
            for test_x, test_a, test_y, test_r, test_gx, test_ga  in test_dataloader_auc_acc:
                if not args.use_sensitive:
                    test_x = torch.concat((test_x[:, :16], test_x[:, 32:48]), axis=1)

                    test_a = test_a[:, :16]
                rx, ra, rn, _ = model.get_disentangle(test_x, test_gx, test_ga)
                gx_ = list(test_gx.cpu().detach().numpy())
                ga_ = list(test_ga.cpu().detach().numpy())
#                 print(user_group_feature.values())
                gx_feature = np.array(list(user_group_feature.values()))[gx_]
                ga_feature = np.array(list(item_group_feature.values()))[ga_]


                test_rx = ut.distcorr(rx.cpu().detach().numpy(), gx_feature)
                test_ra = ut.distcorr(ra.cpu().detach().numpy(), ga_feature)
                test_rn = ut.distcorr(rn.cpu().detach().numpy(), np.concatenate((gx_feature, gx_feature), axis=1))


                if args.downstream == 'MLP':
                    test_y_pred = F.sigmoid(model.predict(test_x, test_a).reshape(-1)).cpu().detach().numpy()
                    test_auc = skm.roc_auc_score(test_y.cpu().numpy(), test_y_pred)
                    test_acc = skm.accuracy_score(test_y.cpu().numpy(),
                                                  np.where(test_y_pred > 0.5, np.ones_like(test_y_pred),
                                                              np.zeros_like(test_y_pred)))
                    test_logloss = skm.log_loss(test_y.cpu().detach().numpy(), test_y_pred)

                    if args.dataset == 'coat':
                        top20 = np.repeat(np.sort(test_y_pred.reshape(-1, 16))[:, 5].reshape(-1, 1), 16, axis=1).reshape(-1)
                        test_ndcg = skm.ndcg_score(test_y.reshape(-1, 16), test_y_pred.reshape(-1, 16), k=10)
                    else:
                        top20 = np.repeat(np.sort(test_y_pred.reshape(-1, 5))[:, 3].reshape(-1, 1), 5, axis=1).reshape(-1)
                        test_ndcg = skm.ndcg_score(test_y.reshape(-1, 5), test_y_pred.reshape(-1, 5), k=5)
                    test_pre = skm.precision_score(test_y.int().cpu().numpy(), np.where(test_y_pred > top20, np.ones_like(test_y_pred),
                                                                          np.zeros_like(test_y_pred)))
                    test_rec = skm.recall_score(test_y.int().cpu().numpy(), np.where(test_y_pred > top20, np.ones_like(test_y_pred),
                                                                          np.zeros_like(test_y_pred)))

                else:
                    test_y_pred = F.sigmoid(model.predict(test_x, test_a)).cpu().detach().numpy()
                    test_auc = skm.roc_auc_score(test_y.cpu().numpy(), test_y_pred)
                    test_acc = skm.accuracy_score(test_y.cpu().numpy(),
                                                  np.where(test_y_pred > 0.5, np.ones_like(test_y_pred),
                                                              np.zeros_like(test_y_pred)))
                    test_logloss = skm.log_loss(test_y.cpu().detach().numpy(), test_y_pred)

                    if args.dataset == 'coat':
                        top20 = np.repeat(np.sort(test_y_pred.reshape(-1, 16))[:, 5].reshape(-1, 1), 16, axis=1).reshape(-1)
                        test_ndcg = skm.ndcg_score(test_y.reshape(-1, 16), test_y_pred.reshape(-1, 16), k=10)
                    else:
                        top20 = np.repeat(np.sort(test_y_pred.reshape(-1, 5))[:, 3].reshape(-1, 1), 5, axis=1).reshape(-1)
                        test_ndcg = skm.ndcg_score(test_y.reshape(-1, 5), test_y_pred.reshape(-1, 5), k=10)
                    test_pre = skm.precision_score(test_y.cpu().numpy(), np.where(test_y_pred > top20, np.ones_like(test_y_pred),
                                                                          np.zeros_like(test_y_pred)))
                    test_rec = skm.recall_score(test_y.cpu().numpy(), np.where(test_y_pred > top20, np.ones_like(test_y_pred),
                                                                          np.zeros_like(test_y_pred)))

                total_test_auc += test_auc
                total_test_acc += test_acc
                total_test_logloss += test_logloss
                total_test_ndcg += test_ndcg
                total_test_recall += test_rec
                total_test_precision += test_pre
                test_x_dis += test_rx
                test_a_dis += test_ra
                test_non_dis += test_rn

            print(
            "Epoch {}\n test_rx:{}, test_ra:{} test_rn:{} test_auc:{}, test_acc:{}, test_logloss:{}, test_ndcg:{}, test_recall:{}, test_precision:{}, train_confounder_loss:{}, train_ctr_loss:{}".format(epoch, float(
                test_x_dis / test_dataloader_auc_acc_len), float(
                test_a_dis / test_dataloader_auc_acc_len),
                float(test_non_dis / test_dataloader_auc_acc_len),
                float(total_test_auc / test_dataloader_auc_acc_len),
                float(total_test_acc / test_dataloader_auc_acc_len), float(
                total_test_logloss / test_dataloader_auc_acc_len), float(
                total_test_ndcg / test_dataloader_auc_acc_len), float(
                total_test_recall / test_dataloader_auc_acc_len), float(
                total_test_precision / test_dataloader_auc_acc_len), float(
                total_confounder_loss / train_dataloader_len), float(
                total_ctr_loss / train_dataloader_len)
                ))


