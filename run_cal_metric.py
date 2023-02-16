import os
from util.args_loader import get_args
import torch
import numpy as np
from tqdm import tqdm
import time
from sklearn.metrics import auc
import json
from util.logger import Logger
import sys
from util import metrics
import random
import datetime
from sklearn.metrics import roc_auc_score
torch.manual_seed(1)
torch.cuda.manual_seed(1)
np.random.seed(1)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

args = get_args()



os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu


def cal_k_lis(p_value, alpha=0.05):

    num, dim = p_value.shape
    range_lis = np.arange(1, dim + 1) * alpha / dim
    comp_arr = np.array([range_lis] * num)
    p_in_sort = np.sort(p_value)

    comp_arr -= p_in_sort

    res = np.where(comp_arr >= 0, 1, 0).sum(1)
    where_id = np.where(res == 0)
    where_ood = np.where(res >= 1)
    res[where_id] = 1
    res[where_ood] = 0

    return res


def read_data(path):
    with open(path) as file_obj:
        p_value_dit = json.load(file_obj)
    return p_value_dit


def get_tp_fp_of_alpha(p_value_in, p_value_ood, alpha):
    id_pre_label = cal_k_lis(p_value_in, alpha)
    ood_pre_label = cal_k_lis(p_value_ood, alpha)
    tp = sum(id_pre_label)
    fp = sum(ood_pre_label)
    return (tp, fp)


def cal_auc(p_value_in, p_value_ood, space=2000):
    tp_fp = []
    for i in tqdm(range(0, space + 1)):
        alpha = 1 / space * i
        tp_fp.append(get_tp_fp_of_alpha(p_value_in, p_value_ood, alpha))

    tp_fp = sorted(tp_fp, reverse=True)
    tp_fp_arr = np.array(tp_fp).T

    tpr = np.concatenate(([[1.], tp_fp_arr[0] / p_value_in.shape[0], [0.]]))
    fpr = np.concatenate([[1.], tp_fp_arr[1] / p_value_ood.shape[0], [0.]])

    return auc(fpr, tpr)


def cal_auc_by_score(known, novel):
    num_k = known.shape[0]
    num_n = novel.shape[0]

    pred_score = np.concatenate((known, novel))
    true_label = np.zeros(num_k+num_n)
    true_label[:num_k] += 1
    auroc = roc_auc_score(true_label, pred_score)
    return auroc


def BH(args, ood_dataset,method,k):
    p_value_in = []
    p_value_ood = []
    model_zoo = vars(args)[f"{args.in_dataset.replace('-','_')}_model_zoo"]
    for model_name in model_zoo:
        if method == 'knn':
            p_value_dir = f"p-value/{args.in_dataset}/{ood_dataset}/{model_name}/p_value{k}.json"
        else:
            p_value_dir = f"p-value/{method}/{args.in_dataset}/{ood_dataset}/{model_name}/p_value.json"
        p_value_dit = read_data(p_value_dir)
        p_value_in.append(p_value_dit['in_p'])
        p_value_ood.append(p_value_dit['out_p'])
    p_value_in = np.array(p_value_in).T
    p_value_ood = np.array(p_value_ood).T
    id_pre_label = cal_k_lis(p_value_in)
    ood_pre_label = cal_k_lis(p_value_ood)

    TPR = sum(id_pre_label) / len(id_pre_label)
    FPR = sum(ood_pre_label) / len(ood_pre_label)
    AUROC = cal_auc(p_value_in, p_value_ood)
    result = {}
    result['TPR'] = TPR
    result['FPR'] = FPR
    result['AUROC'] = AUROC
    print('model_zoo={}'.format(model_zoo))
    return result


method = 'knn'
K = args.K
sys.stdout = Logger(os.path.join('llog', 'log of cal res {} by {}, k={}.txt'.format(args.in_dataset,method,K)))


all_results = []
for ood_dataset in args.out_datasets:

    result = BH(args, ood_dataset,method,K)
    all_results.append(result)
    print('Result by BH when id is {} and ood is {} and k is {}'.format(args.in_dataset, ood_dataset, K))
    print(result)
    print()
metrics.print_results(all_results, args.out_datasets, f'BH+{method}+{K}')











