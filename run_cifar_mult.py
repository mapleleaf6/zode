import os
import time
import datetime
from util.args_loader import get_args
from util import metrics
import torch
import faiss
import numpy as np
from util.logger import Logger
import json
import sys
torch.manual_seed(1)
torch.cuda.manual_seed(1)
np.random.seed(1)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

args = get_args()


os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu


def get_one_model_p_value(ftrain, ftest, food, model_name, ood_dataset, K):

    #################### KNN score OOD detection #################

    index = faiss.IndexFlatL2(ftrain.shape[1])
    index.add(ftrain)


    D, _ = index.search(ftest, K)
    scores_in = -D[:,-1]
    all_results = []
    D, _ = index.search(food, K)
    scores_ood_test = -D[:,-1]
    results, in_p, out_p = metrics.cal_p_value(scores_in, scores_ood_test)
    all_results.append(results)

    return in_p, out_p, results  # array([num_in,]), array([num_out,])


def reload_feat(path, model_name):

    if 'train' in path or 'val' in path:
        feat_log, score_log, label_log = np.load(path, allow_pickle=True)
    else:
        feat_log, score_log = np.load(path, allow_pickle=True)
    feat_log, score_log = feat_log.T.astype(np.float32), score_log.T.astype(np.float32)
    class_num = score_log.shape[1]

    normalizer = lambda x: x / (np.linalg.norm(x, ord=2, axis=-1, keepdims=True) + 1e-10)
    if model_name.startswith('resnet'):
        model_num = int(''.join([x for x in model_name if x.isdigit()]))
        dim = feat_log.shape[1]
        if model_num < 50:
            # print(model_name, model_num, dim, 'if')
            prepos_feat = lambda x: np.ascontiguousarray(normalizer(x[:, range(dim - 512, dim)]))  # Last Layer only
        else:
            # print(model_name, model_num, dim, 'else')
            prepos_feat = lambda x: np.ascontiguousarray(normalizer(x[:, range(dim - 2048, dim)]))  # Last Layer only
    elif model_name.startswith('densenet'):
        dim = feat_log.shape[1]
        # print(model_name, dim)
        prepos_feat = lambda x: np.ascontiguousarray(normalizer(x[:, range(dim - 342, dim)]))  # Last Layer only
    pos_feat_log = prepos_feat(feat_log)   # [num, 512 or 2048 or 342]
    return pos_feat_log


def save_file(data, file_name):
    with open(file_name, 'w') as file_obj:
        json.dump(data, file_obj)

FORCE_RUN = True
K=args.K
sys.stdout = Logger(os.path.join('llog', 'log of mult-test by knn when k={}.txt'.format(K)))
model_zoo = vars(args)[f"{args.in_dataset.replace('-', '_')}_model_zoo"]
for model_name in model_zoo:
    flag = 1
    all_results=[]
    for ood_dataset in args.out_datasets:
        p_value_dir = f"p-value/{args.in_dataset}/{ood_dataset}/{model_name}/p_value{K}.json"
        if FORCE_RUN or not os.path.exists(p_value_dir):
            if not os.path.exists(os.path.dirname(p_value_dir)):
                os.makedirs(os.path.dirname(p_value_dir))
            if flag:
                cache_name = f"cache/{args.in_dataset}/{args.in_dataset}_train_{model_name}_in_alllayers.npy"
                ftrain = reload_feat(cache_name, model_name)

                cache_name = f"cache/{args.in_dataset}/{args.in_dataset}_val_{model_name}_in_alllayers.npy"
                ftest = reload_feat(cache_name, model_name)
                p_dit = {}
                flag = 0
            cache_name = f"cache/{args.in_dataset}/{ood_dataset}/{model_name}/{ood_dataset}vs{args.in_dataset}_{model_name}_out_alllayers.npy"
            food = reload_feat(cache_name, model_name)
            in_p, out_p, results = get_one_model_p_value(ftrain, ftest, food, model_name, ood_dataset, K=K)
            all_results.append(results)
            p_dit['in_p'] = in_p.tolist()
            p_dit['out_p'] = out_p.tolist()
            save_file(p_dit, p_value_dir)
    print(f'when model is {model_name},k={K},the result is:')
    metrics.print_all_results(all_results, args.out_datasets, 'knn')
    print()



