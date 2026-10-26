import numpy as np
import scipy.sparse as sp
import torch
from collections import deque, defaultdict
import pickle
import os
import json

def csr_to_adj_list(csr_mat):
    adj_list = defaultdict(list)
    coo_mat = csr_mat.tocoo()
    for u, v in zip(coo_mat.row, coo_mat.col):
        adj_list[u].append(v)
    return adj_list

def csr_norm(csr_mat, mean_flag=False):
    rowsum = np.array(csr_mat.sum(1))
    rowsum = np.power(rowsum+1e-8, -0.5).flatten()
    rowsum[np.isinf(rowsum)] = 0.
    rowsum_diag = sp.diags(rowsum)
    colsum = np.array(csr_mat.sum(0))
    colsum = np.power(colsum+1e-8, -0.5).flatten()
    colsum[np.isinf(colsum)] = 0.
    colsum_diag = sp.diags(colsum)
    if mean_flag == False:
        return rowsum_diag*csr_mat*colsum_diag
    else:
        return rowsum_diag*csr_mat

def matrix_to_tensor(cur_matrix, device='cuda'):
    if type(cur_matrix) != sp.coo_matrix:
        cur_matrix = cur_matrix.tocoo()
    indices = torch.from_numpy(np.vstack((cur_matrix.row, cur_matrix.col)).astype(np.int64))
    values = torch.from_numpy(cur_matrix.data)
    shape = torch.Size(cur_matrix.shape)
    return torch.sparse.FloatTensor(indices, values, shape).to(torch.float32).to(device)

def bfs_shortest_path(ui_adj_list, iu_adj_list, start_user, target_item):
    queue = deque([(start_user, [start_user])])
    visited_users = set()
    visited_items = set()
    while queue:
        current_user, path = queue.popleft()
        if current_user in visited_users:
            continue
        visited_users.add(current_user)

        for item in ui_adj_list.get(current_user, []):
            if item == target_item:
                return path + [item]
            
            if item not in visited_items:
                visited_items.add(item)
                for user in iu_adj_list.get(item, []):
                    if user not in visited_users:
                        queue.append((user, path + [item, user]))    
    return None

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def save_results(model_name, dataset_name, metrics, filename='RESULT.md'):
    file_exists = os.path.exists(filename)
    with open(filename, 'a') as f:
        if not file_exists:
            f.write("| Model | Dataset | Recall@10 | Recall@20 | Recall@40 | NDCG@10 | NDCG@20 | NDCG@40 |\n")
            f.write("|---|---|---|---|---|---|---|---|\n")
        
        r10 = metrics.get('Recall@10', 0.0)
        r20 = metrics.get('Recall@20', 0.0)
        r40 = metrics.get('Recall@40', 0.0)
        n10 = metrics.get('NDCG@10', 0.0)
        n20 = metrics.get('NDCG@20', 0.0)
        n40 = metrics.get('NDCG@40', 0.0)
        
        f.write(f"| {model_name} | {dataset_name} | {r10:.4f} | {r20:.4f} | {r40:.4f} | {n10:.4f} | {n20:.4f} | {n40:.4f} |\n")
