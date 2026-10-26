import os
import json
import pickle
import torch
from utils import csr_to_adj_list, csr_norm, matrix_to_tensor


class DataLoader:
    def __init__(self, data_path, dataset_name, device='cuda'):
        self.data_path = data_path
        self.dataset_name = dataset_name
        self.device = device
        self.dataset_dir = os.path.join(self.data_path, self.dataset_name)

        self.load_data()
        self.preprocess_graph()
        self.load_embeddings()

    def load_data(self):
        print('Loading data...')
        self.train_data = json.load(open(os.path.join(self.dataset_dir, 'train.json')))
        self.val_data = json.load(open(os.path.join(self.dataset_dir, 'val.json')))
        self.test_data = json.load(open(os.path.join(self.dataset_dir, 'test.json')))

        self.ui_graph_raw = pickle.load(open(os.path.join(self.dataset_dir, 'train_mat'), 'rb'))
        self.n_users = self.ui_graph_raw.shape[0]
        self.n_items = self.ui_graph_raw.shape[1]
        self.iu_graph_raw = self.ui_graph_raw.T

    def preprocess_graph(self):
        print('Preprocessing graph...')
        self.ui_adj_list = csr_to_adj_list(self.ui_graph_raw)
        self.iu_adj_list = csr_to_adj_list(self.iu_graph_raw)

        self.ui_graph_norm = csr_norm(self.ui_graph_raw, mean_flag=True)
        self.iu_graph_norm = csr_norm(self.iu_graph_raw, mean_flag=True)

        self.ui_graph_tensor = matrix_to_tensor(self.ui_graph_norm, self.device)
        self.iu_graph_tensor = matrix_to_tensor(self.iu_graph_norm, self.device)

    def load_embeddings(self):
        print('Loading embeddings...')
        global_profile_emb_path = os.path.join(self.dataset_dir, 'lpg_global_profile_embedding')
        if os.path.exists(global_profile_emb_path):
            self.global_profile_embedding = torch.tensor(pickle.load(open(global_profile_emb_path, 'rb')))
            if self.global_profile_embedding.dtype != torch.float32:
                self.global_profile_embedding = self.global_profile_embedding.float()
            self.embedding_dim = self.global_profile_embedding.shape[1]
            self.global_profile_embedding = self.global_profile_embedding.to(self.device)
        else:
            print('Warning: Global profile embeddings not found. Initializing random embeddings.')
            self.embedding_dim = 3072
            self.global_profile_embedding = torch.randn(self.n_users, self.embedding_dim).to(self.device)

        multi_view_emb_path = os.path.join(self.dataset_dir, 'mvir_multi_view_intent_embedding')
        if os.path.exists(multi_view_emb_path):
            self.multi_view_intent_embedding = torch.tensor(pickle.load(open(multi_view_emb_path, 'rb')))
            if self.multi_view_intent_embedding.dtype != torch.float32:
                self.multi_view_intent_embedding = self.multi_view_intent_embedding.float()
            self.multi_view_intent_embedding = self.multi_view_intent_embedding.to(self.device)
        else:
            print('Warning: Multi-view intent embeddings not found.')
            self.multi_view_intent_embedding = None

    def remove_edge_from_graph(self, uid, item):
        ui_indices = self.ui_graph_tensor.coalesce().indices()
        ui_values = self.ui_graph_tensor.coalesce().values()
        mask_ui = ~((ui_indices[0] == uid) & (ui_indices[1] == item))
        new_ui_indices = ui_indices[:, mask_ui]
        new_ui_values = ui_values[mask_ui]
        new_ui_graph = torch.sparse_coo_tensor(new_ui_indices, new_ui_values, self.ui_graph_tensor.size(), device=self.device)

        iu_indices = self.iu_graph_tensor.coalesce().indices()
        iu_values = self.iu_graph_tensor.coalesce().values()
        mask_iu = ~((iu_indices[0] == item) & (iu_indices[1] == uid))
        new_iu_indices = iu_indices[:, mask_iu]
        new_iu_values = iu_values[mask_iu]
        new_iu_graph = torch.sparse_coo_tensor(new_iu_indices, new_iu_values, self.iu_graph_tensor.size(), device=self.device)

        return new_ui_graph, new_iu_graph
