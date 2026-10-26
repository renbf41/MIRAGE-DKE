import torch
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import random


class GNNTrainer:
    def __init__(self, data_loader, retriever_model, gnn_model, lr=0.001):
        self.data_loader = data_loader
        self.retriever_model = retriever_model
        self.gnn_model = gnn_model
        self.optimizer = optim.Adam(self.gnn_model.parameters(), lr=lr)
        self.device = data_loader.device

    def retrieve_mvir_subgraphs(self, uid):
        uid = int(uid)
        distance_tensor = self.update_distance_tensor_inference(uid)

        with torch.no_grad():
            _, top_k_indices_stage1, _ = self.retriever_model(
                self.data_loader.global_profile_embedding,
                uid,
                distance_tensor,
            )

            if top_k_indices_stage1.numel() == 0:
                single_node = torch.tensor([uid], device=self.device)
                return [single_node, single_node, single_node]

            if self.data_loader.multi_view_intent_embedding is None:
                raise ValueError('Multi-view intent embeddings required for Stage 2 retrieval.')

            intents = self.data_loader.multi_view_intent_embedding[uid]
            subgraphs_nodes = []

            _, _, transformed_embeddings = self.retriever_model(
                self.data_loader.global_profile_embedding,
                uid,
                distance_tensor,
                query_embedding=None,
            )

            candidate_embeddings = transformed_embeddings[top_k_indices_stage1]

            for i in range(3):
                intent_embedding = intents[i].unsqueeze(0)
                sims = torch.nn.functional.cosine_similarity(intent_embedding, candidate_embeddings)

                k_2 = len(top_k_indices_stage1) // 2
                if k_2 < 1:
                    k_2 = 1

                _, top_k_2_local_indices = torch.topk(sims, k_2)
                top_k_indices_stage2 = top_k_indices_stage1[top_k_2_local_indices]

                target_user_tensor = torch.tensor([uid], device=self.device)
                final_indices = torch.unique(torch.cat([top_k_indices_stage2, target_user_tensor]))
                subgraphs_nodes.append(final_indices)

            return subgraphs_nodes

    def build_subgraph(self, user_indices):
        user_indices_cpu = user_indices.cpu().numpy()
        sub_ui_graph = self.data_loader.ui_graph_raw[user_indices_cpu, :]
        coo = sub_ui_graph.tocoo()

        local_user_ids = coo.row
        global_user_ids = user_indices[local_user_ids]
        global_item_ids = torch.tensor(coo.col, device=self.device, dtype=torch.long)

        edge_index = torch.stack([global_user_ids, global_item_ids])
        return edge_index

    def update_distance_tensor_inference(self, target_user):
        ui_graph = self.data_loader.ui_graph_tensor
        iu_graph = self.data_loader.iu_graph_tensor

        num_users = ui_graph.size(0)
        distance_tensor = torch.full((num_users,), 5, device=self.device)

        ui_indices = ui_graph.coalesce().indices()
        iu_indices = iu_graph.coalesce().indices()

        first_hop_items = ui_indices[1][ui_indices[0] == target_user]
        first_hop_users = iu_indices[1][iu_indices[0].unsqueeze(1).eq(first_hop_items).any(dim=1)]
        first_hop_users = first_hop_users[first_hop_users != target_user]
        distance_tensor[first_hop_users] = 1

        return distance_tensor

    def train_epoch(self):
        self.gnn_model.train()
        total_loss = 0
        count = 0

        for uid, items in tqdm(self.data_loader.train_data.items()):
            if not items:
                continue

            uid = int(uid)
            subgraph_node_lists = self.retrieve_mvir_subgraphs(uid)
            edge_indices = [self.build_subgraph(nodes) for nodes in subgraph_node_lists]

            global_profile = self.data_loader.global_profile_embedding[uid]
            final_user_emb, item_embs, _ = self.gnn_model(edge_indices, global_profile, uid)

            pos_item = items[-1]
            neg_item = self.sample_negative(items)

            pos_item_emb = item_embs[pos_item]
            neg_item_emb = item_embs[neg_item]

            pos_score = torch.sum(final_user_emb * pos_item_emb)
            neg_score = torch.sum(final_user_emb * neg_item_emb)
            loss = -torch.log(torch.sigmoid(pos_score - neg_score) + 1e-8)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            count += 1

        return total_loss / count

    def evaluate(self, test_data, k_list=[10, 20, 40]):
        self.gnn_model.eval()
        self.retriever_model.eval()

        metrics = {f'Recall@{k}': 0.0 for k in k_list}
        metrics.update({f'NDCG@{k}': 0.0 for k in k_list})
        count = 0

        for uid, items in tqdm(test_data.items(), desc='Evaluating'):
            if not items:
                continue

            uid = int(uid)
            target_item = items[0]

            try:
                subgraph_node_lists = self.retrieve_mvir_subgraphs(uid)
            except Exception:
                continue

            edge_indices = [self.build_subgraph(nodes) for nodes in subgraph_node_lists]

            with torch.no_grad():
                global_profile = self.data_loader.global_profile_embedding[uid]
                final_user_emb, item_embs, _ = self.gnn_model(edge_indices, global_profile, uid)
                scores = torch.matmul(item_embs, final_user_emb)

                if str(uid) in self.data_loader.train_data:
                    train_items = self.data_loader.train_data[str(uid)]
                    scores[train_items] = -float('inf')

                max_k = max(k_list)
                _, top_k_indices = torch.topk(scores, max_k)
                top_k_items = top_k_indices.cpu().numpy()

                for k in k_list:
                    if target_item in top_k_items[:k]:
                        metrics[f'Recall@{k}'] += 1
                        rank = int(np.where(top_k_items[:k] == target_item)[0][0])
                        metrics[f'NDCG@{k}'] += 1.0 / np.log2(rank + 2.0)

                count += 1

        for k in metrics:
            metrics[k] /= count if count > 0 else 1

        return metrics

    def sample_negative(self, items):
        while True:
            neg = random.randint(0, self.data_loader.n_items - 1)
            if neg not in items:
                return neg
