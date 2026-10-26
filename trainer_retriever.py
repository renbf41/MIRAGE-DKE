import torch
import torch.optim as optim
from utils import bfs_shortest_path
import os
import pickle
from tqdm import tqdm


class DistanceAwareRetrieverTrainer:
    def __init__(self, data_loader, model, lr=0.001, top_k=100):
        self.data_loader = data_loader
        self.model = model
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.top_k = top_k
        self.device = data_loader.device
        self.max_distance = 5
        self._training_samples_cache = None
        self._precompute_training_samples()

    def _precompute_training_samples(self):
        cache_path = os.path.join(self.data_loader.dataset_dir, 'distance_aware_retriever_training_cache.pkl')

        if os.path.exists(cache_path):
            print('Loading pre-computed training samples from cache...')
            with open(cache_path, 'rb') as f:
                self._training_samples_cache = pickle.load(f)
            print(f"Loaded {len(self._training_samples_cache)} cached training samples.")
            return

        print('Pre-computing training samples (this only happens once)...')
        cache = {}

        ui_adj_list = self.data_loader.ui_adj_list
        iu_adj_list = self.data_loader.iu_adj_list

        for uid, items in tqdm(self.data_loader.train_data.items(), desc='Pre-computing samples'):
            uid = int(uid)
            if len(items) == 0:
                continue

            target_item = items[-1]
            removed_ui = False
            removed_iu = False

            if target_item in ui_adj_list[uid]:
                ui_adj_list[uid].remove(target_item)
                removed_ui = True
            if uid in iu_adj_list[target_item]:
                iu_adj_list[target_item].remove(uid)
                removed_iu = True

            shortest_path = bfs_shortest_path(ui_adj_list, iu_adj_list, uid, target_item)
            distance_dict = self._compute_distance_dict_cpu(ui_adj_list, iu_adj_list, uid)

            if removed_ui:
                ui_adj_list[uid].append(target_item)
            if removed_iu:
                iu_adj_list[target_item].append(uid)

            if shortest_path:
                label_node_idx = shortest_path[-2]
                cache[uid] = (uid, label_node_idx, distance_dict)

        self._training_samples_cache = cache

        print(f"Saving {len(cache)} training samples to cache...")
        with open(cache_path, 'wb') as f:
            pickle.dump(cache, f)
        print('Pre-computation complete.')

    def _compute_distance_dict_cpu(self, ui_adj_list, iu_adj_list, target_user):
        distance_dict = {}

        first_hop_items = ui_adj_list.get(target_user, [])
        first_hop_users = set()
        for item in first_hop_items:
            for user in iu_adj_list.get(item, []):
                if user != target_user:
                    first_hop_users.add(user)

        for user_id in first_hop_users:
            distance_dict[user_id] = 1

        second_hop_users = set()
        for user_id in first_hop_users:
            for item in ui_adj_list.get(user_id, []):
                for second_user in iu_adj_list.get(item, []):
                    if second_user != target_user and second_user not in first_hop_users:
                        second_hop_users.add(second_user)

        for user_id in second_hop_users:
            distance_dict[user_id] = 2

        return distance_dict

    def _distance_dict_to_tensor(self, distance_dict, num_users):
        distance_tensor = torch.full((num_users,), self.max_distance, device=self.device)
        if distance_dict:
            users = list(distance_dict.keys())
            distances = list(distance_dict.values())
            distance_tensor[users] = torch.tensor(distances, device=self.device, dtype=distance_tensor.dtype)
        return distance_tensor

    def update_distance_tensor(self, ui_graph, iu_graph, target_user):
        num_users = ui_graph.size(0)
        distance_tensor = torch.full((num_users,), self.max_distance, device=self.device)

        ui_indices = ui_graph.coalesce().indices()
        iu_indices = iu_graph.coalesce().indices()

        first_hop_items = ui_indices[1][ui_indices[0] == target_user]
        first_hop_users = iu_indices[1][torch.isin(iu_indices[0], first_hop_items)]
        first_hop_users = first_hop_users[first_hop_users != target_user]
        distance_tensor[first_hop_users] = 1

        second_hop_items = ui_indices[1][torch.isin(ui_indices[0], first_hop_users)]
        second_hop_users = iu_indices[1][torch.isin(iu_indices[0], second_hop_items)]

        mask = ~torch.isin(second_hop_users, first_hop_users)
        second_hop_users = second_hop_users[mask]
        second_hop_users = second_hop_users[second_hop_users != target_user]
        distance_tensor[second_hop_users] = 2

        return distance_tensor

    def get_training_sample(self, uid, train_items):
        uid = int(uid)
        if len(train_items) == 0:
            return None, None, None

        target_item = train_items[-1]
        temp_ui_graph, temp_iu_graph = self.data_loader.remove_edge_from_graph(uid, target_item)
        distance_tensor = self.update_distance_tensor(temp_ui_graph, temp_iu_graph, uid)

        ui_adj_list = self.data_loader.ui_adj_list
        iu_adj_list = self.data_loader.iu_adj_list

        removed_ui = False
        removed_iu = False

        if target_item in ui_adj_list[uid]:
            ui_adj_list[uid].remove(target_item)
            removed_ui = True
        if uid in iu_adj_list[target_item]:
            iu_adj_list[target_item].remove(uid)
            removed_iu = True

        shortest_path = bfs_shortest_path(ui_adj_list, iu_adj_list, uid, target_item)

        if removed_ui:
            ui_adj_list[uid].append(target_item)
        if removed_iu:
            iu_adj_list[target_item].append(uid)

        if shortest_path:
            label_node_idx = shortest_path[-2]
            return uid, label_node_idx, distance_tensor

        return None, None, None

    def train_epoch(self):
        self.model.train()
        total_loss = 0
        recall_count = 0
        count = 0

        num_users = self.data_loader.n_users

        for uid, items in tqdm(self.data_loader.train_data.items()):
            uid = int(uid)

            if self._training_samples_cache and uid in self._training_samples_cache:
                source_node, label_node, distance_dict = self._training_samples_cache[uid]
                distance_tensor = self._distance_dict_to_tensor(distance_dict, num_users)
            else:
                source_node, label_node, distance_tensor = self.get_training_sample(uid, items)

            if source_node is None:
                continue

            self.optimizer.zero_grad()

            loss_global, _, transformed_embeddings = self.model(
                self.data_loader.global_profile_embedding,
                source_node,
                distance_tensor,
                label_node,
                query_embedding=None,
            )

            loss_multi_view = 0
            top_k_indices_mvir = torch.tensor([], device=self.device)

            if self.data_loader.multi_view_intent_embedding is not None:
                intents = self.data_loader.multi_view_intent_embedding[source_node]

                losses = []
                all_indices = []
                for i in range(3):
                    intent_emb = intents[i].unsqueeze(0)
                    loss_i, idx_i, _ = self.model(
                        self.data_loader.global_profile_embedding,
                        source_node,
                        distance_tensor,
                        label_node,
                        query_embedding=intent_emb,
                        transformed_embeddings=transformed_embeddings,
                    )
                    losses.append(loss_i)
                    all_indices.append(idx_i)

                loss_multi_view = torch.min(torch.stack(losses))
                top_k_indices_mvir = torch.unique(torch.cat(all_indices))
            else:
                raise ValueError('Multi-view intent embeddings not found. Please run generate_embeddings.py first.')

            loss = loss_global + loss_multi_view
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            if label_node in top_k_indices_mvir:
                recall_count += 1
            count += 1

        if count == 0:
            return 0, 0
        return total_loss / count, recall_count / count

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)
