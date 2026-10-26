import torch
import torch.nn as nn
import torch.nn.functional as F

class DistanceAwareSemanticRetriever(nn.Module):
    def __init__(self, embedding_dim, top_k, num_negatives=256):
        super(DistanceAwareSemanticRetriever, self).__init__()
        self.distance_encoding = nn.Embedding(3, 2)
        self.linear = nn.Linear(embedding_dim + 2, embedding_dim, bias=False)
        self.top_k = top_k
        self.num_negatives = num_negatives

    def forward(self, node_embeddings, source_node_idx, distance_tensor, label_node_idx=None, query_embedding=None, transformed_embeddings=None):
        distances = distance_tensor.long()
        if transformed_embeddings is None:
            distance_indices = torch.clamp(distances - 1, min=0, max=2)
            distance_encodings = self.distance_encoding(distance_indices)
            all_node_embeddings_with_distance = torch.cat((node_embeddings, distance_encodings), dim=1)
            transformed_embeddings = self.linear(all_node_embeddings_with_distance)

        if query_embedding is not None:
            if query_embedding.dim() == 1:
                query_embedding = query_embedding.unsqueeze(0)
            source_embedding = query_embedding
        else:
            source_embedding = node_embeddings[source_node_idx].unsqueeze(0)

        cosine_sim = F.cosine_similarity(source_embedding, transformed_embeddings, dim=1)

        valid_indices = (distances <= 2).nonzero(as_tuple=True)[0]
        valid_cosine_sim = cosine_sim[valid_indices]
        valid_distances = distances[valid_indices]

        distance_weights = torch.ones_like(valid_distances, dtype=torch.float32, device=cosine_sim.device)
        distance_weights[valid_distances == 1] *= 1.5
        distance_weights[valid_distances == 2] *= 2.0
        weighted_similarity = valid_cosine_sim * distance_weights
        
        if len(weighted_similarity) < self.top_k:
            top_k_indices = valid_indices
        else:
            _, top_k_indices_local = torch.topk(weighted_similarity, self.top_k)
            top_k_indices = valid_indices[top_k_indices_local]

        loss = None
        if label_node_idx is not None:
            loss = self.compute_loss_negative_sampling(
                source_embedding.squeeze(0), 
                label_node_idx, 
                transformed_embeddings,
                source_node_idx
            )

        return loss, top_k_indices, transformed_embeddings

    def compute_loss_negative_sampling(self, query_embedding, label_node_idx, transformed_embeddings, source_node_idx):
        num_users = transformed_embeddings.size(0)
        device = transformed_embeddings.device

        neg_candidates = torch.randint(0, num_users, (self.num_negatives + 20,), device=device)

        mask = (neg_candidates != label_node_idx) & (neg_candidates != source_node_idx)
        neg_indices = neg_candidates[mask][:self.num_negatives]

        if len(neg_indices) < self.num_negatives:
            remaining = self.num_negatives - len(neg_indices)
            extra_candidates = torch.randint(0, num_users, (remaining + 10,), device=device)
            mask_extra = (extra_candidates != label_node_idx) & (extra_candidates != source_node_idx)
            neg_indices = torch.cat([neg_indices, extra_candidates[mask_extra]])[:self.num_negatives]

        sample_indices = torch.cat([torch.tensor([label_node_idx], device=device), neg_indices])
        sampled_embeddings = transformed_embeddings[sample_indices]
        scores = torch.mv(sampled_embeddings, query_embedding)

        target = torch.tensor(0, device=device)
        loss = F.cross_entropy(scores.unsqueeze(0), target.unsqueeze(0))

        return loss

    def compute_loss(self, query_embedding, label_node_idx, transformed_embeddings):
        scores = torch.matmul(transformed_embeddings, query_embedding)
        log_probs = F.log_softmax(scores, dim=0)
        loss = -log_probs[label_node_idx]
        return loss

class GNNRecommender(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim, n_layers=3):
        super(GNNRecommender, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.n_layers = n_layers
        
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        
        nn.init.normal_(self.user_embedding.weight, std=0.1)
        nn.init.normal_(self.item_embedding.weight, std=0.1)

    def forward(self, edge_index):
        user_indices = edge_index[0]
        item_indices = edge_index[1] + self.num_users

        src = torch.cat([user_indices, item_indices])
        dst = torch.cat([item_indices, user_indices])
        edge_index_sym = torch.stack([src, dst])

        x = torch.cat([self.user_embedding.weight, self.item_embedding.weight], dim=0)

        row, col = edge_index_sym
        deg = torch.bincount(row, minlength=x.size(0)).float()
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        embs = [x]
        x_current = x

        for _ in range(self.n_layers):
            x_current = self.propagate(x_current, edge_index_sym, norm)
            embs.append(x_current)

        embs = torch.stack(embs, dim=1)
        final_emb = torch.mean(embs, dim=1)

        users_emb_final, items_emb_final = torch.split(final_emb, [self.num_users, self.num_items])
        return users_emb_final, items_emb_final

    def propagate(self, x, edge_index, norm):
        row, col = edge_index

        out = torch.zeros_like(x)
        msg = x[col] * norm.unsqueeze(1)
        out.index_add_(0, row, msg)
        return out

    def score(self, user_emb, item_emb):
        return torch.sum(user_emb * item_emb, dim=1)

class ProfileGatedAggregationModule(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim, intent_dim=3072, n_layers=3):
        super().__init__()
        self.gnn = GNNRecommender(num_users, num_items, embedding_dim, n_layers)

        self.profile_gate = nn.Sequential(
            nn.Linear(intent_dim + embedding_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 1, bias=False)
        )

    def forward(self, subgraphs_indices, user_profile_emb, target_uid):
        subgraph_embeddings = []

        for edge_index in subgraphs_indices:
            u_emb_all, i_emb_all = self.gnn(edge_index)
            subgraph_embeddings.append((u_emb_all, i_emb_all))

        target_u_embs = []
        for u_emb, _ in subgraph_embeddings:
            target_u_embs.append(u_emb[target_uid])

        target_u_embs = torch.stack(target_u_embs)

        if user_profile_emb.dim() == 1:
            user_profile_emb = user_profile_emb.unsqueeze(0)

        profile_expanded = user_profile_emb.expand(3, -1)
        gate_input = torch.cat([profile_expanded, target_u_embs], dim=1)
        scores = self.profile_gate(gate_input).squeeze()
        weights = F.softmax(scores, dim=0)
        final_user_emb = torch.matmul(weights, target_u_embs)
        item_embs_static = self.gnn.item_embedding.weight

        return final_user_emb, item_embs_static, weights


class MIRAGERecommender(ProfileGatedAggregationModule):
    pass
