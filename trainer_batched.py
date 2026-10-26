import random
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
import torch.optim as optim


class MaterializedSubgraphs:
    def __init__(self, ui_graph, views):
        self.users, self.items, self.src, self.dst, self.target = {}, {}, {}, {}, {}
        self.update(ui_graph, views, views.keys())

    def update(self, ui_graph, views, user_ids):
        graph = sp.csr_matrix(ui_graph)
        for u in user_ids:
            for k, members in enumerate(views[u]):
                members = np.asarray(members, dtype=np.int64)
                sub = graph[members].tocoo()
                items, dst = np.unique(sub.col, return_inverse=True)
                key = (u, k)
                self.users[key] = members
                self.items[key] = items.astype(np.int64)
                self.src[key] = sub.row.astype(np.int64)
                self.dst[key] = dst.astype(np.int64)
                self.target[key] = int(np.searchsorted(members, u))

    def batch(self, uids, device):
        user_ids, item_ids, src, dst, target = [], [], [], [], []
        n_user_nodes = sum(len(self.users[(u, k)]) for u in uids for k in range(3))
        offset_u = offset_i = 0
        for u in uids:
            for k in range(3):
                key = (u, k)
                user_ids.append(self.users[key])
                item_ids.append(self.items[key])
                src.append(self.src[key] + offset_u)
                dst.append(self.dst[key] + n_user_nodes + offset_i)
                target.append(self.target[key] + offset_u)
                offset_u += len(self.users[key])
                offset_i += len(self.items[key])
        to_tensor = lambda a: torch.from_numpy(np.concatenate(a)).to(device)
        return to_tensor(user_ids), to_tensor(item_ids), to_tensor(src), to_tensor(dst), torch.tensor(target, device=device)


def batched_forward(model, subgraphs, uids, profile, device):
    gnn = model.gnn
    user_ids, item_ids, src, dst, target = subgraphs.batch(uids, device)
    x = torch.cat([gnn.user_embedding(user_ids), gnn.item_embedding(item_ids)], dim=0)
    deg = torch.bincount(torch.cat([src, dst]), minlength=x.size(0)).float()
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0
    norm = (deg_inv_sqrt[src] * deg_inv_sqrt[dst]).unsqueeze(1)
    total = x[target].clone()
    current = x
    for _ in range(gnn.n_layers):
        out = torch.zeros_like(current)
        out.index_add_(0, src, current[dst] * norm)
        out.index_add_(0, dst, current[src] * norm)
        current = out
        total = total + current[target]
    views = (total / (gnn.n_layers + 1)).view(len(uids), 3, -1)
    gate_profile = profile[torch.as_tensor(uids, device=device)].unsqueeze(1).expand(-1, 3, -1)
    weights = F.softmax(model.profile_gate(torch.cat([gate_profile, views], dim=2)).squeeze(-1), dim=1)
    return torch.einsum('bk,bkd->bd', weights, views), views, weights


class TrainingGraph:
    def __init__(self, ui_graph, device):
        coo = sp.coo_matrix(ui_graph)
        n_users, n_items = coo.shape
        self.n_users = n_users
        self.src = torch.as_tensor(coo.row, dtype=torch.long, device=device)
        self.dst = torch.as_tensor(coo.col + n_users, dtype=torch.long, device=device)
        deg = torch.bincount(torch.cat([self.src, self.dst]), minlength=n_users + n_items).float()
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0
        self.norm = (deg_inv_sqrt[self.src] * deg_inv_sqrt[self.dst]).unsqueeze(1)


def item_representations(model, graph, layers):
    items = model.gnn.item_embedding.weight
    if not layers:
        return items
    x = torch.cat([model.gnn.user_embedding.weight, items], dim=0)
    total, current = x, x
    for _ in range(layers):
        out = torch.zeros_like(current)
        out.index_add_(0, graph.src, current[graph.dst] * graph.norm)
        out.index_add_(0, graph.dst, current[graph.src] * graph.norm)
        current = out
        total = total + current
    return (total / (layers + 1))[graph.n_users:]


class BatchedTrainer:
    def __init__(self, data_loader, model, views, lr=0.005, batch_size=64, negatives=1, item_layers=3,
                 sampling='random', device='cpu'):
        self.data_loader = data_loader
        self.model = model
        self.device = torch.device(device)
        self.batch_size = batch_size
        self.negatives = negatives
        self.item_layers = item_layers
        self.sampling = sampling
        self.train_data = {int(u): v for u, v in data_loader.train_data.items() if v}
        self.subgraphs = MaterializedSubgraphs(data_loader.ui_graph_raw, views)
        self.graph = TrainingGraph(data_loader.ui_graph_raw, self.device) if item_layers else None
        self.profile = data_loader.global_profile_embedding.float().to(self.device)
        self.optimizer = optim.Adam(model.parameters(), lr=lr)

    def sample_negative(self, items):
        while True:
            neg = random.randint(0, self.data_loader.n_items - 1)
            if neg not in items:
                return neg

    def train_epoch(self):
        self.model.train()
        users = [u for u in self.train_data if (u, 0) in self.subgraphs.users]
        random.shuffle(users)
        total, count = 0.0, 0
        for b in range(0, len(users), self.batch_size):
            batch = users[b:b + self.batch_size]
            pos, neg = [], []
            for u in batch:
                items = self.train_data[u]
                pos.append(items[-1] if self.sampling == 'last' else random.choice(items))
                item_set = set(items)
                neg.extend(self.sample_negative(item_set) for _ in range(self.negatives))
            user_emb, _, _ = batched_forward(self.model, self.subgraphs, batch, self.profile, self.device)
            item_emb = item_representations(self.model, self.graph, self.item_layers)
            pos_t = torch.as_tensor(pos, device=self.device)
            neg_t = torch.as_tensor(neg, device=self.device)
            pos_score = (user_emb * item_emb[pos_t]).sum(1, keepdim=True)
            neg_score = torch.einsum('bd,bnd->bn', user_emb, item_emb[neg_t].view(len(batch), self.negatives, -1))
            loss = -torch.log(torch.sigmoid(pos_score - neg_score) + 1e-8).mean()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            total += loss.item()
            count += 1
        return total / max(count, 1)

    @torch.no_grad()
    def scores(self, users):
        self.model.eval()
        item_emb = item_representations(self.model, self.graph, self.item_layers)
        out = []
        for b in range(0, len(users), 256):
            batch = list(users[b:b + 256])
            user_emb, _, _ = batched_forward(self.model, self.subgraphs, batch, self.profile, self.device)
            score = user_emb @ item_emb.T
            for j, u in enumerate(batch):
                history = self.train_data.get(u, [])
                if history:
                    score[j, history] = -float('inf')
            out.append(score.cpu())
        return torch.cat(out).numpy()

    def evaluate(self, test_data, k_list=(10, 20, 50)):
        targets = {}
        for u, items in test_data.items():
            valid = {int(i) for i in items if 0 <= int(i) < self.data_loader.n_items}
            if valid and (int(u), 0) in self.subgraphs.users:
                targets[int(u)] = valid
        users = sorted(targets)
        top = np.argsort(-self.scores(users), axis=1)[:, :max(k_list)]
        metrics = {f'Recall@{k}': 0.0 for k in k_list}
        metrics.update({f'NDCG@{k}': 0.0 for k in k_list})
        for j, u in enumerate(users):
            gt = targets[u]
            for k in k_list:
                hits = [r for r, i in enumerate(top[j, :k]) if i in gt]
                metrics[f'Recall@{k}'] += len(hits) / len(gt)
                ideal = sum(1.0 / np.log2(r + 2) for r in range(min(len(gt), k)))
                metrics[f'NDCG@{k}'] += sum(1.0 / np.log2(r + 2) for r in hits) / ideal
        return {name: value / len(users) for name, value in metrics.items()}
