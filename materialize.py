import os
import pickle
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F


class ViewMaterializer:
    def __init__(self, data_loader, retriever, top_k):
        self.top_k = top_k
        self.profile = data_loader.global_profile_embedding.detach().float().cpu()
        self.intents = data_loader.multi_view_intent_embedding.detach().float().cpu()
        weight = retriever.linear.weight.detach().float().cpu()
        dim = self.profile.shape[1]
        self.base = self.profile @ weight[:, :dim].T
        self.shift = retriever.distance_encoding.weight.detach().float().cpu() @ weight[:, dim:].T
        self.set_graph(data_loader.ui_graph_raw)

    def set_graph(self, ui_graph):
        self.R = sp.csr_matrix(ui_graph)
        self.RT = self.R.T.tocsr()

    def one_hop(self, uid):
        users = np.unique(self.RT[self.R[uid].indices].indices)
        return users[users != uid]

    def candidates(self, uid):
        hop = self.one_hop(uid)
        if hop.size == 0:
            return hop, None
        transformed = self.base[hop] + self.shift[0]
        sims = F.cosine_similarity(self.profile[uid].unsqueeze(0), transformed, dim=1) * 1.5
        if hop.size >= self.top_k:
            local = torch.topk(sims, self.top_k).indices.numpy()
            hop, transformed = hop[local], transformed[local]
        return hop, transformed

    @torch.no_grad()
    def views(self, uid, single_intent=False):
        cand, transformed = self.candidates(uid)
        if cand.size == 0:
            return [np.array([uid])] * 3
        k = max(len(cand) // 2, 1)
        queries = self.intents[uid][[0, 0, 0]] if single_intent else self.intents[uid]
        out = []
        for query in queries:
            sims = F.cosine_similarity(query.unsqueeze(0), transformed, dim=1)
            local = torch.topk(sims, k).indices.numpy()
            out.append(np.unique(np.concatenate([cand[local], [uid]])))
        return out

    def materialize(self, single_intent=False):
        return {u: self.views(u, single_intent) for u in range(self.R.shape[0]) if self.R[u].nnz}

    def insert(self, views, uid, item, single_intent=False):
        graph = self.R.tolil(copy=True)
        graph[uid, item] = 1
        self.set_graph(graph.tocsr())
        candidate_sets = {int(uid)} | set(self.RT[item].indices.tolist())
        for v in candidate_sets:
            views[v] = self.views(v, single_intent)
        subgraphs = set(candidate_sets)
        for v, user_sets in views.items():
            if any(uid in members for members in user_sets):
                subgraphs.add(v)
        return candidate_sets, subgraphs


def save_views(views, path):
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    pickle.dump(views, open(path, 'wb'))


def load_views(path):
    return pickle.load(open(path, 'rb'))
