import numpy as np
import scipy.sparse as sp


class BeyondAccuracy:
    def __init__(self, ui_graph, head_share=0.2):
        self.R = sp.csr_matrix(ui_graph).astype(np.float64)
        self.n_users, self.n_items = self.R.shape
        self.pop = np.asarray(self.R.sum(axis=0)).ravel()
        order = np.argsort(-self.pop, kind='stable')
        share = np.cumsum(self.pop[order]) / self.pop.sum()
        self.is_head = np.zeros(self.n_items, dtype=bool)
        self.is_head[order[:int(np.searchsorted(share, head_share)) + 1]] = True
        self.novelty_item = -np.log2((self.pop + 1.0) / self.n_users)
        self.norms = np.sqrt(np.asarray(self.R.multiply(self.R).sum(axis=0)).ravel()) + 1e-12
        self.pop_rank = np.argsort(-self.pop, kind='stable')

    def primitive(self, uid, k):
        history = self.R[uid].indices
        seen = set(history.tolist())
        most_pop = [i for i in self.pop_rank[:k + len(history)] if i not in seen][:k]
        if history.size == 0:
            return set(most_pop)
        co = np.asarray((self.R[:, history].T @ self.R).todense())
        sims = (co / self.norms[history][:, None] / self.norms[None, :]).sum(axis=0)
        sims[history] = -np.inf
        return set(most_pop) | set(np.argpartition(-sims, k)[:k].tolist())

    def evaluate(self, users, lists, targets, k=20):
        recall, ndcg, novelty, aplt, unexp, ser = [], [], [], [], [], []
        for u, ranked in zip(users, lists):
            ranked = list(ranked[:k])
            gt = targets[int(u)]
            hits = [r for r, i in enumerate(ranked) if i in gt]
            ideal = sum(1.0 / np.log2(r + 2) for r in range(min(len(gt), k)))
            obvious = self.primitive(int(u), k)
            unexpected = [i for i in ranked if i not in obvious]
            recall.append(len(hits) / len(gt))
            ndcg.append(sum(1.0 / np.log2(r + 2) for r in hits) / ideal)
            novelty.append(self.novelty_item[ranked].mean())
            aplt.append((~self.is_head[ranked]).mean())
            unexp.append(len(unexpected) / k)
            ser.append(len([i for i in unexpected if i in gt]) / len(gt))
        exposure = np.bincount(np.asarray([l[:k] for l in lists]).ravel(), minlength=self.n_items).astype(np.float64)
        sorted_exposure = np.sort(exposure)
        n = sorted_exposure.size
        gini = np.sum((2 * np.arange(1, n + 1) - n - 1) * sorted_exposure) / (n * sorted_exposure.sum())
        return {f'Recall@{k}': np.mean(recall), f'NDCG@{k}': np.mean(ndcg), f'Nov@{k}': np.mean(novelty),
                f'APLT@{k}': np.mean(aplt), f'Unexp@{k}': np.mean(unexp), f'Ser@{k}': np.mean(ser),
                f'Cov@{k}': (exposure > 0).mean(), f'Gini@{k}': gini}
