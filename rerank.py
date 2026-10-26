import numpy as np


def intent_item_affinity(intent_emb, item_emb, gamma=1.0, cap=0.1):
    sims = intent_emb @ item_emb.T
    low, high = sims.min(axis=1, keepdims=True), sims.max(axis=1, keepdims=True)
    return cap * ((sims - low) / (high - low + 1e-12)) ** gamma


def intent_coverage_greedy(relevance, affinity, weights, lam, n):
    relevance = (relevance - relevance.min()) / (relevance.max() - relevance.min() + 1e-12)
    remaining = np.ones(affinity.shape[0])
    scale = 1.0 / ((weights[:, None] * affinity).sum(0).max() + 1e-12)
    taken = np.zeros(len(relevance), dtype=bool)
    selected, intents = [], []
    for _ in range(n):
        coverage = (weights * remaining)[:, None] * affinity
        gain = (1 - lam) * relevance + lam * scale * coverage.sum(0)
        gain[taken] = -np.inf
        j = int(np.argmax(gain))
        selected.append(j)
        intents.append(int(np.argmax(coverage[:, j])))
        taken[j] = True
        remaining *= 1 - affinity[:, j]
    return selected, intents


def rerank(scores, intent_emb, item_emb, lam=0.5, n=20, pool=200, gamma=1.0, cap=0.1, weights=None):
    weights = np.full(intent_emb.shape[0], 1.0 / intent_emb.shape[0]) if weights is None else np.asarray(weights, dtype=float)
    candidates = np.argsort(-scores, kind='stable')[:pool]
    affinity = intent_item_affinity(intent_emb, item_emb[candidates], gamma, cap)
    selected, intents = intent_coverage_greedy(scores[candidates], affinity, weights, lam, n)
    return candidates[selected], np.asarray(intents)


def rerank_all(score_matrix, intent_emb, item_emb, **kwargs):
    lists, owners = [], []
    for j in range(score_matrix.shape[0]):
        items, intents = rerank(score_matrix[j], intent_emb[j], item_emb, **kwargs)
        lists.append(items)
        owners.append(intents)
    return np.stack(lists), np.stack(owners)
