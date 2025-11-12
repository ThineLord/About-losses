from __future__ import annotations

import torch
import math


def recall_at_k(scores: torch.Tensor, ground_truth: list[list[int]] | list[set[int]], k: int) -> float:
    """
    计算 Recall@K。

    参数
    - scores: [B, N] 用户对所有物品的打分
    - ground_truth: 长度为 B 的列表，每个元素是该用户的测试正例 item 集合
    - k: 截止位置
    返回 Python float
    """
    device = scores.device
    B, N = scores.shape
    topk = torch.topk(scores, k=min(k, N), dim=1).indices.cpu()
    hits = 0
    total = 0
    for u in range(B):
        gt_set = set(ground_truth[u])
        pred = set(topk[u].tolist())
        hits += len(pred & gt_set)
        total += len(gt_set)
    return float(hits / max(total, 1))


def ndcg_at_k(scores: torch.Tensor, ground_truth: list[list[int]] | list[set[int]], k: int) -> float:
    """
    计算 NDCG@K（binary relevance）。
    """
    B, N = scores.shape
    k = min(k, N)
    idx = torch.topk(scores, k=k, dim=1).indices.cpu()
    import math

    def dcg(rel):
        s = 0.0
        for rank, r in enumerate(rel, start=1):
            s += (2**r - 1) / math.log2(rank + 1)
        return s

    ndcg_sum = 0.0
    for u in range(B):
        gt_set = set(ground_truth[u])
        rel = [1 if i in gt_set else 0 for i in idx[u].tolist()]
        dcg_u = dcg(rel)
        ideal_rel = sorted(rel, reverse=True)
        idcg_u = dcg(ideal_rel)
        ndcg_sum += (dcg_u / idcg_u) if idcg_u > 0 else 0.0
    print("ndcg_sum:", ndcg_sum)
    return float(ndcg_sum / B)


def precision_at_k(scores: torch.Tensor, ground_truth: list[list[int]] | list[set[int]], k: int) -> float:
    """
    计算 Precision@K。

    参数
    - scores: [B, N] 用户对所有物品的打分
    - ground_truth: 长度为 B 的列表，每个元素是该用户的测试正例 item 集合
    - k: 截止位置
    返回 Python float
    """
    device = scores.device
    B, N = scores.shape
    topk = torch.topk(scores, k=min(k, N), dim=1).indices.cpu()
    hits = 0
    for u in range(B):
        gt_set = set(ground_truth[u])
        pred = set(topk[u].tolist())
        hits += len(pred & gt_set)
    return float(hits / (B * k))
