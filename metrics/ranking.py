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
    计算 NDCG@K

    参数
    - scores: [B, N] 用户对所有物品的打分
    - ground_truth: 长度为 B 的列表，每个元素是该用户的测试正例 item 集合
    - k: 截止位置
    返回 Python float
    """
    B, N = scores.shape
    k = min(k, N)
    _, topk_indices = torch.topk(scores, k=k, dim=1)
    discounts = [1 / math.log2(i + 2) for i in range(k)]
    
    def calc_ndcg_single(user_idx):
        gt_set = set(ground_truth[user_idx])
        topk = topk_indices[user_idx].tolist()
        dcg = sum(discounts[i] for i in range(k) if topk[i] in gt_set)
        idcg = sum(discounts[i] for i in range(min(k, len(gt_set))))
        return dcg / idcg if idcg > 0 else 0.0
    
    return sum(calc_ndcg_single(u) for u in range(B)) / B

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
