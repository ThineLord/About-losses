from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import torch


class BaseRankingLoss(ABC, torch.nn.Module):
    """
    排序损失抽象基类。

    约定输入：
    - user_ids: [B]
    - pos_item_ids: [B] 或 [B, P]，若为 [B] 代表每个用户 1 个正例；若 [B, P] 则多正例
    - all_item_scores: [B, N] 可选，全量物品打分（用于精确或更好地估计分位点/归一化）
    - pos_scores: [B] 或 [B, P]，若已提前计算
    - neg_item_ids: [B, Nn] 可选，采样负例 ID
    - neg_scores: [B, Nn] 可选，采样负例打分

    子类需实现 forward，并尽量兼容上述两种场景：
    1) 有 all_item_scores（全量）
    2) 无 all_item_scores，仅有 sampled negatives（采样）
    """

    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def forward(
        self,
        user_ids: torch.Tensor,
        pos_item_ids: torch.Tensor,
        *,
        all_item_scores: Optional[torch.Tensor] = None,
        pos_scores: Optional[torch.Tensor] = None,
        neg_item_ids: Optional[torch.Tensor] = None,
        neg_scores: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """返回标量损失。"""
        raise NotImplementedError

    @staticmethod
    def estimate_topk_quantile(
        *,
        all_item_scores: Optional[torch.Tensor],  # [B, N]
        pos_scores: torch.Tensor,  # [B] or [B, P]
        neg_scores: Optional[torch.Tensor],  # [B, Nn]
        topk: int,
    ) -> torch.Tensor:
        """
        估计用户层面的 Top-K 分位点 β_u^K。

        策略：
        - 若提供 all_item_scores，直接精确取第 K 大（即 topk 后取最小的那个）。
        - 否则使用采样近似：将 pos_scores 与 neg_scores 合并后取第 K 大，属于 MC 近似。
        返回形状：[B]
        """
        if all_item_scores is not None:
            # 取第 K 大：先 topk 取 K 个，再取这 K 个中的最小值（即第 K 大）
            # all_item_scores: [B, N]
            k = min(topk, all_item_scores.size(1))
            topk_vals, _ = torch.topk(all_item_scores, k, dim=1)
            beta = topk_vals[:, -1]
            return beta

        if neg_scores is None:
            # 仅有正例，无法估计，退化为最大正例分数作为下界
            if pos_scores.dim() == 1:
                return pos_scores
            return pos_scores.max(dim=1).values

        # 采样 MC 近似：合并正负分数后取第 K 大
        # 统一形状到 [B, P]
        if pos_scores.dim() == 1:
            pos = pos_scores.unsqueeze(1)
        else:
            pos = pos_scores
        concat = torch.cat([pos, neg_scores], dim=1)
        k = min(topk, concat.size(1))
        topk_vals, _ = torch.topk(concat, k, dim=1)
        beta = topk_vals[:, -1]
        return beta

