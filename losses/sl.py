from __future__ import annotations

from typing import Optional

import torch
from torch import nn

from .base import BaseRankingLoss


class SoftmaxLoss(BaseRankingLoss):
    """
    Softmax 基线（List-wise 变体）

    对每个正例 i∈P_u：
    L_i = log sum_j exp((s_uj - s_ui)/tau_d)

    - 若提供 all_item_scores: j 遍历全量物品（排除 i 可选，此处包含 i 影响极小且更稳健）
    - 否则使用 pos+neg 的采样集合近似
    """

    def __init__(self, tau_d: float = 1.0) -> None:
        super().__init__()
        self.tau_d = float(tau_d)

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
        if pos_scores is None and all_item_scores is None:
            raise ValueError("需要 pos_scores 或 all_item_scores 以计算 SoftmaxLoss")

        if pos_scores is None:
            # 从全量打分取出正例分数（pos_item_ids 形状 [B] 或 [B, P]）
            if pos_item_ids.dim() == 1:
                pos = all_item_scores.gather(1, pos_item_ids.unsqueeze(1)).squeeze(1)
            else:
                pos = all_item_scores.gather(1, pos_item_ids)
        else:
            pos = pos_scores

        if all_item_scores is None:
            if neg_scores is None:
                # 仅有正例时，退化：对每个正例分母就是自身 -> log(1)=0
                denom = pos if pos.dim() == 2 else pos.unsqueeze(1)
                lse = torch.log(torch.exp((denom - denom) / self.tau_d).sum(dim=1) + 1e-12)
                return lse.mean()
            # 使用 pos + neg 近似全量
            if pos.dim() == 1:
                pos_e = pos.unsqueeze(1)  # [B, 1]
            else:
                pos_e = pos  # [B, P]
            # 构建候选集合 [B, P+Nn]
            candidates = torch.cat([pos_e, neg_scores], dim=1)
            # 针对每个正例 i 计算 log-sum-exp over candidates
            # s_uj - s_ui: [B, P, P+Nn]
            diff = candidates.unsqueeze(1) - pos_e.unsqueeze(-1)
            lse = torch.logsumexp(diff / self.tau_d, dim=-1)  # [B, P]
            return lse.mean()

        # 全量打分可用：
        if pos.dim() == 1:
            pos_e = pos.unsqueeze(1)  # [B, 1]
        else:
            pos_e = pos  # [B, P]
        diff = all_item_scores.unsqueeze(1) - pos_e.unsqueeze(-1)  # [B, P, N]
        lse = torch.logsumexp(diff / self.tau_d, dim=-1)  # [B, P]
        return lse.mean()

