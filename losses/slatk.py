from __future__ import annotations

from typing import Optional

import torch
from torch import nn

from .base import BaseRankingLoss


class SoftmaxLossAtK(BaseRankingLoss):
    """
    SoftmaxLoss@K (SLatK)

    公式（简化实现）：
    L_{SL@K}(u) = \sum_{i\in P_u} sigma_w(s_{ui}-beta_u^K) * log \sum_j exp((s_{uj}-s_{ui})/tau_d)

    - beta_u^K：用户 u 的 Top-K 分位点，优先使用 all_item_scores 精确计算；否则用 (pos+neg) 采样集合做 MC 近似
    - sigma_w(x) = sigmoid(x / tau_w)
    - j 的范围：若有 all_item_scores，用全量物品；否则用采样集合 (pos+neg)
    """

    def __init__(self, topk: int, tau_d: float = 1.0, tau_w: float = 1.0) -> None:
        super().__init__()
        self.topk = int(topk)
        self.tau_d = float(tau_d)
        self.tau_w = float(tau_w)

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
            raise ValueError("需要 pos_scores 或 all_item_scores 以计算 SLatK")

        # 获得 pos_scores
        if pos_scores is None:
            if pos_item_ids.dim() == 1:
                pos = all_item_scores.gather(1, pos_item_ids.unsqueeze(1)).squeeze(1)
            else:
                pos = all_item_scores.gather(1, pos_item_ids)
        else:
            pos = pos_scores

        # 估计 beta_u^K
        beta = BaseRankingLoss.estimate_topk_quantile(
            all_item_scores=all_item_scores,
            pos_scores=pos,
            neg_scores=neg_scores,
            topk=self.topk,
        )  # [B]

        # 权重 sigma_w(s_ui - beta_u^K)
        if pos.dim() == 1:
            pos_e = pos.unsqueeze(1)  # [B, 1]
        else:
            pos_e = pos  # [B, P]
        weight = torch.sigmoid((pos_e - beta.unsqueeze(1)) / max(self.tau_w, 1e-12))  # [B, P]

        # 计算 Softmax 项
        if all_item_scores is None:
            if neg_scores is None:
                denom = pos_e  # 仅自身，logsumexp(0)=0
                lse = torch.logsumexp((denom - pos_e) / self.tau_d, dim=-1)
            else:
                candidates = torch.cat([pos_e, neg_scores], dim=1)  # [B, P+Nn]
                diff = candidates.unsqueeze(1) - pos_e.unsqueeze(-1)  # [B, P, P+Nn]
                lse = torch.logsumexp(diff / self.tau_d, dim=-1)  # [B, P]
        else:
            diff = all_item_scores.unsqueeze(1) - pos_e.unsqueeze(-1)  # [B, P, N]
            lse = torch.logsumexp(diff / self.tau_d, dim=-1)  # [B, P]

        loss = (weight * lse).mean()
        return loss

