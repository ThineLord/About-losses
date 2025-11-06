from __future__ import annotations

import torch
from torch import nn

from .base import BaseRankingLoss


class BPRLoss(BaseRankingLoss):
    """
    Bayesian Personalized Ranking (BPR) 损失：
    L = - E_{(u,i,j)} log sigma(s_ui - s_uj)

    需要提供 neg_scores 或 (模型, neg_item_ids) 以计算 pairwise 差值。
    若 pos_scores 为 [B, P] 且 neg_scores 为 [B, Nn]，则会进行两两配对并取平均。
    """

    def __init__(self) -> None:
        super().__init__()
        self.sigmoid = nn.Sigmoid()

    def forward(
        self,
        user_ids: torch.Tensor,
        pos_item_ids: torch.Tensor,
        *,
        all_item_scores: torch.Tensor | None = None,
        pos_scores: torch.Tensor | None = None,
        neg_item_ids: torch.Tensor | None = None,
        neg_scores: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if pos_scores is None:
            raise ValueError("BPRLoss 需要提供 pos_scores 或在外部先计算。")
        if neg_scores is None:
            raise ValueError("BPRLoss 需要提供 neg_scores（采样负例）。")

        if pos_scores.dim() == 1:
            pos = pos_scores.unsqueeze(1)  # [B, 1]
        else:
            pos = pos_scores  # [B, P]
        neg = neg_scores  # [B, Nn]

        # 配对并计算差值：
        # s_ui: [B, P, 1], s_uj: [B, 1, Nn] -> diff: [B, P, Nn]
        diff = pos.unsqueeze(-1) - neg.unsqueeze(1)
        loss = -torch.log(torch.sigmoid(diff) + 1e-12).mean()
        return loss

