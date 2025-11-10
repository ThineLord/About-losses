import torch
from torch import nn
from typing import Optional


class MatrixFactorization(nn.Module):
    """
    简单 MF（矩阵分解）模型：score(u, i) = <p_u, q_i>

    参数
    - num_users: 用户数
    - num_items: 物品数
    - embedding_dim: 嵌入维度
    - user_reg, item_reg: L2 正则系数
    """

    def __init__(
        self,
        num_users: int,
        num_items: int,
        embedding_dim: int = 64,
        user_reg: float = 1e-6,
        item_reg: float = 1e-6,
    ) -> None:
        super().__init__()
        self.user_embeddings = nn.Embedding(num_users, embedding_dim)
        self.item_embeddings = nn.Embedding(num_items, embedding_dim)
        nn.init.normal_(self.user_embeddings.weight, std=0.01)
        nn.init.normal_(self.item_embeddings.weight, std=0.01)
        self.user_reg = user_reg
        self.item_reg = item_reg

    def forward(self, user_ids: torch.Tensor, item_ids: torch.Tensor) -> torch.Tensor:
        p_u = self.user_embeddings(user_ids)
        q_i = self.item_embeddings(item_ids)
        return (p_u * q_i).sum(dim=-1)

    @torch.no_grad()
    def full_item_scores(self, user_ids: torch.Tensor) -> torch.Tensor:
        """返回 [B, N_items] 的全物品打分矩阵。"""
        p_u = self.user_embeddings(user_ids)  # [B, D]
        q = self.item_embeddings.weight  # [N, D]
        return p_u @ q.t()

    def l2_regularization(self, user_ids: torch.Tensor, pos_item_ids: torch.Tensor, neg_item_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        reg = self.user_embeddings(user_ids).pow(2).sum() * self.user_reg
        reg = reg + self.item_embeddings(pos_item_ids).pow(2).sum() * self.item_reg
        if neg_item_ids is not None:
            reg = reg + self.item_embeddings(neg_item_ids).pow(2).sum() * self.item_reg
        return reg

