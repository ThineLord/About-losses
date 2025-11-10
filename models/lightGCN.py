import torch
from torch import nn
from typing import List, Tuple, Optional


class LightGCN(nn.Module):

    def __init__(
        self,
        num_users: int,
        num_items: int,
        embedding_dim: int = 64,
        user_reg: float = 1e-6,
        item_reg: float = 1e-6,
        num_layers: int = 3,
        edges: List[Tuple[int, int]] = None,
    ) -> None:
        super().__init__()
        self.user_embeddings = nn.Embedding(num_users, embedding_dim)
        self.item_embeddings = nn.Embedding(num_items, embedding_dim)
        nn.init.normal_(self.user_embeddings.weight, std=0.01)
        nn.init.normal_(self.item_embeddings.weight, std=0.01)
        self.user_reg = user_reg
        self.item_reg = item_reg
        self.num_layers = num_layers
        self._graph = self._build_graph(edges)
        self.register_buffer('graph', self._graph)

    def _build_graph(self, edges: List[Tuple[int, int]]):
        num_users = self.user_embeddings.num_embeddings
        num_items = self.item_embeddings.num_embeddings
        matrix_size = num_users + num_items
        edges = [(u, v + num_users) for u, v in edges] + [(v + num_users, u) for u, v in edges]
        rows, cols = zip(*edges)
        degrees = torch.zeros(matrix_size)
        for u, _ in edges:
            degrees[u] += 1
        sqrt_deg = torch.sqrt(degrees)
        values = [1.0 / (sqrt_deg[u] * sqrt_deg[v]) for u, v in edges]
        return torch.sparse_coo_tensor(
            torch.tensor([rows, cols]),
            torch.tensor(values),
            (matrix_size, matrix_size),
        ).coalesce()

    def _embed(self) -> torch.Tensor:
        all_embeddings = torch.cat([self.user_embeddings.weight, self.item_embeddings.weight], dim=0)
        embeddings_list: List[torch.Tensor] = [all_embeddings]
        for _ in range(self.num_layers):
            all_embeddings = torch.sparse.mm(self.graph, all_embeddings)
            embeddings_list.append(all_embeddings)
        final_embeddings = torch.stack(embeddings_list, dim=1).mean(dim=1)
        return final_embeddings

    def forward(self, user_ids: torch.Tensor, item_ids: torch.Tensor) -> torch.Tensor:
        final_embeddings = self._embed()
        user_embeds = final_embeddings[user_ids]
        item_embeds = final_embeddings[item_ids + self.user_embeddings.num_embeddings]
        return (user_embeds * item_embeds).sum(dim=-1)

    @torch.no_grad()
    def full_item_scores(self, user_ids: torch.Tensor) -> torch.Tensor:
        final_embeddings = self._embed()
        user_embeds = final_embeddings[user_ids] # [B, D]
        item_embeds = final_embeddings[self.user_embeddings.num_embeddings:] # [N, D]
        return user_embeds @ item_embeds.t()

    def l2_regularization(self, user_ids: torch.Tensor, pos_item_ids: torch.Tensor, neg_item_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        final_embeddings = self._embed()
        reg = final_embeddings[user_ids].pow(2).sum() * self.user_reg
        reg = reg + final_embeddings[pos_item_ids + self.user_embeddings.num_embeddings].pow(2).sum() * self.item_reg
        if neg_item_ids is not None:
            reg = reg + final_embeddings[neg_item_ids + self.user_embeddings.num_embeddings].pow(2).sum() * self.item_reg
        return reg
