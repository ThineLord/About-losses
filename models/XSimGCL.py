import torch
from torch import nn
from typing import List, Tuple, Optional


class XSimGCL(nn.Module):

    def __init__(
        self,
        num_users: int,
        num_items: int,
        embedding_dim: int = 64,
        user_reg: float = 1e-6,
        item_reg: float = 1e-6,
        num_layers: int = 3,
        edges: List[Tuple[int, int]] = None,
        contrast_weight: float = 0.2,
        contrast_layer: int = 1,
        noise_eps: float = 0.1,
        info_nce_tau: float = 0.2,
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
        self.contrast_weight = contrast_weight
        self.contrast_layer = contrast_layer
        self.noise_eps = noise_eps
        self.info_nce_tau = info_nce_tau

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
        embeddings_list: List[torch.Tensor] = []
        for _ in range(self.num_layers):
            all_embeddings = torch.sparse.mm(self.graph, all_embeddings)
            if self.training:
                noise = torch.randn_like(all_embeddings, device=next(self.parameters()).device)
                noise = nn.functional.normalize(noise, dim=1)
                all_embeddings += torch.sign(all_embeddings) * noise * self.noise_eps
            embeddings_list.append(all_embeddings)
        final_embeddings = torch.stack(embeddings_list, dim=1).mean(dim=1)
        contrast_embeddings = embeddings_list[self.contrast_layer - 1]
        return final_embeddings, contrast_embeddings
    
    @staticmethod
    def _infoNCE(v1: torch.Tensor, v2: torch.Tensor, tau: float) -> torch.Tensor:
        v1 = nn.functional.normalize(v1, dim=1)
        v2 = nn.functional.normalize(v2, dim=1)
        logits = (v1 @ v2.t()) / tau
        return -nn.functional.log_softmax(logits, dim=1).diag().mean()

    def forward(self, user_ids: torch.Tensor, item_ids: torch.Tensor) -> torch.Tensor:
        final_embeddings, _ = self._embed()
        user_embeds = final_embeddings[user_ids]
        item_embeds = final_embeddings[item_ids + self.user_embeddings.num_embeddings]
        return (user_embeds * item_embeds).sum(dim=-1)
    
    @torch.no_grad()
    def full_item_scores(self, user_ids: torch.Tensor) -> torch.Tensor:
        final_embeddings, _ = self._embed()
        user_embeds = final_embeddings[user_ids] # [B, D]
        item_embeds = final_embeddings[self.user_embeddings.num_embeddings:] # [N, D]
        return user_embeds @ item_embeds.t()

    def contrastive_loss(self, user_ids: torch.Tensor, item_ids: torch.Tensor) -> torch.Tensor:
        final_embeddings, contrast_embeddings = self._embed()
        user_final = final_embeddings[user_ids]
        item_final = final_embeddings[item_ids + self.user_embeddings.num_embeddings]
        user_contrast = contrast_embeddings[user_ids]
        item_contrast = contrast_embeddings[item_ids + self.user_embeddings.num_embeddings]
        user_loss = self._infoNCE(user_final, user_contrast, self.info_nce_tau)
        item_loss = self._infoNCE(item_final, item_contrast, self.info_nce_tau)
        return self.contrast_weight * (user_loss + item_loss)

    def l2_regularization(self, user_ids: torch.Tensor, pos_item_ids: torch.Tensor, neg_item_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        final_embeddings, _ = self._embed()
        reg = final_embeddings[user_ids].pow(2).sum() * self.user_reg
        reg = reg + final_embeddings[pos_item_ids + self.user_embeddings.num_embeddings].pow(2).sum() * self.item_reg
        if neg_item_ids is not None:
            reg = reg + final_embeddings[neg_item_ids + self.user_embeddings.num_embeddings].pow(2).sum() * self.item_reg
        return reg
