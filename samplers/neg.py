from __future__ import annotations

import random
from typing import Dict, List, Set, Tuple

import torch


class UniformNegativeSampler:
    """
    简单均匀负采样器：从 [0, num_items) 中排除用户已交互项，均匀采样 Nn 个负例。
    """

    def __init__(self, num_items: int, user_to_pos_items: Dict[int, Set[int]], seed: int = 42) -> None:
        self.num_items = int(num_items)
        self.user_to_pos_items = user_to_pos_items
        self.rng = random.Random(seed)

    def sample(self, user_ids: torch.Tensor, num_negatives: int) -> torch.Tensor:
        batch_ids = user_ids.tolist()
        neg_lists: List[List[int]] = []
        for u in batch_ids:
            pos = self.user_to_pos_items.get(u, set())
            neg = []
            while len(neg) < num_negatives:
                x = self.rng.randrange(self.num_items)
                if x not in pos:
                    neg.append(x)
            neg_lists.append(neg)
        return torch.tensor(neg_lists, dtype=torch.long, device=user_ids.device)

