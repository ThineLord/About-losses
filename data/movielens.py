from __future__ import annotations

import os
import zipfile
from typing import Dict, List, Set, Tuple

import torch
from torch.utils.data import Dataset, DataLoader


ML100K_URL = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"


def ensure_ml100k(root: str) -> str:
    os.makedirs(root, exist_ok=True)
    target_dir = os.path.join(root, "ml-100k")
    if os.path.exists(target_dir):
        return target_dir
    # 留给 scripts/prepare_movielens.py 下载，此处只负责报错提示
    raise FileNotFoundError(
        f"未找到 {target_dir}，请先运行 scripts/prepare_movielens.py 下载并解压 MovieLens100K。"
    )


def load_ml100k_interactions(root: str, threshold: int = 4) -> Tuple[List[Tuple[int, int]], int, int]:
    """
    读取评分数据并转为隐式交互 (rating>=threshold)。返回 (交互列表, 用户数, 物品数)。
    用户/物品 ID 归一化为从 0 开始的索引。
    """
    data_dir = ensure_ml100k(root)
    path = os.path.join(data_dir, "u.data")
    user_map: Dict[int, int] = {}
    item_map: Dict[int, int] = {}
    uid_next = 0
    iid_next = 0
    interactions: List[Tuple[int, int]] = []
    with open(path, "r") as f:
        for line in f:
            u_raw, i_raw, r, ts = line.strip().split("\t")
            r = int(r)
            if r >= threshold:
                u_raw = int(u_raw)
                i_raw = int(i_raw)
                if u_raw not in user_map:
                    user_map[u_raw] = uid_next
                    uid_next += 1
                if i_raw not in item_map:
                    item_map[i_raw] = iid_next
                    iid_next += 1
                interactions.append((user_map[u_raw], item_map[i_raw]))
    return interactions, uid_next, iid_next


def split_leave_one_out(interactions: List[Tuple[int, int]], num_users: int) -> Tuple[List[Tuple[int, int]], Dict[int, Set[int]], Dict[int, Set[int]]]:
    """
    简单的 per-user 留一划分：最后一个作为测试，倒数第二个作为验证，其余作为训练。
    返回 (train, val_dict, test_dict)。val/test 为 {user: {items...}}
    注意：此处假设 interactions 已按时间排序；如果没有时间戳，我们按读入顺序。
    """
    user_hist: List[List[int]] = [[] for _ in range(num_users)]
    for u, i in interactions:
        user_hist[u].append(i)

    train: List[Tuple[int, int]] = []
    val: Dict[int, Set[int]] = {}
    test: Dict[int, Set[int]] = {}
    for u in range(num_users):
        hist = user_hist[u]
        if len(hist) == 0:
            continue
        if len(hist) >= 2:
            test[u] = {hist[-1]}
            val[u] = {hist[-2]}
            for i in hist[:-2]:
                train.append((u, i))
        else:
            test[u] = {hist[-1]}
    return train, val, test


class TripletDataset(Dataset):
    """
    提供 (u, pos_i) 训练样本；负例在训练时使用采样器生成。
    """

    def __init__(self, pairs: List[Tuple[int, int]]):
        self.pairs = pairs

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        u, i = self.pairs[idx]
        return torch.tensor(u, dtype=torch.long), torch.tensor(i, dtype=torch.long)


def build_user_pos_dict(pairs: List[Tuple[int, int]], num_users: int) -> Dict[int, Set[int]]:
    d: Dict[int, Set[int]] = {u: set() for u in range(num_users)}
    for u, i in pairs:
        if u in d:
            d[u].add(i)
        else:
            d[u] = {i}
    return d
