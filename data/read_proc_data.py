import os
from typing import Dict, List, Set, Tuple
from random import random


def get_data_summary(root: str) -> Tuple[int, int]:
    path = os.path.join(root, "summary.txt")
    with open(path, "r") as f:
        f.readline()
        f.readline()
        num_users = int(f.readline().strip().split(":")[1])
        num_items = int(f.readline().strip().split(":")[1])
    return num_users, num_items


def load_proc_data(root: str) -> Tuple[List[Tuple[int, int]], Dict[int, Set[int]], Dict[int, Set[int]]]:
    train: List[Tuple[int, int]] = []
    val: Dict[int, Set[int]] = {}
    path = os.path.join(root, "train.tsv")
    with open(path, "r") as f:
        f.readline()
        for line in f:
            user_procid, item_procid = line.strip().split("\t")
            user_procid = int(user_procid)
            item_procid = int(item_procid)
            if random() > 0.9:
                train.append((user_procid, item_procid))
            elif user_procid in val:
                val[user_procid].add(item_procid)
            else:
                val[user_procid] = {item_procid}

    test: Dict[int, Set[int]] = {}
    path = os.path.join(root, "test.tsv")
    with open(path, "r") as f:
        f.readline()
        for line in f:
            user_procid, item_procid = line.strip().split("\t")
            user_procid = int(user_procid)
            item_procid = int(item_procid)
            if user_procid in test:
                test[user_procid].add(item_procid)
            else:
                test[user_procid] = {item_procid}

    return train, val, test
