import os
from typing import Dict, List, Set, Tuple
from random import random


def get_data_summary(root: str) -> Tuple[int, int]:
    path = os.path.join(root, "summary.txt")
    with open(path, "r") as f:
        num_users = int(f.readline().strip().split(":")[1])
        num_items = int(f.readline().strip().split(":")[1])
    return num_users, num_items


def load_proc_data(
    root: str,
) -> Tuple[
    List[Tuple[int, int]],
    List[Tuple[int, int]],
    Dict[int, Set[int]],
    Dict[int, Set[int]],
]:
    interactions: List[Tuple[int, int]] = []
    train: List[Tuple[int, int]] = []
    val: Dict[int, Set[int]] = {}
    path = os.path.join(root, "train.tsv")
    with open(path, "r") as f:
        f.readline()
        for line in f:
            user_procid, item_procid = line.strip().split("\t")
            user_procid, item_procid = int(user_procid), int(item_procid)
            interactions.append((user_procid, item_procid))
            if random() > 0.1:
                train.append((user_procid, item_procid))
            elif user_procid not in val:
                val[user_procid] = {}
            val[user_procid].add(item_procid)

    test: Dict[int, Set[int]] = {}
    path = os.path.join(root, "test.tsv")
    with open(path, "r") as f:
        f.readline()
        for line in f:
            user_procid, item_procid = line.strip().split("\t")
            user_procid, item_procid = int(user_procid), int(item_procid)
            interactions.append((user_procid, item_procid))
            if user_procid not in test:
                test[user_procid] = {}
            test[user_procid].add(item_procid)

    return interactions, train, val, test
