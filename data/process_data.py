# Prepare the raw dataset in the directory as follows first:
# data
# ├── process_data.ipynb
# └── raw
#     ├── loc-gowalla_totalCheckins.txt.gz
#     ├── ratings_Books.csv
#     ├── ratings_Electronics.csv
#     └── ratings_Health_and_Personal_Care.csv

from typing import (
    List,
    Tuple,
    Dict,
)
import csv
import gzip
import os
import random
from collections import Counter


def filter_N_cores(
    last_epoch: List[Tuple[int, int]], core: int
) -> List[Tuple[int, int]]:
    """
    Recursively filter interations by N-core
    """
    while last_epoch:
        user_cnt = Counter([x[0] for x in last_epoch])
        item_cnt = Counter([x[1] for x in last_epoch])
        filtered = [
            (u, i) for u, i in last_epoch if user_cnt[u] >= core and item_cnt[i] >= core
        ]
        if len(filtered) == len(last_epoch):
            return filtered
            # return sorted(filtered, key=lambda x: (x[0], x[1]))
        last_epoch = filtered
    return []


def remap(orig_ids: List[int]) -> Dict[int, int]:
    """
    Remap original ids to new ids
    """
    return dict(zip(list(set(orig_ids)), list(range(len(orig_ids)))))


def write_tsv(data: List[Tuple[int, int]], file_path: str, header: List[str]) -> None:
    with open(file_path, "w") as f:
        f.write("\t".join(header) + "\n")
        for x in data:
            f.write("\t".join([str(y) for y in x]) + "\n")


def write_dataset(dataset_name: str, data: List[Tuple[int, int]]) -> None:
    """
    Write processed dataset into folder
    """
    folder = "proc_" + dataset_name
    user_map = remap([x[0] for x in data])
    item_map = remap([x[1] for x in data])

    user_interactions = {}
    for u, i in data:
        u, i = user_map[u], item_map[i]
        if u not in user_interactions:
            user_interactions[u] = []
        user_interactions[u].append(i)

    train_data, test_data = [], []
    for user, items in user_interactions.items():
        random.shuffle(items)
        train_size = int(len(items) * 0.8)
        train_data.extend([(user, i) for i in items[:train_size]])
        test_data.extend([(user, i) for i in items[train_size:]])
    train_items = set([x[1] for x in train_data])
    test_data = [(u, i) for u, i in test_data if i in train_items]

    with open(f"{folder}/summary.txt", "w") as f:
        f.write(f"Number of users: {len(user_map)}\n")
        f.write(f"Number of users: {len(item_map)}\n")
    write_tsv(
        sorted(train_data, key=lambda x: (x[0], x[1])),
        f"{folder}/train.tsv",
        ["user_proc_id", "item_proc_id"],
    )
    write_tsv(
        sorted(test_data, key=lambda x: (x[0], x[1])),
        f"{folder}/test.tsv",
        ["user_proc_id", "item_proc_id"],
    )


def process_amazon(
    dataset_name: str, file_path: str, core: int, star_threshold: int
) -> None:
    """
    Read amazon dataset, filter with core and star threshold setting, remap user and item, write to tsv
    """
    print(f"Processing dataset {dataset_name}")
    with open(file_path, "r") as f:
        reader = csv.reader(f)
        # Filter duplicated or reviews with star < 3
        reviews = list(
            set(
                [
                    (line[0], line[1])
                    for line in reader
                    if float(line[2]) >= star_threshold
                ]
            )
        )

    print("Filtering dataset")
    filtered = filter_N_cores(reviews, core)
    try:
        os.mkdir(f"./proc_{dataset_name}")
    except:
        pass
    print("Writing to csv")
    write_dataset(dataset_name, filtered)
    print("Writing complete\n")


def process_gowalla(
    dataset_name: str, file_path: str, core: int, star_threshold: int
) -> None:
    """
    Read gowalla dataset, filter with core and star threshold setting, remap user and item, write to tsv
    """
    print(f"Processing dataset {dataset_name}")
    with gzip.open(file_path, "rt") as f:
        data = f.readlines()
    data = [x.strip().split("\t") for x in data]
    data = [(int(x[0]), int(x[4])) for x in data]
    # Filter duplicated reviews
    data = list(set(data))

    print("Filtering dataset")
    filtered = filter_N_cores(data, core)
    try:
        os.mkdir(f"./proc_{dataset_name}")
    except:
        pass
    print("Writing to csv")
    write_dataset(dataset_name, filtered)
    print("Writing complete\n")


if __name__ == "__main__":
    random.seed(5331)

    core = 10
    star_threshold = 3

    print(f"Processing datasets with {core} cores and star threshold {star_threshold}")
    process_amazon(
        "Amazon2014-Book",
        "raw/ratings_Books.csv",
        core,
        star_threshold,
    )
    process_amazon(
        "Amazon2014-Electronic",
        "raw/ratings_Electronics.csv",
        core,
        star_threshold,
    )
    process_amazon(
        "Amazon2014-Health",
        "raw/ratings_Health_and_Personal_Care.csv",
        core,
        star_threshold,
    )
    process_gowalla(
        "Gowalla",
        "raw/loc-gowalla_totalCheckins.txt.gz",
        core,
        star_threshold,
    )
