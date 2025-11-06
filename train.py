from __future__ import annotations

import argparse
import os
import random
from typing import Dict, Set

import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from torch.utils.data import DataLoader

from models import MatrixFactorization
from losses import BPRLoss, SoftmaxLoss, SoftmaxLossAtK
from metrics import recall_at_k, ndcg_at_k
from data.movielens import (
    TripletDataset,
    load_ml100k_interactions,
    split_leave_one_out,
    build_user_pos_dict,
)
from samplers import UniformNegativeSampler


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_config(cfg_path: str) -> dict:
    with open(cfg_path, "r") as f:
        return yaml.safe_load(f)


def build_loss(name: str, params: dict) -> nn.Module:
    name = name.lower()
    if name == "bpr":
        return BPRLoss()
    if name == "sl":
        return SoftmaxLoss(tau_d=float(params.get("tau_d", 1.0)))
    if name == "slatk":
        return SoftmaxLossAtK(
            topk=int(params.get("topk", 10)),
            tau_d=float(params.get("tau_d", 1.0)),
            tau_w=float(params.get("tau_w", 1.0)),
        )
    raise ValueError(f"Unknown loss: {name}")


def evaluate(model: MatrixFactorization, user_to_eval_pos: Dict[int, Set[int]], k: int, device: torch.device) -> tuple[float, float]:
    model.eval()
    with torch.no_grad():
        users = sorted(user_to_eval_pos.keys())
        if not users:
            return 0.0, 0.0
        user_ids = torch.tensor(users, dtype=torch.long, device=device)
        scores = model.full_item_scores(user_ids)
        gt = [list(user_to_eval_pos[u]) for u in users]
        rec = recall_at_k(scores, gt, k)
        ndcg = ndcg_at_k(scores, gt, k)
    return rec, ndcg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="cfgs/default.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(int(cfg["train"].get("seed", 42)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 数据
    interactions, num_users, num_items = load_ml100k_interactions(cfg["dataset"]["root"], cfg["dataset"]["threshold"])
    train_pairs, val_dict, test_dict = split_leave_one_out(interactions, num_users)
    train_ds = TripletDataset(train_pairs)
    train_loader = DataLoader(
        train_ds,
        batch_size=int(cfg["dataset"]["batch_size"]),
        shuffle=True,
        num_workers=int(cfg["dataset"].get("num_workers", 0)),
        drop_last=False,
    )
    user_pos_dict = build_user_pos_dict(train_pairs, num_users)
    neg_sampler = UniformNegativeSampler(num_items, user_pos_dict)

    # 模型
    emb_dim = int(cfg["model"]["embedding_dim"])
    model = MatrixFactorization(num_users, num_items, emb_dim, cfg["model"]["user_reg"], cfg["model"]["item_reg"]).to(device)

    # 损失
    loss_name = cfg["train"]["loss"].lower()
    loss_fn = build_loss(loss_name, cfg["train"].get("loss_params", {}))

    # 优化器
    optimizer = optim.Adam(model.parameters(), lr=float(cfg["train"]["lr"]))

    epochs = int(cfg["train"]["epochs"])
    num_negatives = int(cfg["train"]["num_negatives"])  # 用于 softmax 的候选近似与 BPR 的负例
    eval_k = int(cfg["train"]["eval_k"])  # 评估@K

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        for batch in train_loader:
            user_ids, pos_ids = [x.to(device) for x in batch]

            # 负采样
            neg_ids = neg_sampler.sample(user_ids, num_negatives).to(device)  # [B, Nn]

            # 打分
            pos_scores = model(user_ids, pos_ids)  # [B]
            neg_scores = model(user_ids.unsqueeze(1).expand(-1, num_negatives).reshape(-1), neg_ids.reshape(-1)).reshape(-1, num_negatives)

            # 若是 SL/SLatK，可选全量或采样集合。这里默认使用采样集合以提高速度，评估时走全量。
            loss = loss_fn(
                user_ids,
                pos_ids,
                pos_scores=pos_scores,
                neg_item_ids=neg_ids,
                neg_scores=neg_scores,
            )

            # 加 L2 正则
            reg = model.l2_regularization(user_ids, pos_ids, neg_ids)
            total_loss = loss + reg

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            epoch_loss += float(total_loss.detach().cpu())

        # 评估（全量打分）
        val_recall, val_ndcg = evaluate(model, val_dict, eval_k, device)
        test_recall, test_ndcg = evaluate(model, test_dict, eval_k, device)
        print(f"Epoch {epoch:03d} | loss={epoch_loss/len(train_loader):.4f} | val R@{eval_k}={val_recall:.4f} NDCG@{eval_k}={val_ndcg:.4f} | test R@{eval_k}={test_recall:.4f} NDCG@{eval_k}={test_ndcg:.4f}")


if __name__ == "__main__":
    main()

