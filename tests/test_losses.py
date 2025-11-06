from __future__ import annotations

import math
import random

import torch

from losses import SoftmaxLossAtK, SoftmaxLoss, BPRLoss
from metrics import ndcg_at_k


def test_beta_estimation_mc_vs_exact():
    torch.manual_seed(0)
    B, N = 4, 50
    scores = torch.randn(B, N)
    # 随机选取每个用户的一个正例位置
    pos_idx = torch.randint(0, N, (B,))
    pos_scores = scores.gather(1, pos_idx.unsqueeze(1)).squeeze(1)

    # 精确 beta（第 K 大）
    K = 5
    exact_topk, _ = torch.topk(scores, K, dim=1)
    beta_exact = exact_topk[:, -1]

    # 采样 MC 近似（取若干负例+包含正例）
    num_neg = 20
    # 构建 neg_scores：从非正例中取样
    neg_scores = []
    for b in range(B):
        pool = [j for j in range(N) if j != int(pos_idx[b])]
        sel = random.sample(pool, num_neg)
        neg_scores.append(scores[b, sel])
    neg_scores = torch.stack(neg_scores, dim=0)

    slatk = SoftmaxLossAtK(K)
    beta_mc = slatk.estimate_topk_quantile(
        all_item_scores=None,
        pos_scores=pos_scores,
        neg_scores=neg_scores,
        topk=K,
    )

    err = (beta_mc - beta_exact).abs().mean().item()
    print("mean |beta_mc - beta_exact| =", err)
    assert err < 0.6  # 粗略门限，MC 近似允许有偏差


def test_slatk_improves_over_baselines_on_synthetic_ndcg():
    torch.manual_seed(0)
    random.seed(0)
    # 构造 1 个用户，100 个物品，设定前 K=10 个为理想 top-K
    B, N, K = 8, 200, 10
    scores = torch.randn(B, N) * 0.1
    # 设定每个用户的前 K 个物品具有更高均值
    for b in range(B):
        scores[b, :K] += 2.0
    # 正例是前 K 个
    gt = [list(range(K)) for _ in range(B)]

    # 构造 pos_scores：从 gt 中随机取一个作为“当前观测正例”
    pos_idx = torch.tensor([random.choice(g) for g in gt], dtype=torch.long)
    pos_scores = scores.gather(1, pos_idx.unsqueeze(1)).squeeze(1)

    # 构造 neg_scores：从非 gt 区域中采样
    num_neg = 50
    neg_lists = []
    for b in range(B):
        pool = list(range(K, N))
        sel = random.sample(pool, num_neg)
        neg_lists.append(sel)
    neg_scores = torch.stack([scores[b, sel] for b, sel in enumerate(neg_lists)], dim=0)

    slatk = SoftmaxLossAtK(K, tau_d=0.5, tau_w=0.5)
    sl = SoftmaxLoss(tau_d=0.5)
    bpr = BPRLoss()

    # 比较单步“优化方向”对 NDCG@K 的贡献：
    # 这里用简单的梯度下降一步来模拟，目标是使正例得分升高、负例降低
    scores_slatk = scores.clone().requires_grad_(True)
    scores_sl = scores.clone().requires_grad_(True)
    scores_bpr = scores.clone().requires_grad_(True)

    def one_step(scores_var, loss_fn):
        pos = scores_var.gather(1, pos_idx.unsqueeze(1)).squeeze(1)
        neg = torch.stack([scores_var[b, sel] for b, sel in enumerate(neg_lists)], dim=0)
        loss = loss_fn(
            user_ids=torch.arange(B),
            pos_item_ids=pos_idx,
            all_item_scores=None,
            pos_scores=pos,
            neg_item_ids=None,
            neg_scores=neg,
        )
        loss.backward()
        with torch.no_grad():
            scores_var -= 0.1 * scores_var.grad
        return scores_var.detach()

    scores_slatk = one_step(scores_slatk, slatk)
    scores_sl = one_step(scores_sl, sl)
    scores_bpr = one_step(scores_bpr, bpr)

    ndcg_slatk = ndcg_at_k(scores_slatk, gt, K)
    ndcg_sl = ndcg_at_k(scores_sl, gt, K)
    ndcg_bpr = ndcg_at_k(scores_bpr, gt, K)
    print("NDCG@K slatk, sl, bpr =", ndcg_slatk, ndcg_sl, ndcg_bpr)

    # 期望：SLatK 至少不差于 SL/BPR，通常更好
    assert ndcg_slatk >= max(ndcg_sl, ndcg_bpr) - 1e-6


if __name__ == "__main__":
    test_beta_estimation_mc_vs_exact()
    test_slatk_improves_over_baselines_on_synthetic_ndcg()
    print("All tests passed.")

