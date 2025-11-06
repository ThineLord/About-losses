## Losses 使用说明（COMP5331 项目 - Ranking Loss 模块）

本模块提供三种排序损失：
- BPR (`losses.bpr.BPRLoss`)
- Softmax 基线 (`losses.sl.SoftmaxLoss`)
- SoftmaxLoss@K / SLatK（主方法，`losses.slatk.SoftmaxLossAtK`）

模块目标：支持全量打分与采样近似两种训练路径，统一对接模型/采样器/数据集。

### 1) 快速开始

```python
import torch
from losses import BPRLoss, SoftmaxLoss, SoftmaxLossAtK

loss_fn = SoftmaxLossAtK(topk=10, tau_d=0.5, tau_w=0.5)

# 训练步骤中：假设已得到 batch 的 user_ids, pos_item_ids, pos_scores, neg_scores
loss = loss_fn(
    user_ids=user_ids,               # [B]
    pos_item_ids=pos_item_ids,       # [B] 或 [B, P]
    pos_scores=pos_scores,           # [B] 或 [B, P]
    neg_scores=neg_scores,           # [B, Nn]
)
loss.backward()
```

若提供 `all_item_scores`（每个用户对“所有物品”的打分），SLatK/SL 会自动用全量集合计算归一化与 β^K（更准但更慢）：

```python
all_item_scores = model.full_item_scores(user_ids)  # [B, N]
loss = loss_fn(
    user_ids=user_ids,
    pos_item_ids=pos_item_ids,
    all_item_scores=all_item_scores,
)
```

### 2) 接口约定

所有损失类都继承自 `losses.base.BaseRankingLoss`，并共享 `forward` 形参：

- `user_ids: torch.Tensor [B]`
- `pos_item_ids: torch.Tensor [B] 或 [B, P]`
- `all_item_scores: Optional[torch.Tensor [B, N]]`（可选）
- `pos_scores: Optional[torch.Tensor [B] 或 [B, P]]`（可选）
- `neg_item_ids: Optional[torch.Tensor [B, Nn]]`（可选）
- `neg_scores: Optional[torch.Tensor [B, Nn]]`（可选）

使用方式：
- 若传入 `all_item_scores`，损失内部会优先使用全量打分进行计算（精确 Top-K 与归一化）。
- 否则需要传入 `pos_scores` 与 `neg_scores`（采样近似）。负例由采样器生成。

### 3) 三种损失说明

1) BPR（`BPRLoss`）
- 公式：`L = -E[log σ(s_ui - s_uj)]`
- 需要 `pos_scores` 与 `neg_scores`，内部对 (pos, neg) 做两两配对取平均。
- 适合作为经典 pairwise 基线。

2) Softmax 基线（`SoftmaxLoss`）
- 对每个正例 i：`L_i = log Σ_j exp((s_uj - s_ui)/τ_d)`
- 若给 `all_item_scores`，`j` 为全量物品；否则 `j` 为采样候选（pos+neg）。

3) SLatK（`SoftmaxLossAtK`，主方法）
- 在 Softmax 基线基础上，对每个 i 加权：`σ_w(s_ui - β_u^K)`，其中 `σ_w(x)=sigmoid(x/τ_w)`
- `β_u^K`（Top-K 分位点）：
  - 有 `all_item_scores` 时精确取 TopK 的第 K 大；
  - 无全量时用 (pos+neg) 的候选集合作为 MC 近似。
- 超参：`topk`, `tau_d`, `tau_w`。

### 4) Monte-Carlo β^K 估计（无全量打分时）

抽象基类提供工具函数：

```python
from losses.base import BaseRankingLoss

beta = BaseRankingLoss.estimate_topk_quantile(
    all_item_scores=None,
    pos_scores=pos_scores,  # [B] 或 [B, P]
    neg_scores=neg_scores,  # [B, Nn]
    topk=K,
)
```

策略：
- 若有 `all_item_scores`：精确 TopK，取第 K 大；
- 否则合并 `pos_scores` 与 `neg_scores`，在候选集合上取第 K 大（MC 近似）。

### 5) 与 sampler / model / dataloader 的对接

- 模型需要能：
  - 单点打分：`model(user_ids, item_ids) -> [B]`
  - 全量打分（评估或全量损失）：`model.full_item_scores(user_ids) -> [B, N]`

- 采样器需要能：
  - 根据 `user_ids` 产生 `neg_item_ids [B, Nn]`，并确保不与用户历史正例重叠。

- 典型训练（采样近似路径）：

```python
# user_ids, pos_ids from train loader
neg_ids = sampler.sample(user_ids, num_negatives)            # [B, Nn]
pos_scores = model(user_ids, pos_ids)                        # [B]
neg_scores = model(user_ids[:, None].expand(-1, num_negatives).reshape(-1),
                   neg_ids.reshape(-1)).reshape(-1, num_negatives)  # [B, Nn]

loss = loss_fn(user_ids, pos_ids, pos_scores=pos_scores, neg_scores=neg_scores)
loss.backward()
```

- 典型训练（全量路径 - 慎用，慢）：

```python
all_scores = model.full_item_scores(user_ids)  # [B, N]
loss = loss_fn(user_ids, pos_ids, all_item_scores=all_scores)
```

### 6) 在主 Pipeline 中的配置（参考）

在我们提供的最小可运行 demo（`train.py`）里，可通过 YAML 切换：

```yaml
train:
  loss: slatk   # 可选: bpr | sl | slatk
  loss_params:
    topk: 10
    tau_d: 0.5
    tau_w: 0.5
  num_negatives: 200
```

此外也可直接在 Pipeline 端构造：

```python
from losses import SoftmaxLossAtK
loss_fn = SoftmaxLossAtK(topk=10, tau_d=0.5, tau_w=0.5)
```

### 7) 常见问题（FAQ）

- Q: 只传 `pos_scores` 不传 `neg_scores` 可以吗？
  - A: 对 SL/SLatK 来说可以，但分母只包含正例自身，效果退化；建议至少提供若干负例的候选集合。

- Q: SLatK 的 β^K 估计需要多少负例？
  - A: 经验上 Nn=100~1000 之间通常能稳定近似（视数据规模而定）。过小会导致 β^K 偏大/偏小。

- Q: 计算慢怎么办？
  - A: 训练时用采样近似（传入 `pos_scores`/`neg_scores`）；评估再用全量打分。

- Q: 如何扩展新损失？
  - A: 继承 `BaseRankingLoss` 并实现 `forward`；可复用 `estimate_topk_quantile`，保持与现有接口一致以便主 Pipeline 复用。

### 8) 单元测试与验证

- `tests/test_losses.py` 提供两项验证：
  - Monte-Carlo β^K 估计误差相对精确 TopK 较小；
  - 人工数据上，执行一步更新后，SLatK 的 NDCG@K ≥ SL/BPR。

如需与更大规模数据/LightGCN/XSimGCL 集成，只需在训练循环中按上述接口提供 `pos_scores/neg_scores` 或 `all_item_scores` 即可。


